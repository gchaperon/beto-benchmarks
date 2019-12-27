"""
This time i wont use the sliding window aproach, just pass in a whole
sentence, maybe that is the problem with the other script


This script doesn't work, without the sliding window approach, the
sentence length must be to long and my GPU runs out of memory

If the max sentence length is to small, some examples will be truncated
and some labels will be lost
"""
import argparse
import logging
import os
import random
import sys
from dataclasses import dataclass
from datetime import datetime
from itertools import zip_longest
from operator import itemgetter, attrgetter
from platform import node
from typing import List

import numpy as np
from sklearn.metrics import f1_score

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from transformers import (
    BertTokenizer,
    BertConfig,
    BertForTokenClassification,
    get_linear_schedule_with_warmup,
)
from tqdm import tqdm



logger = logging.getLogger(__name__)

# Large sentence length in order to avoid losing tokens at evaluation
MIN_EVAL_LENGTH = 350
LABEL_LIST = ['O', 'I-LOC', 'I-MISC', 'B-ORG', 'B-PER',
              'I-ORG', 'B-MISC', 'B-LOC', 'I-PER']

LABEL_MAP = {label: i for i, label in enumerate(LABEL_LIST)}
IGNORE_INDEX = torch.nn.CrossEntropyLoss().ignore_index


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


# The type anotations are only so the parser doesn't complain
@dataclass
class InputExample:
    sentence: List[str]
    labels: List[str]


@dataclass
class InputFeature:
    input_ids: List[int]
    attention_mask: List[int]
    token_type_ids: List[int]
    labels: List[int]
    length: int

    def __post_init__(self):
        # breakpoint()
        assert all([
            len(attr) == self.length
            for attr in [
                self.input_ids,
                self.attention_mask,
                self.token_type_ids,
                self.labels
            ]
        ])


def load_examples(args, eval_=True):
    logger.info("Reading examples")
    path = os.path.join(args.data_dir, "esp.testa" if eval_ else "esp.train")
    with open(path, "r") as file:
        lines = file.readlines()

    new_lines = [i for i, line in enumerate(lines) if line == "\n"]
    groups = [
        lines[prev_newline + 1: next_newline]
        for prev_newline, next_newline
        in zip([-1] + new_lines[:-1], new_lines)
    ]
    examples = [
        InputExample(
            *zip(*[itemgetter(0, 2)(line.split()) for line in group])
        )
        for group
        in groups
    ]
    logger.info("Done")
    return examples


def examples2features(examples, tokenizer, max_length):
    """
    Each sentence will be converted to an InputFeature and truncated if
    it is to long.
    """

    features = []
    n_overflowing_examples = 0
    logger.info("Converting examples to features")
    for example in tqdm(examples):
        tokens, labels = [], []
        for word, w_label in zip(example.sentence, example.labels):
            for token, t_label_id in zip_longest(
                    tokenizer.tokenize(word),
                    [LABEL_MAP[w_label]],
                    fillvalue=IGNORE_INDEX):
                tokens.append(token)
                labels.append(t_label_id)

        outputs = tokenizer.prepare_for_model(
            ids=tokenizer.convert_tokens_to_ids(tokens),
            max_length=max_length,
            pad_to_max_length=True,
            return_overflowing_tokens=True,
        )

        if "overflowing_tokens" in outputs:
            n_overflowing_examples += 1

        # pad labels
        if len(labels) < max_length:
            labels += [IGNORE_INDEX] * (max_length - len(labels))
        else:
            labels = labels[:max_length]

        feature = InputFeature(
            *itemgetter(
                "input_ids",
                "attention_mask",
                "token_type_ids"
            )(outputs),
            labels,
            max_length,
        )
        # breakpoint()

        features.append(feature)

    logger.info(f"Number of truncated examples: {n_overflowing_examples}")
    # breakpoint()
    return features, n_overflowing_examples


def load_dataset(args, tokenizer, eval_=False):
    """
    Some day i will implement the logic for saving to a cache file
    and loading from it
    """
    examples = load_examples(args, eval_=eval_)

    # This is to prevent some examples lacking predictions if they are
    # to long.
    # I belive there are no examples in the evaluation sets
    # that after tokenization are longer than 512
    if eval_:
        max_length = MIN_EVAL_LENGTH if args.max_length < MIN_EVAL_LENGTH else args.max_length
    else:
        max_length = args.max_length

    features, n_overflowing_examples = examples2features(
        examples, tokenizer, max_length)

    # when evaluating the shouldn't be any overflowing examples
    if eval_:
        assert n_overflowing_examples == 0

    dataset = TensorDataset(
        torch.tensor([f.input_ids for f in features]),
        torch.tensor([f.attention_mask for f in features]),
        torch.tensor([f.token_type_ids for f in features]),
        torch.tensor([f.labels for f in features]),
    )
    # breakpoint()
    return dataset


def train(args, model, dataset):
    tb_writer = SummaryWriter(
        log_dir="runs/conll2002_{}_{}".format(
            datetime.now().strftime('%b%d_%H-%M'),
            node()
        )
    )
    # Apparently weight decay should not aply to bias and normalization layers
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    total_steps = len(dataloader) * args.epochs

    no_decay = ['bias', 'LayerNorm.weight']
    grouped_parameters = [
        {
            'params': [
                p
                for n, p
                in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            'weight_decay': 0.0
        },
        {
            'params': [
                p
                for n, p
                in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            'weight_decay': args.weight_decay
        },
    ]
    optimizer = Adam(
        grouped_parameters,
        lr=args.learn_rate,
        weight_decay=args.weight_decay
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(args.warmup_steps*total_steps),  # warmup is a %
        num_training_steps=total_steps,
    )

    global_step, tr_loss, running_loss = 0, 0., 0.

    for _ in tqdm(range(args.epochs), desc="Epoch"):
        for step, batch in enumerate(tqdm(dataloader, desc="Iteration")):
            model.train()
            batch = [t.to(args.device) for t in batch]

            model.zero_grad()

            loss = model(
                input_ids=batch[0],
                attention_mask=batch[1],
                token_type_ids=batch[2],
                labels=batch[3],
            )[0]
            if args.n_gpu > 1:
                loss = loss.mean()

            loss.backward()
            optimizer.step()
            scheduler.step()

            tr_loss += loss.item()
            running_loss += loss.item()

            if (args.logging_steps > 0
                    and global_step % args.logging_steps == 0):
                tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                tb_writer.add_scalar(
                    "loss", running_loss/args.logging_steps, global_step)
                running_loss = 0.

            global_step += 1

    tb_writer.close()

    return global_step, tr_loss / global_step


def evaluate(args, model, dataset, gold_labels):

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False
    )

    # predictions should always be on cpu
    predictions = torch.tensor([])
    for batch in tqdm(dataloader, desc="Evaluating"):
        model.eval()
        model.to(args.device)

        labels = batch[3]
        batch = [t.to(args.device) for t in batch[:3]]
        with torch.no_grad():
            scores = model(
                input_ids=batch[0],
                attention_mask=batch[1],
                token_type_ids=batch[2],
            )[0]

            positions = labels != IGNORE_INDEX
            # breakpoint()
            predictions = torch.cat([predictions, scores.cpu()[positions]])

    assert len(predictions) == len(gold_labels), \
        f"{len(predictions)} != {len(gold_labels)}"
    breakpoint()
    score = f1_score(
        [LABEL_MAP[label] for label in gold_labels],
        predictions.argmax(dim=1),
        labels=[
            LABEL_MAP[label]
            for label in LABEL_LIST
            if label != IGNORE_INDEX
        ],
        average="micro",
    )
    return {"f1_score": score}


def main(passed_args=None):

    # TODO: cambiar los default
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="../beto-cased", type=str)
    parser.add_argument("--data-dir", default="./data", type=str)

    parser.add_argument("--learn-rate", default=3e-5, type=float)
    parser.add_argument("--epochs", default=3, type=int)
    parser.add_argument("--batch-size", default=4, type=int)

    parser.add_argument("--max-length", default=512, type=int)
    parser.add_argument("--weight-decay", default=0.01, type=float)
    parser.add_argument("--warmup-steps", default=0.1, type=float)

    parser.add_argument("--disable-cuda", action="store_true")
    parser.add_argument("--do-lower-case", action="store_true")
    parser.add_argument("--remove-accents", action="store_false",
                        dest="keep_accents")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--logging-steps", default=50, type=int)

    args = parser.parse_args()

    # In case i forgot to add the flag
    if "uncased" in args.model_dir and not args.do_lower_case:
        option = input(
            "WARNING: --model-dir contains 'uncased' but got no "
            "--do-lower-case option.\nDo you want to continue? [Y/n] ")
        if option == "n":
            sys.exit(0)

    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO
    )

    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device("cuda")
        args.n_gpu = torch.cuda.device_count()
    else:
        args.device = torch.device("cpu")
        args.n_gpu = 0

    tokenizer = BertTokenizer.from_pretrained(
        args.model_dir,
        do_lower_case=args.do_lower_case,
        keep_accents=args.keep_accents,
    )
    config = BertConfig.from_pretrained(
        args.model_dir,
        finetuning_task="conll2002",
        num_labels=len(LABEL_LIST),
    )
    model = BertForTokenClassification.from_pretrained(
        args.model_dir,
        config=config,
    ).to(args.device)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Train !
    # train_dataset = load_dataset(args, tokenizer, eval_=False)

    # global_step, tr_loss = train(args, model, train_dataset)

    # Evaluate !
    # Get the gold labesl from the original file

    gold_labels = [
        line.split()[2]
        for line in open(os.path.join(args.data_dir, "esp.testa"))
        if line != "\n"
    ]
    eval_dataset = load_dataset(args, tokenizer, eval_=True)
    score = evaluate(args, model, eval_dataset, gold_labels)
    print(score)



if __name__ == '__main__':
    main()