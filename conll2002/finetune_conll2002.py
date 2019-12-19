"""
Using the so-called "sliding window" approach
"""
import os
import sys
import random
import argparse
import pprint
import json
from platform import node
from datetime import datetime
from operator import itemgetter, attrgetter
from itertools import zip_longest
from dataclasses import dataclass
from typing import List

import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score

import torch
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import logging

from transformers import (
    BertForTokenClassification,
    BertTokenizer,
    BertConfig,
    get_linear_schedule_with_warmup,
)

logger = logging.getLogger(__name__)


LABEL_LIST = ('B-PER', 'B-MISC', 'I-ORG', 'B-ORG',
              'I-LOC', 'B-LOC', 'I-PER', 'O', 'I-MISC')

LABEL_MAP = {label: i for i, label in enumerate(LABEL_LIST)}
INWORD_PAD_LABEL = "PAD"
LABEL_MAP[INWORD_PAD_LABEL] = -100  # adds a value for the added PAD label

TRAIN_FILE = "esp.train"
DEV_FILE = "esp.testa"


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


@dataclass
class InputExample:
    """
    This class should contain words already tokenized to assure
    the max length is respected
    """
    tokens_a: List[str]
    tokens_b: List[str]
    labels_a: List[str]
    labels_b: List[str]


@dataclass
class InputFeature:
    input_ids: List[int]
    attention_mask: List[int]
    token_type_ids: List[int]
    labels: List[int]

    def __post_init__(self):
        # print("Post init!!")
        # breakpoint()
        assert len(set(map(len, vars(self).values()))) == 1


class CoNLL2002Processor():
    def __init__(self, tokenizer, sequence_length=128):
        self.sequence_length = sequence_length
        self.tokenizer = tokenizer

    def get_dev_examples(self, data_dir):
        words, labels = self.read_file(os.path.join(data_dir, DEV_FILE))
        return self.build_examples(words, labels)

    def get_train_examples(self, data_dir):
        # TODO: cambiar al train set real
        words, labels = self.read_file(os.path.join(data_dir, TRAIN_FILE))
        return self.build_examples(words, labels)

    def build_examples(self, words, labels):
        """
        Build examples using the sliding window aproach
        Instead of converting the labels directly to their corresponding
        value, I add a new label called PAD and use this for the subwords
        created by the tokenizer. This is done for easier debugging.

        In order to account for the added tokens to te input sequence,
        the actual length of an example should be sequence_length - 3:
        [CLS] tok11 tok12 ... [SEP] tok21 tok22 ... [SEP]
        """
        sentence_length = (self.sequence_length - 3) // 2

        logger.info("Tokenizing words")
        tokens, labels = zip(*[
            (token, label)
            for word, o_label
            in tqdm(zip(words, labels), desc="Word", total=len(words))
            for token, label
            in zip_longest(
                self.tokenizer.tokenize(word),
                [o_label],
                fillvalue=INWORD_PAD_LABEL
            )
        ])

        logger.info("Building examples")
        sentence_starts = range(0, len(tokens), sentence_length)
        examples = [
            InputExample(
                tokens_a=tokens[start_a:start_b],
                tokens_b=tokens[start_b:start_b+sentence_length],
                labels_a=labels[start_a:start_b],
                labels_b=labels[start_b:start_b+sentence_length]
            )
            for start_a, start_b
            in tqdm(
                zip(sentence_starts[:-1], sentence_starts[1:]),
                total=len(sentence_starts)-1,
                desc="Example"
            )
        ]

        assert all(
            e1.tokens_b == e2.tokens_a
            for e1, e2 in zip(examples[:-1], examples[1:]))

        # breakpoint()
        return examples

    @classmethod
    def read_file(self, file_name):
        with open(file_name, "r") as file:
            return list(zip(*[
                itemgetter(0, 2)(line.split())
                for line
                in file
                if line != "\n"
            ]))


def examples2features(args, examples, tokenizer):

    logger.info("Converting examples to features")
    features = []
    for example in tqdm(examples, desc="Feature"):
        # Each input feature should consist of a list o token ids,
        # an attention mask, a list of token type ids, and optionally
        # a label or (in this case) a list of labels
        results = tokenizer.prepare_for_model(
            tokenizer.convert_tokens_to_ids(example.tokens_a),
            tokenizer.convert_tokens_to_ids(example.tokens_a),
            max_length=args.max_seq_len,
            pad_to_max_length=True,
            return_token_type_ids=True,
            return_attention_mask=True,
            return_overflowing_tokens=True,
        )
        labels = [
            LABEL_MAP[INWORD_PAD_LABEL],  # CLS
            *[LABEL_MAP[label] for label in example.labels_a],
            LABEL_MAP[INWORD_PAD_LABEL],  # SEP
            *[LABEL_MAP[label] for label in example.labels_b],
            LABEL_MAP[INWORD_PAD_LABEL],  # SEP
        ]
        labels += [LABEL_MAP[INWORD_PAD_LABEL]] \
            * (args.max_seq_len - len(labels))

        # Verify that the length of the padded tokens is correct
        assert len(labels) == args.max_seq_len
        # Verify there were no overflowing tokens
        assert "overflowing_tokens" not in results

        features.append(InputFeature(**results, labels=labels))

    idx = random.randrange(len(features))
    logger.info("******** Random example feature ********")
    logger.info(f"Sentence A: {examples[idx].tokens_a}")
    logger.info(f"Labels A: {examples[idx].labels_a}")
    logger.info(f"Sentence B: {examples[idx].tokens_b}")
    logger.info(f"Labels B: {examples[idx].labels_b}")
    logger.info(f"input_ids: {features[idx].input_ids}")
    logger.info(f"labels: {features[idx].labels}")
    logger.info(f"attention_mask: {features[idx].attention_mask}")
    logger.info(f"token_type_ids: {features[idx].token_type_ids}")
    return features


def load_dataset(args, processor, tokenizer, evaluate=False):
    cache_file = os.path.join(
        args.data_dir,
        "cached_dataset_beto_{}_conll2002_es_{}_{}".format(
            "uncased" if args.do_lower_case else "cased",
            "dev" if evaluate else "train",
            args.max_seq_len,
        )
    )
    # breakpoint()
    if os.path.exists(cache_file) and not args.overwrite_cache:
        logger.info(f"Loading dataset from cached file at {cache_file}")
        dataset = torch.load(cache_file)
    else:
        logger.info(f"Creating features from dataset file at {args.data_dir}")
        # Im just partially sorry for this :D
        examples = (
            processor.get_dev_examples
            if evaluate else
            processor.get_train_examples
        )(args.data_dir)
        features = examples2features(
            args,
            examples,
            tokenizer,
        )
        # Im sorry for this :D
        getter = attrgetter(
            "input_ids", "attention_mask", "token_type_ids", "labels"
        )
        # breakpoint()
        tensors = map(torch.tensor, zip(*[getter(f) for f in features]))
        dataset = TensorDataset(*tensors)
        logger.info(f"Saving dataset into cached file {cache_file}")
        torch.save(dataset, cache_file)

    return dataset


def train(args, dataset, model):
    tb_writer = SummaryWriter(
        log_dir="runs/conll2002_{}_{}".format(
            datetime.now().strftime('%b%d_%H-%M'),
            node()
        )
    )

    # Apparently weight decay should not aply to bias and normalization layers
    # list of two dicts, one where the
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
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    total_steps = len(dataloader) * args.epochs
    optimizer = Adam(
        grouped_parameters,
        lr=args.learn_rate,
        weight_decay=args.weight_decay
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(args.warmup*total_steps),  # warmup is a percentage
        num_training_steps=total_steps,
    )
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(dataset)}")
    logger.info(f"  Num Epochs = {args.epochs}")
    logger.info(f"  Train batch size = {args.batch_size}")
    logger.info(f"  Total optimization steps = {total_steps}")

    global_step = 0
    tr_loss, running_loss = 0., 0.
    for _ in tqdm(range(args.epochs), desc="Epoch"):
        for step, batch in enumerate(tqdm(dataloader, desc="Iteration")):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)

            model.zero_grad()

            loss = model(
                input_ids=batch[0],
                attention_mask=batch[1],
                token_type_ids=batch[2],
                labels=batch[3]
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


def evaluate(args, model, dataset, processor):

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    logger.info("***** Running evaluation *****")
    logger.info(f"  Num examples = {len(dataset)}")
    logger.info(f"  Batch size = {args.batch_size}")
    # breakpoint()
    # preds is always on cpu
    preds = torch.tensor([])
    for idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            scores = model(
                input_ids=batch[0],
                attention_mask=batch[1],
                token_type_ids=batch[2],
                # labels=batch[3]
            )[0]
            labels = batch[3]
            # breakpoint()
            if idx == 0:
                # Only for the first example add the first sentence
                half_len = args.max_seq_len // 2
                positions = labels[0, :half_len] != LABEL_MAP[INWORD_PAD_LABEL]
                preds = torch.cat([
                    preds,
                    scores[0, :half_len][positions].cpu()
                ])
                # breakpoint()
            # For every example add the second sentence
            breakpoint()
            half_len = args.max_seq_len // 2
            positions = labels[:, half_len:] != LABEL_MAP[INWORD_PAD_LABEL]
            preds = torch.cat([
                preds,
                scores[:, half_len:][positions].cpu()
            ])

    # Read the gold labels directly from file to verify dimensions are ok
    # breakpoint()
    gold_labels = [
        LABEL_MAP[label]
        for label
        in processor.read_file(os.path.join(args.data_dir, DEV_FILE))[1]
    ]
    # breakpoint()
    assert len(preds) == len(gold_labels)

    score = f1_score(
        gold_labels,
        preds.argmax(dim=1),
        labels=[LABEL_MAP[label] for label in LABEL_LIST if label != "O"],
        average="micro",
    )
    # breakpoint()
    return {"f1_score": score}


def main(passed_args=None):
    parser = argparse.ArgumentParser()

    # TODO: add required
    # Required
    parser.add_argument("--model-dir", default="../beto-uncased", type=str)
    parser.add_argument("--data-dir", default="./data", type=str)
    parser.add_argument("--output-dir", default="./outputs", type=str)

    # Hyperparams to perform search on
    parser.add_argument("--learn-rate", default=5e-5, type=float)
    parser.add_argument("--batch-size", default=16, type=int)
    parser.add_argument("--epochs", default=3, type=int)

    # Hyperparams that where relatively common
    parser.add_argument("--max-seq-len", default=128, type=int)
    parser.add_argument("--weight-decay",  default=0.01, type=float)
    parser.add_argument("--warmup", default=0.1, type=float,
                        help="Percentage of warmup steps. In range [0, 1]")

    # General options
    parser.add_argument("--do-lower-case", action="store_true")
    parser.add_argument("--overwrite-cache", action="store_true")
    parser.add_argument("--overwrite-output-dir", action="store_true")
    parser.add_argument("--disable-cuda", action="store_true")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--logging-steps", default=50, type=int)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")

    args = parser.parse_args(passed_args)
    if (
            os.path.exists(args.output_dir)
            and os.listdir(args.output_dir)  # verifica vacio
            and not args.skip_train
            and not args.overwrite_output_dir
       ):
        raise ValueError(
            f"Output dir ({args.output_dir}) already exists and is not empty. "
            "Please use --overwrite-output-dir"
        )
    # Check this in case someone forgets to add the option
    if "uncased" in args.model_dir and not args.do_lower_case:
        option = input(
            "WARNING: --model-dir contains 'uncased' but got no "
            "--do-lower-case option.\nDo you want to continue? [Y/n] ")
        if option == "n":
            sys.exit(0)

    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device("cuda")
        args.n_gpu = torch.cuda.device_count()
    else:
        args.device = torch.device("cpu")
        args.n_gpu = 0
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO
    )
    # breakpoint()
    # ****************** Set the seed *********************
    set_seed(args)

    if not args.skip_train:
        # ****************** Load model ***********************

        tokenizer = BertTokenizer.from_pretrained(
            args.model_dir,
            do_lower_case=args.do_lower_case
        )
        processor = CoNLL2002Processor(
            tokenizer, sequence_length=args.max_seq_len)

        config = BertConfig.from_pretrained(
            args.model_dir,
            num_labels=len(LABEL_LIST),
            finetuning_task="conll2002",
        )
        model = BertForTokenClassification.from_pretrained(
            args.model_dir,
            config=config,
        ).to(args.device)

        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)
        train_dataset = load_dataset(
            args, processor, tokenizer, evaluate=False)
        # Train
        global_step, tr_loss = train(args, train_dataset, model)
        logger.info(f" global_step = {global_step}, average loss = {tr_loss}")

        # ****************** Save fine-tuned model ************
        os.makedirs(args.output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {args.output_dir}")
        model_to_save = (
            model.module
            if isinstance(model, torch.nn.DataParallel) else
            model)
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        torch.save(args, os.path.join(args.output_dir, "train_args.bin"))

    if not args.skip_eval:
        # load saved if training was skipped
        if args.skip_train:
            # TODO: cargar solo los args relevantes para entrenamiento
            # y no las opciones generales
            args = torch.load(
                os.path.join(args.output_dir, "train_args.bin"))
            model = BertForTokenClassification.from_pretrained(
                args.output_dir)
            tokenizer = BertTokenizer.from_pretrained(
                args.output_dir, do_lower_case=args.do_lower_case)
            model.to(args.device)
            processor = CoNLL2002Processor(
                tokenizer, sequence_length=args.max_seq_len)

            if args.n_gpu > 1:
                model = torch.nn.DataParallel(model)

        eval_dataset = load_dataset(args, processor, tokenizer, evaluate=True)
        results = evaluate(args, model, eval_dataset, processor)
        # ********************* Save results ******************
        logger.info(f"Saving results to {args.output_dir}/dev_results.json")
        logger.info(f"Training args are saved to {args.output_dir}/"
                    "train_args.json")
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        with open(os.path.join(args.output_dir, "dev_results.json"), "w") as f:
            json.dump(results, f)
        with open(os.path.join(args.output_dir, "train_args.json"), "w") as f:
            json.dump({**vars(args), "device": repr(args.device)}, f)
        print(results)

    # breakpoint()


if __name__ == '__main__':
    main()
