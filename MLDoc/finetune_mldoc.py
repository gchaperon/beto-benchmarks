#!/usr/bin/env python3
import os
import csv
import sys
import json
import random
import logging
import argparse
from transformers import (
    BertTokenizer,
    BertConfig,
    BertForSequenceClassification,
    DataProcessor,
    InputExample,
    InputFeatures,
    get_linear_schedule_with_warmup,
)
from tqdm import tqdm
from operator import itemgetter, attrgetter
from itertools import islice
import torch
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import nltk.data
import numpy as np

logger = logging.getLogger(__name__)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


class MLDocProcessor(DataProcessor):
    def __init__(self, n_train=10000):
        self.n_train = n_train

    def get_train_examples(self, data_dir):
        rows = self._read_tsv(os.path.join(
            data_dir,
            f"mldoc.es.train.{self.n_train}"
        ))
        return self._rows2examples(rows)

    def get_dev_examples(self, data_dir):
        rows = self._read_tsv(os.path.join(
            data_dir,
            f"mldoc.es.dev"
        ))
        return self._rows2examples(rows)

    def _rows2examples(self, rows):
        # Spanish sentence tokenizer
        tokenizer = nltk.data.load("tokenizers/punkt/PY3/spanish.pickle")
        examples = []
        logger.info("Reading examples")
        for i, row in enumerate(tqdm(rows)):
            # the text column was saved as a string with the python syntax
            # for bytes literals, so it must be converted to a string literal

            tokens = tokenizer.tokenize(eval(row[1]).decode())
            example = InputExample(
                f"test-{i}",
                tokens[0],
                tokens[1] if len(tokens) > 1 else None,
                label=row[0]
            )
            examples.append(example)

        return examples

    def get_labels(self):
        return ["CCAT", "MCAT", "ECAT", "GCAT"]

    def _read_tsv(self, fpath):
        with open(fpath, "r") as file:
            reader = csv.reader(
                file,
                quoting=csv.QUOTE_NONE,
                delimiter="\t"
            )
            return list(reader)


def examples2features(examples, tokenizer, label_list, max_length=128):
    label_map = {label: i for i, label in enumerate(label_list)}

    logger.info("Converting examples to features")
    features = []
    for ex_index, example in enumerate(tqdm(examples)):
        inputs = tokenizer.encode_plus(
            example.text_a,
            example.text_b,
            add_special_tokens=True,
            max_length=max_length
        )

        # Im so sorry for this xD
        (
            input_ids,
            token_type_ids
        ) = itemgetter(
            "input_ids",
            "token_type_ids"
        )(inputs)
        attention_mask = [1] * len(input_ids)

        # Pad everything
        pad_token = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
        padding_length = max_length - len(input_ids)

        input_ids = input_ids + ([pad_token] * padding_length)
        token_type_ids = token_type_ids + ([0] * padding_length)
        attention_mask = attention_mask + ([0] * padding_length)

        # Assert that everything was padded correctly
        assert len(input_ids) == max_length
        assert len(token_type_ids) == max_length
        assert len(attention_mask) == max_length

        features.append(InputFeatures(
            input_ids,
            attention_mask,
            token_type_ids,
            label=label_map[example.label]
        ))

    # Log some examples to check
    for example, feature in islice(zip(examples, features), 5):
        logger.info("******** Example ********")
        logger.info(f"Guid: {example.guid}")
        logger.info(f"Sentence A: {example.text_a}")
        logger.info(f"Sentence B: {example.text_b}")
        logger.info(f"input_ids: {feature.input_ids}")
        logger.info(f"attention_mask: {feature.attention_mask}")
        logger.info(f"token_type_ids: {feature.token_type_ids}")
        logger.info(f"label: {example.label} (id = {feature.label})")

    return features


def load_dataset(args, processor, tokenizer, evaluate=False):
    """
    To load a dataset we need its location (should be in args), a way
    to process the raw data and a way to tokenize it and convert it
    to features.
    Different datasets should be produced in case we are evaluating or not.
    """

    cache_file = os.path.join(
        args.data_dir,
        "cached_features_beto_{}_mldoc{}_es_{}_{}".format(
            "uncased" if args.do_lower_case else "cased",
            "" if evaluate else processor.n_train,
            "dev" if evaluate else "train",
            args.max_seq_len
        ),
    )

    if os.path.exists(cache_file) and not args.overwrite_cache:
        logger.info(f"Loading features from cached file at {cache_file}")
        features = torch.load(cache_file)
    else:
        logger.info(f"Creating features from dataset file at {args.data_dir}")
        # Im just partially sorry for this :D
        examples = (
            processor.get_dev_examples
            if evaluate else
            processor.get_train_examples
        )(args.data_dir)
        features = examples2features(
            examples,
            tokenizer,
            processor.get_labels(),
            max_length=args.max_seq_len,
        )
        # Save features to cache file
        logger.info(f"Saving features into cached file {cache_file}")
        torch.save(features, cache_file)

    # Im sorry for this :D
    getter = attrgetter(
        "input_ids", "attention_mask", "token_type_ids", "label"
    )
    tensors = map(torch.tensor, zip(*[getter(f) for f in features]))
    dataset = TensorDataset(*tensors)
    return dataset


def train(args, dataset, model):
    tb_writer = SummaryWriter()

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
    logger.info(f"  Learn rate = {args.learn_rate}")
    logger.info(f"  Num Epochs = {args.epochs}")
    logger.info(f"  Train batch size = {args.batch_size}")
    logger.info(f"  Total optimization steps = {total_steps}")

    global_step = 0
    tr_loss, running_loss = 0., 0.
    for _ in tqdm(range(args.epochs), desc="Epoch"):
        for step, batch in enumerate(tqdm(dataloader, desc="Iteration")):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)

            optimizer.zero_grad()

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


def evaluate(args, model, dataset):

    dataloader = DataLoader(dataset, batch_size=args.batch_size)

    logger.info("***** Running evaluation *****")
    logger.info(f"  Num examples = {len(dataset)}")
    logger.info(f"  Batch size = {args.batch_size}")

    # preds is always on cpu
    preds = torch.tensor([])
    gold_labels = torch.tensor([], dtype=torch.long)
    for batch in tqdm(dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            logits = model(
                input_ids=batch[0],
                attention_mask=batch[1],
                token_type_ids=batch[2],
                # labels=batch[3]
            )[0]
            preds = torch.cat([preds, logits.cpu()])
            gold_labels = torch.cat([gold_labels, batch[3].cpu()])

    correct = (preds.argmax(dim=1) == gold_labels).sum().item()
    return {"acc": correct/len(preds)}


def main(passed_args=None):
    parser = argparse.ArgumentParser()

    # Required
    parser.add_argument("--model-dir", default=None, type=str, required=True)
    parser.add_argument("--data-dir", default=None, type=str, required=True)
    parser.add_argument("--output-dir", default=None, type=str, required=True)

    # Hyperparams to perform search on
    parser.add_argument("--learn-rate", default=5e-5, type=float)
    parser.add_argument("--batch-size", default=16, type=int)
    parser.add_argument("--epochs", default=3, type=int)

    # Hyperparams that where relatively common
    parser.add_argument("--max-seq-len", default=128, type=int)
    parser.add_argument("--weight-decay",  default=0.01, type=float)
    parser.add_argument("--warmup", default=0.1, type=float,
                        help="Percentage of warmup steps. In range [0, 1]")

    # Specific params
    parser.add_argument("--train-size", default=10000, type=int)

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

    args.device = None
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device("cuda")
        args.n_gpu = torch.cuda.device_count()
    else:
        args.device = torch.device("cpu")
        args.n_gpu = 0

    print(args)
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO
    )
    # ****************** Set the seed *********************
    set_seed(args)

    # ***************** Train *****************
    processor = MLDocProcessor(n_train=args.train_size)
    if not args.skip_train:
        # ****************** Load model ***********************
        config = BertConfig.from_pretrained(
            args.model_dir,
            num_labels=len(processor.get_labels()),
            finetuning_task="mldoc",
        )
        tokenizer = BertTokenizer.from_pretrained(
            args.model_dir,
            do_lower_case=args.do_lower_case
        )
        model = BertForSequenceClassification.from_pretrained(
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

    # *************** Evaluation *******************
    if not args.skip_eval:
        # load saved if training was skipped
        if args.skip_train:
            args = torch.load(
                os.path.join(args.output_dir, "train_args.bin"))
            model = BertForSequenceClassification.from_pretrained(
                args.output_dir)
            tokenizer = BertTokenizer.from_pretrained(
                args.output_dir, do_lower_case=args.do_lower_case)
            model.to(args.device)
            if args.n_gpu > 1:
                model = torch.nn.DataParallel(model)

        eval_dataset = load_dataset(args, processor, tokenizer, evaluate=True)
        results = evaluate(args, model, eval_dataset)
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

    return results


if __name__ == '__main__':
    main()
