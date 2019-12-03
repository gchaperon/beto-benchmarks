#!/usr/bin/env python3

"""
Everything here is heavily inspired by the run_xnli.py example of the
transformers package

This script does:
1. Read and process de XNLI data
2. Convert the data into features
3. Create a dataset object using said features
4. Load the beto model
5. Train it
6. Evaluate it using the dev data

Author: Gabriel Chaperon
"""

import os
import csv
import random
import argparse
import pprint
from collections import OrderedDict

import torch
import numpy as np

from transformers import (
    DataProcessor,
    InputExample,
    # aparentemente esta funcion es media agnostica a la tarea
    glue_convert_examples_to_features as convert_examples_to_features
)


class XNLIProcessor(DataProcessor):
    """
    Processor for the XNLI dataset. This is a modified version
    of the one available in transformers/data/processors/xnli.py.
    Which in turn was adapted from
    https://github.com/google-research/bert/blob/f39e881b169b9d53bea03d2d341b31707a6c052b/run_classifier.py#L207

    But Gabriel! Why would you repeat the same thing and not just use it ??!!
    Well cos i'm a dense motherfucker and I dislike the coding style that
    that file has. Furthermore, it's super hard for me to understand code
    by just reading code, so I prefer repeating some parts while modifying
    others
    """

    # RECORDAR IGNORAR LA PRIMERA LINEA !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    def __init__(self, train_language, dev_language=None,
                 test_language=None, cased=True):
        self.train_language = train_language
        self.dev_language = dev_language if dev_language else train_language
        self.test_language = test_language if test_language else train_language
        self.cased = cased

    def get_train_examples(self, data_dir):
        rows = self._read_xnli_tsv(
            os.path.join(
                data_dir,
                "XNLI-MT-1.0",
                "multinli",
                f"multinli.train.{self.train_language}.tsv"
            ))
        examples = [
            InputExample(
                guid=f"train-{i}",
                text_a=row["premise"],
                text_b=row["hypo"],
                label=("contradiction"
                       if row["label"] == "contradictory" else
                       row["label"])
            )
            for i, row
            in enumerate(rows)
        ]
        return examples

    def get_dev_examples(self, data_dir):
        self._get_test_or_dev_helper(data_dir, "dev")

    def get_test_examples(self, data_dir):
        self._get_test_or_dev_helper(data_dir, "test")

    def _get_test_or_dev_helper(self, data_dir, stage):
        if stage == "dev":
            language = self.dev_language
        elif stage == "test":
            language = self.test_language

        rows = self._read_xnli_tsv(
            os.path.join(
                data_dir,
                "XNLI-1.0",
                f"xnli.{stage}.tsv"
            ))
        examples = [
            InputExample(
                guid=f"{stage}-{i}",
                text_a=row["sentence1"],
                text_b=row["sentence2"],
                label=row["gold_label"]
            )
            for i, row
            in enumerate(rows)
            if row["language"] == language
        ]
        return examples

    def _read_xnli_tsv(self, input_path):
        """
        Pretty much the same as the _read_tsv from the repo but i wanted it to
        read files as dicts instead of lists cos it's more readable
        """
        with open(input_path, "r") as tsvfile:
            reader = csv.DictReader(
                tsvfile,
                quoting=csv.QUOTE_NONE,
                delimiter="\t"
            )
            if self.cased:
                return list(reader)
            else:
                return [OrderedDict((key, value.lower()) for key, value in row.items()) for row in reader]

    def get_labels():
        return ["contradiction", "entailment", "neutral"]


def set_seed(args):
    """
    Function to **SET ALL THE SEEDS**!!
    Shamelesly stolen from an example in hugginface's transformers repo
    """
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def load_dataset(data_dir, tokenizer, train=True):
    ...


def main(passed_args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=None, type=str, required=True)
    parser.add_argument("--model_dir", default=None, type=str, requiered=True)
    args = parser.parse_args(passed_args)

    print(args)
    processor = XNLIProcessor("es", cased=True)
    path = os.path.join(args.data_dir, "XNLI-MT-1.0", "multinli", "multinli.train.es.tsv")
    # examples = processor.get_train_examples(args.data_dir)
    # wrong_lines = [line for line in lines if len(line) != 3]
    # pprint.pprint(wrong_lines[:2])
    # print(examples[:3])
    lines = processor._read_xnli_tsv(path)
    print(len(lines))
    # breakpoint()
    print(lines[:3])


if __name__ == '__main__':
    main()