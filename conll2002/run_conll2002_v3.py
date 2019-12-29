"""
The idea here is to try again the sliding window approach, but
create the input features differently.

Maybe i was doing something wrong the whole time, and doing this all over
again sometimes solves things

I will also test making the script so that it can test
"""
import os
import torch
import torch.nn as nn
from collections import namedtuple
from tqdm import tqdm
from operator import itemgetter
from itertools import zip_longest
import argparse

from transformers import BertTokenizer


DATA_FILE_DICT = {
    "train": "esp.train",
    "dev": "esp.testa",
    "test": "esp.testb",
}

IGNORE_INDEX = nn.CrossEntropyLoss().ignore_index


Example = namedtuple("Example", ["tokens", "labels", "prediction_mask"])
# No token_type_ids, because i will treat the whole example as just
# one sentence
Feature = namedtuple(
    "Feature", ["input_ids", "attention_mask", "labels", "prediction_mask"])

INWORD_PAD_LABEL = "PAD"


def load_examples(args, tokenizer, stage):
    """
    Using the sliding window approach, this means that for every new
    example, only 64 (half of the max length) new words are added,
    and the last 64 from the previous examplr are kept as context.

    Each example should also include a mask defining the words that
    are to be predicted for the example. This is because at prediction
    time, we want to keep only the predictions for the words that were
    added to the example and the ones that were kep as context should
    have been predicted in a previous example.

    Besides that, since bert is a sub word model, only the firs token
    of a word should be labeled correctly, the sub word tokens should
    be labeled using IGNORE_INDEX, which is a label that will be ignored
    by the loss function
    """

    with open(os.path.join(args.data_dir, DATA_FILE_DICT[stage])) as in_f:
        lines = in_f.readlines()

    tokens, labels = [], []
    for line in tqdm(lines, desc="Tokenizing words"):
        if line != "\n":
            word, original_label = itemgetter(0, 2)(line.split())
            tokenized = tokenizer.tokenize(word)
            tokens += tokenized
            labels += [original_label] + \
                [INWORD_PAD_LABEL] * (len(tokenized) - 1)

    assert len(tokens) == len(labels)
    # To account for the CLS and SEP tokens
    half_len = (args.max_length - 2) // 2
    examples = []
    for i in tqdm(
        range(0, len(tokens)-half_len, half_len),
        desc="Creating examples",
    ):
        token_window = tokens[i:i+2*half_len]
        label_window = labels[i:i+2*half_len]

        if i == 0:
            prediction_mask = [1]*2*half_len
        else:
            prediction_mask = [0]*half_len + [1]*(len(token_window) - half_len)

        example = Example(
            token_window,
            label_window,
            prediction_mask
        )
        examples.append(example)
    breakpoint()
    return examples


def load_dataset(args, tokenizer, stage):
    examples = load_examples(args, tokenizer, stage)

    ...


def train(args):

    tokenizer = BertTokenizer.from_pretrained(
        args.model_dir,
        do_lower_case=args.do_lower_case,
        keep_accents=args.keep_accents,
    )

    train_dataset = load_dataset(args, tokenizer, "train")
    breakpoint()
    ...


def test(args):
    ...


def main(passed_args=None):
    parser = argparse.ArgumentParser()
    parser.set_defaults(func=lambda *args: parser.print_help())
    subparsers = parser.add_subparsers(title="subcommands")
    parser_train = subparsers.add_parser("train")
    parser_test = subparsers.add_parser("test")
    # Common arguments
    for p in [parser_train, parser_test]:
        p.add_argument("--model-dir", default="../beto-cased", type=str)
        p.add_argument("--data-dir", default="./data", type=str)
        p.add_argument("--output-dir", default="./outputs", type=str)

    # Specific for train
    parser_train.add_argument("--max_length", default=128, type=int)
    parser_train.add_argument("--do-lower-case", action="store_true")
    parser_train.add_argument(
        "--remove-accents", action="store_false", dest="keep_accents")
    parser_train.add_argument("--disable-cuda", action="store_true")

    # Set default functions
    parser_train.set_defaults(func=train)
    parser_test.set_defaults(func=test)


    args = parser.parse_args()

    return args.func(args)
    # breakpoint()

    ...


if __name__ == '__main__':
    main()