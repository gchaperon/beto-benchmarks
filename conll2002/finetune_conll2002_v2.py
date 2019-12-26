"""
This time i wont use the sliding window aproach, just pass in a whole
sentence, maybe that is the problem with the other script
"""

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

from itertools import zip_longest
from dataclasses import dataclass
from typing import List
import argparse
import os
from operator import itemgetter, attrgetter
import logging
from transformers import BertTokenizer
from tqdm import tqdm


logger = logging.getLogger(__name__)


LABEL_LIST = ['O', 'I-LOC', 'I-MISC', 'B-ORG', 'B-PER',
              'I-ORG', 'B-MISC', 'B-LOC', 'I-PER']

LABEL_MAP = {label: i for i, label in enumerate(LABEL_LIST)}
IGNORE_INDEX = torch.nn.CrossEntropyLoss().ignore_index


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
    breakpoint()
    return features


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
        max_length = 512 if args.max_length < 512 else args.max_length
    else:
        max_length = args.max_length

    features = examples2features(examples, tokenizer, max_length)

    dataset = 






def main(passed_args=None):

    # TODO: cambiar los default
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="../beto-cased", type=str)
    parser.add_argument("--data-dir", default="./data", type=str)

    parser.add_argument("--max-length", default=128, type=int)

    parser.add_argument("--do-lower-case", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO
    )
    examples = load_examples(args, eval_=False)

    tokenizer = BertTokenizer.from_pretrained(
        args.model_dir,
        do_lower_case=args.do_lower_case,
    )

    features = examples2features(args, examples, tokenizer)
    breakpoint()
    ...


if __name__ == '__main__':
    main()