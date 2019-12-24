"""
This time i wont use the sliding window aproach, just pass in a whole
sentence, maybe that is the problem with the other script
"""
from dataclasses import dataclass
from typing import List
import argparse
import os
from operator import itemgetter
import logging


logger = logging.getLogger(__name__)

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


def load_examples(args, evaluate):
    path = os.path.join(args.data_dir, "esp.testa" if evaluate else "esp.train")
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
    breakpoint()
    return examples


def main(passed_args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument()


    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO
    )
    ...


if __name__ == '__main__':
    main()