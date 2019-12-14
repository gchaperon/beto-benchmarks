import os
import torch
import argparse
import random
import numpy as np
import pprint
from tqdm import tqdm
from operator import itemgetter
from itertools import zip_longest
from dataclasses import dataclass
from typing import List

import logging

from transformers import (
    BertForTokenClassification,
    BertTokenizer,
    BertConfig,
)

logger = logging.getLogger(__name__)


LABEL_LIST = ('B-PER', 'B-MISC', 'I-ORG', 'B-ORG',
              'I-LOC', 'B-LOC', 'I-PER', 'O', 'I-MISC')

LABEL_MAP = {label: i for i, label in enumerate(LABEL_LIST)}
PAD_LABEL = "PAD"
LABEL_MAP[PAD_LABEL] = -1  # adds a value for the added PAD label


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
        words, labels = self._read_file(os.join(data_dir, "esp.testa"))
        return self._build_examples(words, labels)

    def get_train_examples(self, data_dir):
        words, labels = self._read_file(os.join(data_dir, "esp.train"))
        return self._build_examples(words, labels)

    def _build_examples(self, words, labels):
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
                fillvalue=PAD_LABEL
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

    def _read_file(self, file_name):
        with open(file_name, "r") as file:
            return zip(*[
                itemgetter(0, 2)(line.split())
                for line
                in file
                if line != "\n"
            ])



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
            -1,  # CLS
            *[LABEL_MAP[label] for label in example.labels_a],
            -1,  # SEP
            *[LABEL_MAP[label] for label in example.labels_b],
            -1,  # SEP
        ]
        labels += [-1] * (args.max_seq_len - len(labels))
        assert len(labels) == args.max_seq_len
        assert "overflowing_tokens" not in results

        features.append(InputFeature(**results, labels=labels))

        # breakpoint()


def main(passed_args=None):
    parser = argparse.ArgumentParser()

    # Required
    parser.add_argument("--model-dir", default="../beto-uncased", type=str)
    parser.add_argument("--data-dir", default="./data", type=str)
    parser.add_argument("--output-dir", default="./outputs", type=str)

    # Hyperparams that where relatively common
    parser.add_argument("--max-seq-len", default=128, type=int)
    args = parser.parse_args(passed_args)

    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO
    )

    # TODO: reemplazar el do_lower_case
    tokenizer = BertTokenizer.from_pretrained(
        args.model_dir,
        do_lower_case=True
    )

    processor = CoNLL2002Processor(tokenizer, sequence_length=args.max_seq_len)
    words, labels = processor._read_file("data/esp.testa")
    examples = processor._build_examples(words, labels)

    for example in examples[:5]:
        print(example)

    features = examples2features(args, examples, tokenizer)
    # breakpoint()


if __name__ == '__main__':
    main()
