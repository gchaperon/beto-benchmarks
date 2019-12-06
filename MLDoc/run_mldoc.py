#!/usr/bin/env
import os
import csv
import logging
from transformers import (
    BertTokenizer,
    DataProcessor,
    InputExample,
    BertForSequenceClassification,
    InputFeatures,
    )
from tqdm import tqdm
from operator import itemgetter
from itertools import islice

import nltk.data

logger = logging.getLogger(__name__)


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

            # breakpoint()
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


def examples2features(examples,
                      tokenizer,
                      label_list,
                      max_length=128,
                      ):
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
        pad_token = tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0]
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
        logger.info(f"token_type_ids: {token_type_ids}")
        logger.info(f"label: {example.label} (id = {feature.label})")

    return features


def main():
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO
    )
    processor = MLDocProcessor()
    # print(processor.get_dev_examples("./data")[:5])

    # rows = processor._read_tsv("./data/mldoc.es.dev")

    # assert all(len(row) == 2 for row in rows)
    tokenizer = BertTokenizer.from_pretrained("../beto", do_lower_case=True)
    examples = processor.get_train_examples("./data")
    features = examples2features(examples, tokenizer, processor.get_labels())
    # assert all(example.text_b is not None for example in processor.get_train_examples("./data"))


if __name__ == '__main__':
    main()