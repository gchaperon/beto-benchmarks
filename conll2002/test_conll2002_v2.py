import json
import argparse
import torch
import os
import logging
from dataclasses import dataclass
from typing import List
from operator import itemgetter
from itertools import zip_longest
from tqdm import tqdm
from sklearn.metrics import f1_score

from torch.utils.data import TensorDataset, DataLoader

from transformers import (
    BertTokenizer,
    BertForTokenClassification
)


logger = logging.getLogger(__name__)

TEST_FILE = "esp.testb"
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


def load_examples(args):
    logger.info("Reading examples")
    path = os.path.join(args.data_dir, TEST_FILE)
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


def load_dataset(args, tokenizer):
    """
    Some day i will implement the logic for saving to a cache file
    and loading from it
    """
    examples = load_examples(args)

    features, n_overflowing_examples = examples2features(
        examples, tokenizer, args.max_length)

    assert n_overflowing_examples == 0

    dataset = TensorDataset(
        torch.tensor([f.input_ids for f in features]),
        torch.tensor([f.attention_mask for f in features]),
        torch.tensor([f.token_type_ids for f in features]),
        torch.tensor([f.labels for f in features]),
    )
    # breakpoint()
    return dataset


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
    # breakpoint()
    result = f1_score(
        gold_labels,
        predictions.argmax(dim=1),
        labels=[LABEL_MAP[label] for label in LABEL_LIST if label != "O"],
        average="micro",
    )
    return {"f1_score": result}


def main(passed_args=None):
    parser = argparse.ArgumentParser()
    # Required
    parser.add_argument("--model-dir", default=None, type=str, required=True)
    parser.add_argument("--data-dir", default=None, type=str, required=True)
    parser.add_argument("--output-dir", default=None, type=str, required=True)

    parser.add_argument("--batch-size", default=8, type=int)
    parser.add_argument("--disable-cuda", action="store_true")

    # Specific parameters
    parser.add_argument("--f1-average-type", default="micro", type=str)

    args = parser.parse_args()
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device("cuda")
        args.n_gpu = torch.cuda.device_count()
    else:
        args.device = torch.device("cpu")
        args.n_gpu = 0

    # Recover args from training
    prev_args = torch.load(os.path.join(args.model_dir, "train_args.bin"))
    args.do_lower_case = prev_args.do_lower_case
    args.max_length = prev_args.max_length
    args.keep_accents = prev_args.keep_accents

    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO
    )

    tokenizer = BertTokenizer.from_pretrained(
        args.model_dir,
        do_lower_case=args.do_lower_case,
        keep_accents=args.keep_accents,
    )
    model = BertForTokenClassification.from_pretrained(args.model_dir)
    model.to(args.device)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # *** Evaluate ! ***

    # Get the gold labesl from the original file
    gold_labels = [
        LABEL_MAP[line.split()[2]]
        for line in open(os.path.join(args.data_dir, TEST_FILE))
        if line != "\n"
    ]
    test_dataset = load_dataset(args, tokenizer)
    results = evaluate(args, model, test_dataset, gold_labels)

    logger.info(f"Saving results to {args.output_dir}/test_results.json")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    with open(os.path.join(args.output_dir, "test_results.json"), "w") as f:
        json.dump(results, f)

    logger.info(f"Results: {results}")


if __name__ == '__main__':
    main()
