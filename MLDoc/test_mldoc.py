"""
In hindsight, i should've made all of this more modular. Sigh...
"""

import argparse
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    DataProcessor,
    InputExample,
    InputFeatures,
)
import torch
from torch.utils.data import TensorDataset, DataLoader
import os
import csv
import nltk
import logging
from tqdm import tqdm
from operator import attrgetter, itemgetter
from itertools import islice
import json

logger = logging.getLogger(__name__)


class MLDocProcessorTest(DataProcessor):
    def get_test_examples(self, data_dir):
        rows = self._read_tsv(os.path.join(data_dir, f"mldoc.es.test"))
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
                label=row[0],
            )
            examples.append(example)

        return examples

    def get_labels(self):
        return ["CCAT", "MCAT", "ECAT", "GCAT"]

    def _read_tsv(self, fpath):
        with open(fpath, "r") as file:
            reader = csv.reader(file, quoting=csv.QUOTE_NONE, delimiter="\t")
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
            max_length=max_length,
        )

        # Im so sorry for this xD
        (input_ids, token_type_ids) = itemgetter(
            "input_ids", "token_type_ids"
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

        features.append(
            InputFeatures(
                input_ids,
                attention_mask,
                token_type_ids,
                label=label_map[example.label],
            )
        )

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


def load_dataset(args, processor, tokenizer):
    cache_file = os.path.join(
        args.data_dir,
        "cached_features_beto_{}_mldoc_es_test_{}".format(
            "uncased" if args.do_lower_case else "cased", args.max_seq_len,
        ),
    )

    if os.path.exists(cache_file) and not args.overwrite_cache:
        logger.info(f"Loading features from cached file at {cache_file}")
        features = torch.load(cache_file)
    else:
        logger.info(f"Creating features from dataset file at {args.data_dir}")
        examples = processor.get_test_examples(args.data_dir)
        features = examples2features(
            examples,
            tokenizer,
            processor.get_labels(),
            max_length=args.max_seq_len,
        )
        # Save features to cache file
        logger.info(f"Saving features into cached file {cache_file}")
        torch.save(features, cache_file)

    # Im just partially sorry for this :D
    getter = attrgetter(
        "input_ids", "attention_mask", "token_type_ids", "label"
    )
    tensors = map(torch.tensor, zip(*[getter(f) for f in features]))
    dataset = TensorDataset(*tensors)
    return dataset


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
            )[0]
            preds = torch.cat([preds, logits.cpu()])
            gold_labels = torch.cat([gold_labels, batch[3].cpu()])

    correct = (preds.argmax(dim=1) == gold_labels).sum().item()
    return {"acc": correct / len(preds)}


def main(passed_args=None):
    parser = argparse.ArgumentParser()
    # Required
    parser.add_argument("--model-dir", default=None, type=str, required=True)
    parser.add_argument("--data-dir", default=None, type=str, required=True)
    parser.add_argument("--output-dir", default=None, type=str, required=True)

    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--disable-cuda", action="store_true")
    parser.add_argument("--overwrite-cache", action="store_true")

    args = parser.parse_args()

    args.device = None
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device("cuda")
        args.n_gpu = torch.cuda.device_count()
    else:
        args.device = torch.device("cpu")
        args.n_gpu = 0

    # Recover args from training
    prev_args = torch.load(os.path.join(args.model_dir, "train_args.bin"))
    args.do_lower_case = prev_args.do_lower_case
    args.max_seq_len = prev_args.max_seq_len

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # ********* Load model ****************
    tokenizer = BertTokenizer.from_pretrained(
        args.model_dir, do_lower_case=args.do_lower_case
    )
    model = BertForSequenceClassification.from_pretrained(args.model_dir)
    model.to(args.device)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    breakpoint()
    # ************* Eval *************
    processor = MLDocProcessorTest()
    test_dataset = load_dataset(args, processor, tokenizer)
    results = evaluate(args, model, test_dataset)
    # ********************* Save results ******************
    logger.info(f"Saving results to {args.output_dir}/test_results.json")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    with open(os.path.join(args.output_dir, "test_results.json"), "w") as f:
        json.dump(results, f)
    print(results)


if __name__ == "__main__":
    main()
