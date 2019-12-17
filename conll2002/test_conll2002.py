"""
In hindsight, i should've made all of this more modular. Sigh...
"""
import random
import argparse
from transformers import (
    BertForTokenClassification,
    BertTokenizer,
)
from itertools import zip_longest
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
from dataclasses import dataclass
from typing import List
from sklearn.metrics import f1_score

logger = logging.getLogger(__name__)

LABEL_LIST = ('B-PER', 'B-MISC', 'I-ORG', 'B-ORG',
              'I-LOC', 'B-LOC', 'I-PER', 'O', 'I-MISC')

LABEL_MAP = {label: i for i, label in enumerate(LABEL_LIST)}
INWORD_PAD_LABEL = "PAD"
LABEL_MAP[INWORD_PAD_LABEL] = -100  # adds a value for the added PAD label

TEST_FILE = "esp.testb"


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


# TODO: implemenentar el algoritmo real que usan para parsear el test set
class CoNLL2002ProcessorTest():
    def __init__(self, tokenizer, sequence_length=128):
        self.sequence_length = sequence_length
        self.tokenizer = tokenizer

    def get_test_examples(self, data_dir):
        words, labels = self.read_file(os.path.join(data_dir, TEST_FILE))
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
        "cached_dataset_beto_{}_conll2002_es_test_{}".format(
            "uncased" if args.do_lower_case else "uncased",
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
        examples = processor.get_test_examples(args.data_dir)
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
            if idx == 0:
                # breakpoint()
                # Only for the first example add the first sentence
                half_len = args.max_seq_len // 2
                positions = labels[0, :half_len] != LABEL_MAP[INWORD_PAD_LABEL]
                preds = torch.cat([
                    preds,
                    scores[0, :half_len][positions].cpu()
                ])
                # breakpoint()
            # For every example add the second sentence
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
        in processor.read_file(os.path.join(args.data_dir, TEST_FILE))[1]
    ]
    # breakpoint()
    assert len(preds) == len(gold_labels)

    score = f1_score(
        gold_labels,
        preds.argmax(dim=1),
        average=args.f1_average_type
    )

    breakpoint()
    # TODO: borrar esto, tarea de NLP
    with open("nlp_tarea2", "w") as f:
        for index in preds.argmax(dim=1):
            f.write(LABEL_LIST[index])
            f.write("\n")

    # breakpoint()
    return {"f1_score": score}


def main(passed_args=None):
    parser = argparse.ArgumentParser()
    # Required
    parser.add_argument("--model-dir", default=None, type=str, required=True)
    parser.add_argument("--data-dir", default=None, type=str, required=True)
    parser.add_argument("--output-dir", default=None, type=str, required=True)

    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--disable-cuda", action="store_true")
    parser.add_argument("--overwrite-cache", action="store_true")

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
    args.max_seq_len = prev_args.max_seq_len

    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO
    )

    # ********* Load model ****************
    tokenizer = BertTokenizer.from_pretrained(
        args.model_dir,
        do_lower_case=args.do_lower_case
    )
    model = BertForTokenClassification.from_pretrained(args.model_dir)
    model.to(args.device)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    # ************* Eval *************
    processor = CoNLL2002ProcessorTest(tokenizer)
    test_dataset = load_dataset(args, processor, tokenizer)
    results = evaluate(args, model, test_dataset, processor)
    # ********************* Save results ******************
    logger.info(f"Saving results to {args.output_dir}/test_results.json")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    with open(os.path.join(args.output_dir, "test_results.json"), "w") as f:
        json.dump(results, f)
    print(results)


if __name__ == '__main__':
    main()
