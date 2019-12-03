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
import pprint
import logging
import argparse
from collections import OrderedDict

import tqdm
import torch
import numpy as np
from torch.utils.data import (TensorDataset, RandomSampler, DataLoader)
from torch.utils.tensorboard import SummaryWriter
from torch.optim import AdamW
from transformers import (
    DataProcessor,
    InputExample,
    BertConfig,
    BertForSequenceClassification,
    BertTokenizer,
    # aparentemente esta funcion es media agnostica a la tarea
    glue_convert_examples_to_features as convert_examples_to_features,
    get_linear_schedule_with_warmup
)

logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
        return self._get_test_or_dev_helper(data_dir, "dev")

    def get_test_examples(self, data_dir):
        return self._get_test_or_dev_helper(data_dir, "test")

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

    @classmethod
    def get_labels(cls):
        return ["contradiction", "entailment", "neutral"]


def set_seed(seed):
    """
    Function to **SET ALL THE SEEDS**!!
    Shamelesly stolen from an example in hugginface's transformers repo
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.device_count() > 0:
        torch.cuda.manual_seed_all(seed)


def load_dataset(data_dir, tokenizer, stage="train"):
    processor = XNLIProcessor("es", cased=False)
    label_list = processor.get_labels()
    if stage == "train":
        examples = processor.get_train_examples(data_dir)
    elif stage == "dev":
        examples = processor.get_dev_examples(data_dir)
    elif stage == "test":
        examples = processor.get_test_examples(data_dir)

    logger.info("Creating features from dataset file at %s", data_dir)
    features = convert_examples_to_features(
        examples,
        tokenizer,
        max_length=128,
        label_list=label_list,
        output_mode="classification",
        pad_on_left=False,
        pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
        pad_token_segment_id=0,
    )

    all_input_ids = torch.tensor(
        [f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor(
        [f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor(
        [f.token_type_ids for f in features], dtype=torch.long)
    all_labels = torch.tensor(
        [f.label for f in features], dtype=torch.long)
    dataset = TensorDataset(
        all_input_ids,
        all_attention_mask,
        all_token_type_ids,
        all_labels
    )
    # breakpoint()
    return dataset


def train(model, dataset):
    train_epochs = 2

    tb_writer = SummaryWriter()

    sampler = RandomSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=16)
    # The optimizer
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [
                p
                for n, p 
                in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            'weight_decay': 0.01  # Default for AdamW in torch
        },
        {
            'params': [
                p 
                for n, p 
                in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            'weight_decay': 0.0}
        ]
    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=5e-5,
        eps=1e-8
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=1,
        num_training_steps=train_epochs * len(dataloader)
    )

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Num Epochs = %d", train_epochs)
    logger.info("  Total optimization steps = %d", train_epochs * len(dataloader))

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = tqdm.trange(train_epochs, desc="Epoch")

    log_every = 50
    set_seed(42)  # No idea why here and outside as well
    for _ in train_iterator:
        epoch_iterator = tqdm.tqdm(dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(DEVICE) for t in batch)
            inputs = dict(zip(
                ["input_ids", "attention_mask", "token_type_ids", "labels"],
                batch
            ))
            outputs = model(**inputs)
            loss = outputs[0]
            if torch.cuda.device_count() > 1:
                loss = loss.mean()
            loss.backward()
            tr_loss += loss.item()

            optimizer.step()
            scheduler.step()
            model.zero_grad()
            global_step += 1

            if global_step % log_every == 0:
                # Aca tengo caleta de dudas sobre que es lo que tengo que 
                # loggear y que es lo que tengo que devolver al final
                # por ahora voy a llegar y copiar nomas
                tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                tb_writer.add_scalar(
                    'loss', (tr_loss - logging_loss)/log_every, global_step)
                logging_loss = tr_loss

    tb_writer.close()

    return global_step, tr_loss / global_step


def main(passed_args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=None, type=str, required=True)
    parser.add_argument("--model_dir", default=None, type=str, required=True)
    parser.add_argument("--cased", default=False, type=bool)
    args = parser.parse_args(passed_args)

    print(args)
    # processor = XNLIProcessor("es", cased=args.cased)
    # path = os.path.join(args.data_dir, "XNLI-MT-1.0", "multinli", "multinli.train.es.tsv")
    # examples = processor.get_train_examples(args.data_dir)
    # wrong_lines = [line for line in lines if len(line) != 3]
    # pprint.pprint(wrong_lines[:2])
    # print(examples[:3])
    # lines = processor._read_xnli_tsv(path)
    # print(len(lines))
    # # breakpoint()
    # print(lines[:3])

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)

    # Set seed
    set_seed(42)

    num_labels = len(XNLIProcessor.get_labels())
    config = BertConfig.from_pretrained(
        args.model_dir,
        num_labels=num_labels,
        finetuning_task="xnli"
    )
    tokenizer = BertTokenizer.from_pretrained(
        args.model_dir,
        do_lower_case=args.cased,
    )
    model = BertForSequenceClassification.from_pretrained(
        args.model_dir,
        config=config,
    )
    model.to(DEVICE)
    # breakpoint()
    dataset = load_dataset(args.data_dir, tokenizer, stage="train")
    global_step, tr_loss = train(model, dataset)
    logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)


if __name__ == '__main__':
    main()