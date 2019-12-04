#!/usr/bin/env python3

"""
Modified from examples/run_xnli.py, mainly change the processor a bit and
removed the distributed capabilities for clarity.

Also, I changed the formatting in many places

This script wasn't designed to work with anything else than beto (spanish bert)

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
import glob
import random
import pprint
import logging
import argparse
from collections import OrderedDict

import torch
import numpy as np
from torch.utils.data import (
    TensorDataset,
    RandomSampler,
    DataLoader,
    SequentialSampler,
)
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torch.optim import AdamW
from transformers import (
    WEIGHTS_NAME,
    DataProcessor,
    InputExample,
    BertConfig,
    BertForSequenceClassification,
    BertTokenizer,
    # aparentemente esta funcion es media agnostica a la tarea
    glue_convert_examples_to_features as convert_examples_to_features,
    get_linear_schedule_with_warmup
)

from tqdm import tqdm, trange

# from transformers import AdamW

from transformers import xnli_compute_metrics as compute_metrics
from transformers import xnli_output_modes as output_modes
# from transformers import xnli_processors as processors

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

    def __init__(self, train_language="es", test_language=None, cased=True):
        self.train_language = train_language
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
                return [
                    OrderedDict(
                        (key, value.lower())
                        for key, value
                        in row.items()
                    )
                    for row
                    in reader
                ]

    @classmethod
    def get_labels(cls):
        return ["contradiction", "entailment", "neutral"]


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def train(args, train_dataset, model, tokenizer):
    """ Train the model """
    tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // \
            (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // \
            args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [
                p
                for n, p
                in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            'weight_decay': args.weight_decay
        },
        {
            'params': [
                p
                for n, p
                in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            'weight_decay': 0.0
        }
    ]
    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=args.learning_rate,
        eps=args.adam_epsilon
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=t_total
    )

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex"
                "to use fp16 training.")
        model, optimizer = amp.initialize(
            model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d",
                args.per_gpu_train_batch_size)
    logger.info("  Total train batch size "
                "(w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps)
    logger.info("  Gradient Accumulation steps = %d",
                args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(
        int(args.num_train_epochs),
        desc="Epoch",
        disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Added here for reproductibility (between python 2 and 3)
    for _ in train_iterator:
        epoch_iterator = tqdm(
            train_dataloader,
            desc="Iteration",
            disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2],
                      'labels':         batch[3]}
            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

            if args.n_gpu > 1:
                loss = loss.mean()  # average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(
                        amp.master_params(optimizer),
                        args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if (args.logging_steps > 0 and
                   global_step % args.logging_steps == 0):
                    # Log metrics
                    if args.evaluate_during_training:
                        results = evaluate(args, model, tokenizer)
                        for key, value in results.items():
                            tb_writer.add_scalar(
                                'eval_{}'.format(key), value, global_step)
                    tb_writer.add_scalar(
                        'lr', scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar(
                        'loss',
                        (tr_loss - logging_loss)/args.logging_steps,
                        global_step)
                    logging_loss = tr_loss

                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(
                        args.output_dir, 'checkpoint-{}'.format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model.save_pretrained(output_dir)
                    torch.save(
                        args, os.path.join(output_dir, 'training_args.bin'))
                    logger.info("Saving model checkpoint to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    tb_writer.close()

    return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, prefix=""):
    eval_task_names = (args.task_name,)
    eval_outputs_dirs = (args.output_dir,)

    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        eval_dataset = load_and_cache_examples(args, eval_task, tokenizer, evaluate=True)

        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        # multi-gpu eval
        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        # Eval!
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {'input_ids':      batch[0],
                          'attention_mask': batch[1],
                          'labels':         batch[3]}
                if args.model_type != 'distilbert':
                    inputs['token_type_ids'] = batch[2] if args.model_type in ['bert'] else None  # XLM and DistilBERT don't use segment_ids
                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs['labels'].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        if args.output_mode == "classification":
            preds = np.argmax(preds, axis=1)
        else:
            raise ValueError('No other `output_mode` for XNLI.')
        result = compute_metrics(eval_task, preds, out_label_ids)
        results.update(result)

        output_eval_file = os.path.join(eval_output_dir, prefix, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results {} *****".format(prefix))
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

    return results


def load_and_cache_examples(args, task, tokenizer, evaluate=False):
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    processor = processors[task](language=args.language, train_language=args.train_language)
    output_mode = output_modes[task]
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(args.data_dir, 'cached_{}_{}_{}_{}_{}'.format(
        'test' if evaluate else 'train',
        list(filter(None, args.model_name_or_path.split('/'))).pop(),
        str(args.max_seq_length),
        str(task),
        str(args.train_language if (not evaluate and args.train_language is not None) else args.language)))
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels()
        examples = processor.get_test_examples(args.data_dir) if evaluate else processor.get_train_examples(args.data_dir)
        features = convert_examples_to_features(examples,
                                                tokenizer,
                                                label_list=label_list,
                                                max_length=args.max_seq_length,
                                                output_mode=output_mode,
                                                pad_on_left=False,
                                                pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                                pad_token_segment_id=0,
        )
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    if output_mode == "classification":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    else:
        raise ValueError('No other `output_mode` for XNLI.')

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
    return dataset


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