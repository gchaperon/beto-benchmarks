"""
The idea here is to try again the sliding window approach, but
create the input features differently.

Maybe i was doing something wrong the whole time, and doing this all over
again sometimes solves things

I will also test making the script so that it can test
"""
import os
import sys
import random
import json
from platform import node
from datetime import datetime
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from collections import namedtuple
from tqdm import tqdm
from operator import itemgetter
import argparse
import logging
import numpy as np
from transformers import (
    BertTokenizer,
    BertForTokenClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import f1_score

logger = logging.getLogger("__name__")


DATA_FILE_DICT = {
    "train": "esp.train",
    "dev": "esp.testa",
    "test": "esp.testb",
}

Example = namedtuple("Example", ["tokens", "labels", "prediction_mask"])

IGNORE_INDEX = nn.CrossEntropyLoss().ignore_index
INWORD_PAD_LABEL = "PAD"
LABEL_LIST = ['O', 'I-LOC', 'I-MISC', 'B-ORG', 'B-PER',
              'I-ORG', 'B-MISC', 'B-LOC', 'I-PER']

LABEL_MAP = {label: i for i, label in enumerate(LABEL_LIST)}
LABEL_MAP[INWORD_PAD_LABEL] = IGNORE_INDEX


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


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

    assert args.max_length % 2 == 0
    half_len = args.max_length // 2
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

        assert len(token_window) == len(label_window) == len(prediction_mask)
        example = Example(
            token_window,
            label_window,
            prediction_mask
        )
        examples.append(example)

    # breakpoint()
    return examples


def load_dataset(args, tokenizer, stage):

    examples = load_examples(args, tokenizer, stage)
    input_ids, labels, prediction_masks = [], [], []
    for i, (tokens, ex_labels, prediction_mask) in enumerate(tqdm(
        examples, desc="Converting examples"
    )):
        ids = tokenizer.convert_tokens_to_ids(tokens)
        label_ids = [LABEL_MAP[l] for l in ex_labels]

        # the last examples should be padded
        if i == len(examples) - 1:
            pad_length = args.max_length - len(tokens)
            ids += [tokenizer.pad_token_id] * pad_length
            label_ids += [IGNORE_INDEX] * pad_length
            prediction_mask += [0] * pad_length

        input_ids.append(ids)
        labels.append(label_ids)
        prediction_masks.append(prediction_mask)

    # Verify that everything is the same length
    for list_ in (input_ids, labels, prediction_masks):
        for item in list_:
            assert len(item) == args.max_length

    dataset = TensorDataset(
        torch.tensor(input_ids),
        torch.tensor(labels),
        torch.tensor(prediction_masks),
    )
    # breakpoint()
    return dataset


def train_model(args, model, dataset):
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    total_steps = len(dataloader) * args.epochs

    no_decay = ['bias', 'LayerNorm.weight']
    grouped_parameters = [
        {
            'params': [
                p
                for n, p
                in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            'weight_decay': 0.0
        },
        {
            'params': [
                p
                for n, p
                in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            'weight_decay': args.weight_decay
        },
    ]
    optimizer = Adam(
        grouped_parameters,
        lr=args.learn_rate,
        weight_decay=args.weight_decay
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(args.warmup_steps*total_steps),  # warmup is a %
        num_training_steps=total_steps,
    )

    tb_writer = SummaryWriter(
        log_dir="runs/conll2002_{}_{}".format(
            datetime.now().strftime('%b%d_%H-%M'),
            node()
        )
    )
    # *** Finally train ***

    global_step, tr_loss, running_loss = 0, 0., 0.
    for _ in tqdm(range(args.epochs), desc="Epoch"):
        for step, batch in enumerate(tqdm(dataloader, desc="Iteration")):
            model.train()
            batch = [t.to(args.device) for t in batch]

            model.zero_grad()

            loss = model(
                input_ids=batch[0],
                labels=batch[1],
            )[0]
            if args.n_gpu > 1:
                loss = loss.mean()

            loss.backward()
            optimizer.step()
            scheduler.step()

            tr_loss += loss.item()
            running_loss += loss.item()

            if (args.logging_steps > 0
                    and global_step % args.logging_steps == 0):
                tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                tb_writer.add_scalar(
                    "loss", running_loss/args.logging_steps, global_step)
                running_loss = 0.

            global_step += 1

    tb_writer.close()

    return global_step, tr_loss / global_step


def evaluate(args, model, dataset, gold_labels):

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    predictions = torch.tensor([], dtype=torch.long).to(args.device)
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            model.eval()

            input_ids, labels, prediction_mask = \
                [t.to(args.device) for t in batch]

            scores = model(input_ids)[0]
            # breakpoint()
            positions = prediction_mask.eq(1) * labels.ne(IGNORE_INDEX)
            predictions = torch.cat([
                predictions,
                scores[positions].argmax(dim=1)
            ])

    # bring predictions back to cpu
    predictions = predictions.cpu()
    # check length
    assert len(predictions) == len(gold_labels), \
        f"{len(predictions)} != {len(gold_labels)}"

    results = {
        "f1_score": f1_score(
            gold_labels,
            predictions,
            # consider only the entity labels
            labels=[LABEL_MAP[label] for label in LABEL_LIST if label != "O"],
            average=args.average_type,
        )
    }

    # breakpoint()
    return results


def load_model(args, test):
    # if the model is for testing, attempt to load previous arguments
    if test:
        try:
            prev_args = torch.load(
                os.path.join(args.model_dir, "train_args.bin"))
            args.max_length = prev_args.max_length
            args.do_lower_case = prev_args.do_lower_case
            args.keep_accents = prev_args.keep_accents
        except FileNotFoundError:
            pass

    tokenizer = BertTokenizer.from_pretrained(
        args.model_dir,
        do_lower_case=args.do_lower_case,
        keep_accents=args.keep_accents,
    )
    model = BertForTokenClassification.from_pretrained(
        args.model_dir,
        finetuning_task="conll2002",
        num_labels=len(LABEL_LIST),
    ).to(args.device)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    return model, tokenizer


def get_gold_labels(args, stage):
    gold_labels = []
    with open(os.path.join(args.data_dir, DATA_FILE_DICT[stage])) as file:
        for line in file:
            if line != "\n":
                gold_labels.append(LABEL_MAP[line.split()[2]])

    return gold_labels


def save_pretrained(args, model, tokenizer):
    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"Saving model checkpoint to {args.output_dir}")
    if isinstance(model, torch.nn.DataParallel):
        model = model.module

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    torch.save(args, os.path.join(args.output_dir, "train_args.bin"))


def save_results(args, results, stage):
    logger.info(f"Saving results to "
                f"{os.path.join(args.output_dir, f'{stage}_results.json')}")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    results_file = open(
        os.path.join(args.output_dir, f"{stage}_results.json"), "w")
    with results_file:
        json.dump(results, results_file)
    if stage == "dev":
        logger.info(f"Training args are saved to "
                    f"{os.path.join(args.output_dir, 'train_args.json')}")
        with open(os.path.join(args.output_dir, "train_args.json"), "w") as f:
            json.dump(
                {
                    key: value
                    for key, value 
                    in vars(args).items()
                    if key not in ("device", "func")
                },
                f
            )


def train_main(args):
    # *** Check output dir is ok ***
    # This must be done here cause its ok to use an already existing
    # output dir in the test_main function
    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)  # check empty
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output dir ({args.output_dir}) already exists and is not empty. "
            "Please use --overwrite-output-dir"
        )
    # *** Set the seed ***
    set_seed(args)
    # *** Load stuff ***
    model, tokenizer = load_model(args, test=False)

    # *** Train! ***
    train_dataset = load_dataset(args, tokenizer, "train")
    global_step, train_loss = train_model(args, model, train_dataset)
    save_pretrained(args, model, tokenizer)

    # *** Evaluate ! ***
    # get the gold labels from file
    gold_labels = get_gold_labels(args, "dev")
    dev_dataset = load_dataset(args, tokenizer, "dev")
    results = evaluate(args, model, dev_dataset, gold_labels)
    save_results(args, results, "dev")
    logger.info(f"Results: {results}")
    return results


def test_main(args):
    model, tokenizer = load_model(args, test=True)
    test_dataset = load_dataset(args, tokenizer, "test")
    gold_labels = get_gold_labels(args, "test")
    results = evaluate(args, model, test_dataset, gold_labels)
    save_results(args, results, "test")
    logger.info(f"Results: {results}")
    return results


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
        p.add_argument("--batch-size", default=16, type=int)

        # Arguments that should be overwritten in case a train_args.bin
        # is found in the model dir
        p.add_argument("--max-length", default=128, type=int)
        p.add_argument("--do-lower-case", action="store_true")
        p.add_argument("--remove-accents", action="store_false",
                       dest="keep_accents")
        # Other arguments
        p.add_argument(
            "--average-type",
            default="micro",
            type=str,
            choices=("micro", "macro", "weighted", "samples")
        )
        p.add_argument("--disable-cuda", action="store_true")

    # Specific for train
    parser_train.add_argument("--learn-rate", default=3e-5, type=float)
    parser_train.add_argument("--epochs", default=1, type=int)

    parser_train.add_argument("--weight-decay", default=0.01, type=float)
    parser_train.add_argument("--warmup-steps", default=0.1, type=float)

    parser_train.add_argument("--overwrite-output-dir", action="store_true")
    parser_train.add_argument("--seed", default=1234, type=int)
    parser_train.add_argument("--logging-steps", default=50, type=int)

    # Set default functions
    parser_train.set_defaults(func=train_main)
    parser_test.set_defaults(func=test_main)

    args = parser.parse_args()

    # *** Finish setting up the args ***
    # In case i forgot to add the flag
    if "uncased" in args.model_dir and not args.do_lower_case:
        option = input(
            "WARNING: --model-dir contains 'uncased' but got no "
            "--do-lower-case option.\nDo you want to continue? [Y/n] ")
        if option == "n":
            sys.exit(0)
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device("cuda")
        args.n_gpu = torch.cuda.device_count()
    else:
        args.device = torch.device("cpu")
        args.n_gpu = 0

    # *** Set up logging ***
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO
    )

    return args.func(args)


if __name__ == '__main__':
    main()