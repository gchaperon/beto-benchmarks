import argparse
import logging
import os
import random
import sys
from collections import namedtuple
from datetime import datetime
from enum import Enum, auto
from itertools import zip_longest
from operator import itemgetter
from platform import node

import torch
from conllu import parse_incr
from model import BertForDependencyParsing
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import (
    BertModel,
    BertTokenizer,
    get_linear_schedule_with_warmup,
)

logger = logging.getLogger("__name__")


class Stage(Enum):
    TRAIN = auto()
    DEV = auto()
    TEST = auto()

    def __str__(self):
        return self.name.lower()


# Id of the head of subword tokens (ignored)
PAD_ID = -1
PAD_UPOSTAG = PAD_DEPREL = ""
IGNORED_POS = ["PUNCT", "SYM"]
EXAMPLE_FIELDS = ["id", "form", "upostag", "head", "deprel"]

PAD_DEPREL_ID = PAD_HEAD = CrossEntropyLoss().ignore_index


class Example(
    namedtuple(
        "Example",
        [*(field + "s" for field in EXAMPLE_FIELDS), "prediction_mask"],
    )
):
    __slots__ = ()

    def __str__(self):
        return (
            "Example(\n"
            f"ids={self.ids}\n"
            f"forms={self.forms}\n"
            f"upostags={self.upostags}\n"
            f"heads={self.heads}\n"
            f"deprels={self.deprels}\n"
            f"prediction_mask={self.prediction_mask}\n"
            ")"
        )


FILE_TEMPLATES = [
    os.path.join("UD_Spanish-AnCora", "es_ancora-ud-{}.conllu"),
    os.path.join("UD_Spanish-GSD", "es_gsd-ud-{}.conllu"),
]


def load_examples(args, tokenizer, stage):
    """
    This will be kinda slow in favor of readability
    """

    sentences = []
    for template in FILE_TEMPLATES:
        file_name = template.format(stage)
        with open(os.path.join(args.data_dir, file_name)) as file:
            for sentence in tqdm(
                parse_incr(file), desc=f"Reading {file_name}"
            ):
                sentences.append(sentence)
    # logger.info(len([token for sentence in sentences for token in sentence]))
    filtered = []
    for sentence in tqdm(sentences, desc="Filtering sentences"):
        filtered_sentence = [
            token
            for token in sentence
            if token["upostag"] not in IGNORED_POS
            and isinstance(token["id"], int)
        ]
        # Apparently there is a sentence that is only a dot
        if filtered_sentence:
            filtered.append(filtered_sentence)

    logger.info(
        "Overflowing: "
        f"{len([sentence for sentence in filtered if len(sentence) > 140])}"
    )

    examples = []
    for sentence in tqdm(filtered, desc="Creating examples"):
        tokenized_tuples = []
        for id_, form, upostag, head, deprel in map(
            itemgetter(*EXAMPLE_FIELDS), sentence
        ):
            for i, token in enumerate(tokenizer.tokenize(form)):
                if i < 1:
                    tokenized_tuples.append(
                        (id_, token, upostag, head, deprel, 1)
                    )
                else:
                    tokenized_tuples.append(
                        (PAD_ID, token, PAD_UPOSTAG, PAD_HEAD, PAD_DEPREL, 0)
                    )

        # breakpoint()
        examples.append(Example(*zip(*tokenized_tuples)))
        # examples.append()
        # id_, form, head, deprel =
        # for token in tokenizer.tokenize(word["form"]):
        #

    unique_postags = list(
        {tag for example in examples for tag in example.upostags}
    )
    unique_deprels = list(
        {rel for example in examples for rel in example.deprels}
    )

    return examples, unique_postags, unique_deprels


def load_dataset(args, tokenizer, stage):

    examples, unique_postags, unique_deprels = load_examples(
        args, tokenizer, stage
    )
    # breakpoint()
    pos_mapping = {tag: i for i, tag in enumerate(unique_postags)}
    rel_mapping = {rel: i for i, rel in enumerate(unique_deprels)}
    rel_mapping[PAD_DEPREL] = PAD_DEPREL_ID

    (
        input_ids,
        pos_tags_ids,
        attention_masks,
        rels_ids,
        headss,
        prediction_masks,
    ) = ([], [], [], [], [], [])
    for (ids, forms, upostags, heads, deprels, prediction_mask) in tqdm(
        examples, desc="Converting examples"
    ):
        token_ids = tokenizer.convert_tokens_to_ids(forms)
        tags_ids = [pos_mapping[tag] for tag in upostags]
        deprels_ids = [rel_mapping[rel] for rel in deprels]

        pad_length = args.max_length - len(token_ids)

        token_ids = (
            token_ids[: args.max_length]
            + [tokenizer.pad_token_id] * pad_length
        )
        tags_ids = (
            tags_ids[: args.max_length]
            + [pos_mapping[PAD_UPOSTAG]] * pad_length
        )
        heads = list(heads)[: args.max_length] + [PAD_HEAD] * pad_length
        deprels_ids = (
            deprels_ids[: args.max_length] + [PAD_DEPREL_ID] * pad_length
        )

        prediction_mask = (
            list(prediction_mask)[: args.max_length] + [0] * pad_length
        )

        attention_mask = [1] * (
            len(ids) if len(ids) <= args.max_length else args.max_length
        ) + [0] * pad_length

        input_ids.append(token_ids)
        pos_tags_ids.append(tags_ids)
        attention_masks.append(attention_mask)
        rels_ids.append(deprels_ids)
        headss.append(heads)
        prediction_masks.append(prediction_mask)

    for ll in (
        input_ids,
        pos_tags_ids,
        attention_masks,
        rels_ids,
        headss,
        prediction_masks,
    ):
        for l in ll:
            assert len(l) == args.max_length

    dataset = TensorDataset(
        torch.tensor(input_ids),
        torch.tensor(pos_tags_ids),
        torch.tensor(attention_masks),
        torch.tensor(rels_ids),
        torch.tensor(headss),
        torch.tensor(prediction_masks),
    )
    # Display random example
    k = random.randrange(len(examples))
    logger.info(f"Random example:\n{examples[k]}")
    logger.info(f"Features:\n{dataset[k]}")
    # breakpoint()
    return dataset, unique_postags, unique_deprels, pos_mapping, rel_mapping


def train(args, model, dataset):
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    total_steps = len(dataloader) * args.epochs

    criterion = CrossEntropyLoss()
    no_decay = ["bias", "LayerNorm.weight"]
    grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
    ]
    optimizer = Adam(
        grouped_parameters, lr=args.learn_rate, weight_decay=args.weight_decay
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(args.warmup_steps * total_steps),  # warmup is a %
        num_training_steps=total_steps,
    )

    tb_writer = SummaryWriter(
        log_dir="runs/deppar_{}_{}".format(
            datetime.now().strftime("%b%d_%H-%M"), node()
        )
    )

    # Finally train
    global_step, tr_loss, running_loss = 0, 0.0, 0.0
    for _ in tqdm(range(args.epochs), desc="Epoch"):
        for (
            step,
            (
                input_ids,
                pos_tags_ids,
                attention_mask,
                rels_ids,
                heads,
                prediction_mask,
            ),
        ) in enumerate(
            tqdm(
                ((t.to(args.device) for t in batch) for batch in dataloader),
                desc="Iteration",
                total=len(dataloader),
            )
        ):
            model.train()

            model.zero_grad()

            s_arc, s_rel = model(
                input_ids=input_ids,
                pos_tags_ids=pos_tags_ids,
                attention_mask=attention_mask,
                prediction_mask=prediction_mask,
            )

            # Its a bit more complicated to compute the loss in this case
            # breakpoint()
            # TODO: compute loss

            # Ignore values that are outside of range (?)
            # I'm not so sure about the behaviour of view()
            heads[heads >= args.max_length] = criterion.ignore_index
            arc_loss = criterion(
                s_arc.view(-1, args.max_length), heads.view(-1)
            )

            rels_ids[heads == criterion.ignore_index] = criterion.ignore_index
            # breakpoint()

            eii = s_rel[
                torch.arange(s_rel.shape[0]).view(-1, 1),
                torch.arange(args.max_length),
                heads.clamp(min=0, max=args.max_length - 1),
            ].view(-1, s_rel.shape[-1])

            # breakpoint()
            rel_loss = criterion(eii, rels_ids.view(-1))

            loss = arc_loss + rel_loss

            # if args.n_gpu > 1:
            #     loss = loss.mean()

            loss.backward()
            optimizer.step()
            scheduler.step()

            tr_loss += loss.item()
            running_loss += loss.item()

            if (
                args.logging_steps > 0
                and global_step % args.logging_steps == 0
            ):
                tb_writer.add_scalar(
                    "lr", scheduler.get_lr()[0], global_step
                )
                tb_writer.add_scalar(
                    "loss", running_loss / args.logging_steps, global_step
                )
                running_loss = 0.0

            global_step += 1

    tb_writer.close()

    return global_step, tr_loss / global_step


def train_main(args):
    # breakpoint()

    tokenizer = BertTokenizer.from_pretrained(
        args.model_dir, do_lower_case=args.do_lower_case
    )
    # TODO: cambiar stage

    dataset, unique_postags, unique_deprels, *_ = load_dataset(
        args, tokenizer, Stage.TEST
    )
    # TODO: aprender a usar torchtext

    model = BertForDependencyParsing.from_pretrained(
        args.model_dir,
        n_pos_tags=len(unique_postags),
        n_rels=len(unique_deprels),
    ).to(args.device)

    asd = train(args, model, dataset)
    breakpoint()
    # input_ids = dataset[0][0].unsqueeze(0)
    # pos_tags_ids = dataset[0][1].unsqueeze(0)
    # attention_mask = dataset[0][2].unsqueeze(0)
    # prediction_mask = dataset[0][5].unsqueeze(0)
    # breakpoint()
    # outputs = model(
    #     input_ids=input_ids,
    #     pos_tags_ids=pos_tags_ids,
    #     attention_mask=attention_mask,
    #     prediction_mask=prediction_mask,
    # )
    # breakpoint()


def test_main(args):
    ...


def main(passed_args=None):
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(title="commands")
    parser_train = subparsers.add_parser("train")
    parser_test = subparsers.add_parser("test")

    # Common arguments
    for p in [parser_train, parser_test]:
        p.add_argument("--data-dir", default="./ud-treebanks-v2.2", type=str)
        p.add_argument("--model-dir", default="../beto-cased", type=str)
        p.add_argument("--do-lower-case", action="store_true")
        p.add_argument("--max-length", default=128, type=int)
        p.add_argument("--batch-size", default=16, type=int)

        # Other arguments
        p.add_argument("--disable-cuda", action="store_true")

    # Specific for train
    parser_train.add_argument("--learn-rate", default=3e-5, type=float)
    parser_train.add_argument("--epochs", default=1, type=int)

    parser_train.add_argument("--weight-decay", default=0.01, type=float)
    parser_train.add_argument("--warmup-steps", default=0.1, type=float)
    parser_train.add_argument("--logging-steps", default=50, type=int)

    # Set default functions
    parser.set_defaults(func=lambda *args: parser.print_help())
    parser_train.set_defaults(func=train_main)
    parser_test.set_defaults(func=test_main)

    args = parser.parse_args(passed_args)

    # *** Finish setting up the args ***
    # In case i forgot to add the flag
    if "uncased" in args.model_dir and not args.do_lower_case:
        option = input(
            "WARNING: --model-dir contains 'uncased' but got no "
            "--do-lower-case option.\nDo you want to continue? [Y/n] "
        )
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
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    args.func(args)


if __name__ == "__main__":
    main()
