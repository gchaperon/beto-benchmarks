import csv
from transformers import DataProcessor, InputExample


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

    def __init__(self, train_language="es", eval_language=None):
        self.train_language = train_language
        self.eval_language = eval_language if eval_language else train_language

    def get_train_examples(self, train_file_path):
        rows = self._read_xnli_tsv(train_file_path)
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

    def get_eval_examples(self, eval_file_path):
        rows = self._read_xnli_tsv(eval_file_path)
        examples = [
            InputExample(
                guid=f"eval-{i}",
                text_a=row["sentence1"],
                text_b=row["sentence2"],
                label=row["gold_label"]
            )
            for i, row
            in enumerate(rows)
            if row["language"] == self.eval_language
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
            return list(reader)

    @classmethod
    def get_labels(cls):
        return ["contradiction", "entailment", "neutral"]


class MLDocProcessor(DataProcessor):
    ...

    def __init__(self):
        ...
