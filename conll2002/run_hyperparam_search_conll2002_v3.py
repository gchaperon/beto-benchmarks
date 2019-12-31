import argparse
import os
from itertools import product
import run_conll2002_v3

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO
)

# *** Hyper param search ***
LEARN_RATES = [2e-5, 3e-5, 5e-5]
BATCH_SIZE = [16, 32]
EPOCHS = [3, 4]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True, type=str)
    parser.add_argument("--data-dir", required=True, type=str)
    parser.add_argument("--output-dir", required=True, type=str)

    parser.add_argument("--do-lower-case", action="store_true")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    model_name = os.path.basename(os.path.normpath(args.model_dir))
    results_file = os.path.join(
        args.output_dir,
        f"dev_results_{model_name}_conll2002.txt"
    )
    with open(results_file, "w", buffering=1) as out_f:
        for i, (learn_rate, batch_size, epochs) \
                in enumerate(product(LEARN_RATES, BATCH_SIZE, EPOCHS)):
            logger.info(f"Running experiment {i+1}")

            experiment_output_dir = os.path.join(
                args.output_dir,
                f"experiment_{i+1}_{model_name}_conll2002"
            )
            experiment_args = [
                "train",
                "--model-dir", args.model_dir,
                "--data-dir", args.data_dir,
                "--output-dir", experiment_output_dir,

                "--learn-rate", learn_rate,
                "--batch-size", batch_size,
                "--epochs", epochs,
            ]
            if args.do_lower_case:
                experiment_args += ["--do-lower-case"]

            experiment_args = list(map(str, experiment_args))
            # breakpoint()
            results = run_conll2002_v3.main(experiment_args)

            out_f.write(f"Experiment {i + 1}\n")
            out_f.write(
                f"\tlr={learn_rate}, batch={batch_size}, epochs={epochs}\n")
            out_f.write(f"\t{results}\n\n")


if __name__ == '__main__':
    main()
