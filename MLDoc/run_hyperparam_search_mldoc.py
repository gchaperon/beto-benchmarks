import argparse
import os
from itertools import product
import finetune_mldoc

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


LEARN_RATES = [2e-5, 3e-5, 5e-5]
BATCH_SIZES = [16, 32]
EPOCHS = [3, 4]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True, type=str)
    parser.add_argument("--data-dir", required=True, type=str)
    parser.add_argument("--output-dir", required=True, type=str)

    parser.add_argument("--do-lower-case", action="store_true")

    args = parser.parse_args()
    # breakpoint()
    with open(
        os.path.join(
            args.output_dir,
            f"results_{os.path.basename(args.model_dir)}_mldoc.txt",
        ),
        "w",
    ) as out_f:
        # breakpoint()
        for i, (learn_rate, batch_size, epochs) in enumerate(
            product(LEARN_RATES, BATCH_SIZES, EPOCHS)
        ):

            logger.info(f"Running experiment {i+1}")

            experiment_args = [
                "--model-dir",
                args.model_dir,
                "--data-dir",
                args.data_dir,
                "--output-dir",
                os.path.join(
                    args.output_dir,
                    f"experiment_{i+1}_{os.path.basename(args.model_dir)}_mldoc",
                ),
                "--learn-rate",
                learn_rate,
                "--batch-size",
                batch_size,
                "--epochs",
                epochs,
                "--overwrite-cache",
            ]
            experiment_args += (
                ["--do-lower-case"] if args.do_lower_case else []
            )

            # breakpoint()
            experiment_args = list(map(str, experiment_args))
            results = finetune_mldoc.main(experiment_args)
            out_f.write(f"Experiment {i + 1}\n")
            out_f.write(
                f"\tlr={learn_rate}, batch={batch_size}, epochs={epochs}\n"
            )
            out_f.write(f"\t{results}\n\n")


if __name__ == "__main__":
    main()
