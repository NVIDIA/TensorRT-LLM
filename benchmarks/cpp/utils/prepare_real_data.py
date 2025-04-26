import random
import re
from typing import Optional

import click
from utils.abstractions import (DatasetConfig, LengthDistribution,
                                NormalLengthDistribution)
from utils.export import export_workload_from_args
from utils.generate import generate_real_dataset


def validate_output_len_dist(
        value: Optional[str]) -> Optional[LengthDistribution]:
    """Validate the --output-len-dist option."""
    if value is None:
        return value
    m = re.match(r"(\d+),(\d+)", value)
    if m:
        return NormalLengthDistribution(
            mean=int(m.group(1)),
            std_dev=int(m.group(2)),
        )
    else:
        raise AssertionError(
            "Incorrect specification for --output-len-dist. Correct format: --output-len-dist <output_len_mean>,<output_len_stdev>"
        )


@click.command()
@click.option("--dataset-name",
              required=True,
              type=str,
              help=f"Dataset name in HuggingFace.")
@click.option(
    "--dataset-config-name",
    type=str,
    default=None,
    help=f"Dataset config name in HuggingFace (if exists).",
)
@click.option("--dataset-split",
              type=str,
              required=True,
              help=f"Split of the dataset to use.")
@click.option("--dataset-input-key",
              type=str,
              help=f"The dataset dictionary key for input.")
@click.option(
    "--dataset-image-key",
    type=str,
    default="image",
    help=f"The dataset dictionary key for images.",
)
@click.option(
    "--dataset-prompt-key",
    type=str,
    default=None,
    help=f"The dataset dictionary key for prompt (if exists).",
)
@click.option(
    "--dataset-prompt",
    type=str,
    default=None,
    help=f"The prompt string when there is no prompt key for the dataset.",
)
@click.option(
    "--dataset-output-key",
    type=str,
    default=None,
    help=f"The dataset dictionary key for output (if exists).",
)
@click.option(
    "--num-requests",
    type=int,
    default=None,
    help=
    "Number of requests to be generated. Will be capped to min(dataset.num_rows, num_requests).",
)
@click.option(
    "--max-input-len",
    type=int,
    default=None,
    help=
    "Maximum input sequence length for a given request. This will be used to filter out the requests with long input sequence length. Default will include all the requests.",
)
@click.option(
    "--output-len-dist",
    type=str,
    default=None,
    help=
    "Output length distribution. Default will be the length of the golden output from the dataset. Format: <output_len_mean>,<output_len_stdev>. E.g. 100,10 will randomize the output length with mean=100 and variance=10.",
)
@click.pass_obj
def dataset(root_args, **kwargs):
    """Prepare dataset from real dataset."""
    dataset_config = DatasetConfig(**{
        k[8:]: v
        for k, v in kwargs.items() if k.startswith("dataset_")
    })
    random_source = random.Random(root_args.random_seed)
    output_length_distribution = (validate_output_len_dist(
        kwargs["output_len_dist"]) if kwargs["output_len_dist"] else None)
    workload = generate_real_dataset(
        dataset_config,
        root_args.tokenizer,
        kwargs["max_input_len"],
        output_length_distribution,
        root_args.task_id_distribution,
        kwargs["num_requests"],
        random_source,
    )
    export_workload_from_args(root_args, workload)
