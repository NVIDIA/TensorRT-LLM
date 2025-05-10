import random

import click
from utils.abstractions import (NormalLengthDistribution,
                                UniformLengthDistribution,
                                UniformTaskIdDistribution)
from utils.export import export_workload_from_args
from utils.generate import generate_synthetic_text_dataset


@click.command()
@click.option("--num-requests",
              required=True,
              type=int,
              help="Number of requests to be generated")
@click.option("--input-mean",
              required=True,
              type=int,
              help="normal dist mean for input tokens")
@click.option("--input-stdev",
              required=True,
              type=int,
              help="normal dist stdev for input tokens")
@click.option("--output-mean",
              required=True,
              type=int,
              help="normal dist mean for output tokens")
@click.option(
    "--output-stdev",
    required=True,
    type=int,
    help="normal dist stdev for output tokens",
)
@click.pass_obj
def token_norm_dist(root_args, **kwargs):
    """Prepare synthetic dataset by generating random tokens with normal dist lengths."""
    random_source = random.Random(root_args.random_seed)
    workload = generate_synthetic_text_dataset(
        root_args.tokenizer,
        NormalLengthDistribution(
            mean=kwargs["input_mean"],
            std_dev=kwargs["input_stdev"],
        ),
        NormalLengthDistribution(
            mean=kwargs["output_mean"],
            std_dev=kwargs["output_stdev"],
        ),
        UniformTaskIdDistribution(
            min_id=root_args.task_id,
            max_id=root_args.task_id,
        ),
        kwargs["num_requests"],
        random_source,
    )

    export_workload_from_args(root_args, workload)


@click.command()
@click.option("--num-requests",
              required=True,
              type=int,
              help="Number of requests to be generated")
@click.option(
    "--input-min",
    required=True,
    type=int,
    help="uniform dist (inclusive) min for input tokens",
)
@click.option(
    "--input-max",
    required=True,
    type=int,
    help="uniform dist (inclusive) max for input tokens",
)
@click.option(
    "--output-min",
    required=True,
    type=int,
    help="uniform dist (inclusive) min for output tokens",
)
@click.option(
    "--output-max",
    required=True,
    type=int,
    help="uniform dist (inclusive) max for output tokens",
)
@click.pass_obj
def token_unif_dist(root_args, **kwargs):
    """Prepare synthetic dataset by generating random tokens with normal uniformly distributed lengths."""
    random_source = random.Random(root_args.random_seed)
    workload = generate_synthetic_text_dataset(
        root_args.tokenizer,
        UniformLengthDistribution(
            min_len=kwargs["input_min"],
            max_len=kwargs["input_max"],
        ),
        UniformLengthDistribution(
            min_len=kwargs["output_min"],
            max_len=kwargs["output_max"],
        ),
        UniformTaskIdDistribution(
            min_id=root_args.task_id,
            max_id=root_args.task_id,
        ),
        kwargs["num_requests"],
        task_id_distribution=root_args.task_id_distribution,
        random_source=random_source,
    )

    export_workload_from_args(root_args, workload)
