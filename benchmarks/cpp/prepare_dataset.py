from typing import Literal

import click
from pydantic import BaseModel, field_validator
from transformers import AutoTokenizer
from utils.prepare_real_data import dataset
from utils.prepare_synthetic_data import token_norm_dist
from utils.utils import get_req_time_interval


class RootArgs(BaseModel):
    tokenizer: str
    output: str
    request_rate: float
    mean_time_bet_reqs: float
    time_delay_dist: Literal["constant", "exponential_dist"]
    random_seed: int

    @field_validator('tokenizer')
    def get_tokenizer(cls, v: str):
        try:
            tokenizer = AutoTokenizer.from_pretrained(v, padding_side='left')
        except EnvironmentError:
            raise ValueError(
                "Cannot find a tokenizer from the given string. Please set tokenizer to the directory that contains the tokenizer, or set to a model name in HuggingFace."
            )
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer


@click.group()
@click.option(
    "--tokenizer",
    required=True,
    type=str,
    help=
    "Tokenizer dir for the model run by gptManagerBenchmark, or the model name from HuggingFace."
)
@click.option("--output",
              type=str,
              help="Output json filename.",
              default="preprocessed_dataset.json")
@click.option(
    "--request-rate",
    type=float,
    help="# of reqs/sec. -1 indicates Speed of Light/Zero-delay injection rate",
    default=-1.0)
@click.option("--time-delay-dist",
              type=click.Choice(["constant", "exponential_dist"]),
              help="Distribution of the time delay.",
              default="exponential_dist")
@click.option(
    "--random-seed",
    required=False,
    type=int,
    help=
    "random seed for exponential delays (dataset/norm-token-dist) and token_ids(norm-token-dist)",
    default=420)
@click.pass_context
def cli(ctx, **kwargs):
    """This script generates dataset input for gptManagerBenchmark."""
    ctx.obj = RootArgs(tokenizer=kwargs['tokenizer'],
                       output=kwargs['output'],
                       request_rate=kwargs['request_rate'],
                       mean_time_bet_reqs=get_req_time_interval(
                           kwargs['request_rate']),
                       time_delay_dist=kwargs['time_delay_dist'],
                       random_seed=kwargs['random_seed'])


cli.add_command(dataset)
cli.add_command(token_norm_dist)

if __name__ == "__main__":
    cli()
