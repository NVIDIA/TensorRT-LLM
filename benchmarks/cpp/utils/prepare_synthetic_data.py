import click
from utils.utils import (dataset_dump, gen_random_tokens, get_list_of_delays,
                         get_norm_dist_tokens)


@click.command()
@click.option("--num-requests",
              required=True,
              type=int,
              help='Number of requests to be generated')
@click.option('--input-mean',
              required=True,
              type=int,
              help='normal dist mean for input tokens')
@click.option('--input-stdev',
              required=True,
              type=int,
              help='normal dist stdev for input tokens')
@click.option('--output-mean',
              required=True,
              type=int,
              help='normal dist mean for output tokens')
@click.option('--output-stdev',
              required=True,
              type=int,
              help='normal dist stdev for output tokens')
@click.pass_obj
def token_norm_dist(root_args, **kwargs):
    """Prepare dataset by generating random tokens."""
    input_ids = []
    input_lens = []
    output_lens = []

    input_lens = get_norm_dist_tokens(kwargs['input_mean'],
                                      kwargs['input_stdev'],
                                      kwargs['num_requests'],
                                      root_args.random_seed)

    num_reqs = len(input_lens)
    output_lens = get_norm_dist_tokens(kwargs['output_mean'],
                                       kwargs['output_stdev'], num_reqs,
                                       root_args.random_seed)
    delays = get_list_of_delays(root_args.time_delay_dist,
                                root_args.mean_time_bet_reqs, num_reqs,
                                root_args.random_seed)

    input_ids = gen_random_tokens(input_lens, root_args.tokenizer,
                                  root_args.random_seed)

    dataset_dump(
        input_ids, output_lens, delays, {
            "workload_type": "token-norm-dist",
            "input_mean": kwargs['input_mean'],
            "input_stdev": kwargs['input_stdev'],
            "output_mean": kwargs['output_mean'],
            "output_stdev": kwargs['output_stdev'],
            "num_requests": kwargs['num_requests'],
            "delay_distr": root_args.time_delay_dist,
            "request_rate": root_args.request_rate,
            "tokenize_vocabsize": root_args.tokenizer.vocab_size
        }, root_args.output)
