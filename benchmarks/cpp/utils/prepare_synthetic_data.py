import random
import warnings

import click
from utils.utils import (gen_random_tokens, get_norm_dist_lengths,
                         get_unif_dist_lengths, print_text_dataset,
                         text_dataset_dump)


def _generate_task_ids_and_lora_config(root_args, num_reqs):
    """Generate task IDs and determine LoRA configuration based on root_args."""
    if root_args.rand_task_id is None:
        task_ids = [root_args.task_id for _ in range(num_reqs)]
    else:
        min_id, max_id = root_args.rand_task_id
        task_ids = [random.randint(min_id, max_id) for _ in range(num_reqs)]

    use_task_ids = root_args.task_id != -1 or root_args.rand_task_id is not None

    # Determine if LoRA should be used (requires both task IDs and lora_dir)
    use_lora = use_task_ids and root_args.lora_dir is not None

    # Warn if task IDs are specified but no LoRA directory is provided
    if use_task_ids and not use_lora:
        warnings.warn(
            "Task IDs require LoRA directory. Use --lora-dir or omit task IDs.",
            UserWarning)

    return (task_ids, task_ids if use_task_ids else None, {
        "lora_dir": root_args.lora_dir
    } if use_lora else None)


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
    """Prepare synthetic dataset by generating random tokens with normal dist lengths."""
    input_ids = []
    input_lens = []
    output_lens = []

    input_lens = get_norm_dist_lengths(kwargs['input_mean'],
                                       kwargs['input_stdev'],
                                       kwargs['num_requests'],
                                       root_args.random_seed)

    num_reqs = len(input_lens)
    output_lens = get_norm_dist_lengths(kwargs['output_mean'],
                                        kwargs['output_stdev'], num_reqs,
                                        root_args.random_seed)

    max_input_len = max(input_lens)
    max_output_len = max(output_lens)

    input_ids = gen_random_tokens(input_lens, root_args.tokenizer,
                                  root_args.random_seed)

    task_ids, print_task_ids, lora_config = _generate_task_ids_and_lora_config(
        root_args, num_reqs)

    if not root_args.std_out:
        text_dataset_dump(
            input_lens, input_ids, output_lens, task_ids, {
                "workload_type": "token-norm-dist",
                "input_mean": kwargs['input_mean'],
                "input_stdev": kwargs['input_stdev'],
                "output_mean": kwargs['output_mean'],
                "output_stdev": kwargs['output_stdev'],
                "num_requests": kwargs['num_requests'],
                "tokenize_vocabsize": root_args.tokenizer.vocab_size,
                "max_input_len": max_input_len,
                "max_output_len": max_output_len
            }, root_args.output)
    else:
        print_text_dataset(input_ids,
                           output_lens,
                           task_ids=print_task_ids,
                           lora_config=lora_config)


@click.command()
@click.option("--num-requests",
              required=True,
              type=int,
              help='Number of requests to be generated')
@click.option('--input-min',
              required=True,
              type=int,
              help='uniform dist (inclusive) min for input tokens')
@click.option('--input-max',
              required=True,
              type=int,
              help='normal dist (inclusive) max for input tokens')
@click.option('--output-min',
              required=True,
              type=int,
              help='normal dist (inclusive) min for output tokens')
@click.option('--output-max',
              required=True,
              type=int,
              help='normal dist (inclusive) max for output tokens')
@click.pass_obj
def token_unif_dist(root_args, **kwargs):
    """Prepare synthetic dataset by generating random tokens with normal uniformly lengths."""
    input_ids = []
    input_lens = []
    output_lens = []

    input_lens = get_unif_dist_lengths(kwargs['input_min'], kwargs['input_max'],
                                       kwargs['num_requests'],
                                       root_args.random_seed)

    num_reqs = len(input_lens)
    output_lens = get_unif_dist_lengths(kwargs['output_min'],
                                        kwargs['output_max'], num_reqs,
                                        root_args.random_seed)

    max_input_len = max(input_lens)
    max_output_len = max(output_lens)

    input_ids = gen_random_tokens(input_lens, root_args.tokenizer,
                                  root_args.random_seed)

    task_ids, print_task_ids, lora_config = _generate_task_ids_and_lora_config(
        root_args, num_reqs)

    if not root_args.std_out:
        text_dataset_dump(
            input_lens, input_ids, output_lens, task_ids, {
                "workload_type": "token-unif-dist",
                "input_min": kwargs['input_min'],
                "input_max": kwargs['input_max'],
                "output_min": kwargs['output_min'],
                "output_max": kwargs['output_max'],
                "num_requests": kwargs['num_requests'],
                "tokenize_vocabsize": root_args.tokenizer.vocab_size,
                "max_input_len": max_input_len,
                "max_output_len": max_output_len
            }, root_args.output)
    else:
        print_text_dataset(input_ids,
                           output_lens,
                           task_ids=print_task_ids,
                           lora_config=lora_config)
