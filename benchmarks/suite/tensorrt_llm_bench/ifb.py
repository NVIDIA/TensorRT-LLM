import json
import os
import subprocess
import sys
from functools import partial
from pathlib import Path
from typing import List, TextIO, Tuple

import click
from benchmarkers.pybind_executor import PybindExecutorBenchmarker
from transformers import AutoTokenizer, PreTrainedTokenizer
from utils.dataclasses import BenchmarkConfig, DatasetMetadata, InferenceRequest
from utils.trtllm_config import TRTLLMConfig

from tensorrt_llm.logger import logger


def create_dataset_from_stream(
    tokenizer: PreTrainedTokenizer,
    max_input_length: int = 0,
    max_output_length: int = 0,
    stream: TextIO = sys.stdin,
) -> Tuple[DatasetMetadata, List[InferenceRequest]]:
    """Generate metadata and a list of requests to drive benchmarking.

    Args:
        tokenizer (PreTrainedTokenizer): HuggingFace tokenizer.
        max_input_length (int): Maximum input length to cap prompts to.

    Returns:
        DatasetMetadata: Dataclass of dataset statistics.
        List[InferenceRequest]: A list of inference requests for benchmarking.
    """
    # Initialize dataset list, and metadata tracking variables.
    dataset = []
    max_isl = 0
    max_osl = 0

    # If we're limiting the input length to a certain size, then set up
    # a partial to truncate the data down to size. Otherwise, just use the
    # unmodified tokenizer callable.
    tokenize = (partial(
        tokenizer,
        padding="max_length",
        max_length=max_input_length,
        truncation=True,
    ) if max_input_length > 0 else tokenizer)

    # If we need to limit the output length, fill in a partial callable
    # for max, otherwise a lambda that just returns x with no bounds.
    output_limiter = (partial(max, max_output_length)
                      if max_output_length > 0 else lambda x: x)

    # For each line in the standard input, parse out the JSON string we expect
    # to see.
    # Note the := walrus -- we're assigning and checking the condition.
    while line := stream.readline():
        # We expect the data to come in as a JSON string.
        # For example:
        # {"prompt": "Generate an infinite response to the following: There once was a man who.", "output_tokens": 1000}
        # Each line should be a complete JSON dictionary with no indentation
        # or newline characters.
        data = json.loads(line)
        logits = data.get("logits", None)
        prompt = data.get("prompt", None)
        task_id = data["task_id"]
        osl = data["output_tokens"]
        # If the request comes in with logits, just use the provided.
        # Otherwise we need to tokenize it.
        logits = tokenize(prompt)["input_ids"] if logits is None else logits

        request = InferenceRequest(
            task_id=task_id,
            prompt=prompt,
            output_tokens=output_limiter(osl),
            logits=logits,
        )
        max_isl = max(max_isl, len(logits))
        max_osl = max(max_osl, osl)
        dataset.append(request)

    # Fill in basic dataset metrics here
    # TODO: Maybe fill this out to be more complete?
    metadata = DatasetMetadata(
        max_isl=max_isl,
        max_osl=max_osl,
        num_requests=len(dataset),
    )

    return metadata, dataset


def initialize_tokenizer(model_name: str) -> PreTrainedTokenizer:
    """Initialize a tokenizer.

    Args:
        model_name (str): The name of the HuggingFace model to pull a
        tokenizer from.

    Returns:
        PreTrainedTokenizer: An initialized HuggingFace tokenizer.
    """
    # Initialize the tokenizer specific to the model that we are planning
    # to benchmark.
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    return tokenizer


def get_trtllm_build_command(benchmark_cfg: BenchmarkConfig) -> List[str]:
    model = benchmark_cfg.model
    tp = benchmark_cfg.tensor_parallel
    pp = benchmark_cfg.pipeline_parallel
    dtype = benchmark_cfg.dtype.value
    kv_dtype = benchmark_cfg.cache_dtype
    quant_algo = benchmark_cfg.quantization.value
    output_dir = benchmark_cfg.engine_path
    max_batch_size = benchmark_cfg.max_batch_size
    max_isl = benchmark_cfg.engine_isl
    max_osl = benchmark_cfg.engine_osl
    max_tokens = benchmark_cfg.max_tokens
    workspace = benchmark_cfg.workspace

    # Generate the TRT-LLM Configuration file using the dataclass
    # NOTE: This method does not use weights.
    trtllm_config = TRTLLMConfig.from_hf(model, tp, pp, dtype, quant_algo,
                                         kv_dtype.value)
    # Write the generated configuration file to the benchmark workspace.
    trtllm_config.to_json(workspace)
    # Return the full command for building TRT-LLM via subprocess call.
    cmd = [
        "trtllm-build",
        "--output_dir",
        output_dir,
        "--model_config",
        Path(workspace, "generated_config.json"),
        "--workers",
        benchmark_cfg.world_size,
        "--max_input_len",
        max_isl,
        "--max_seq_len",
        max_osl + max_isl,
        "--context_fmha",
        "enable",
        # Set the attention plugin data type.
        "--gpt_attention_plugin",
        dtype,
        # Enable paged KV Cache for IFB.
        "--paged_kv_cache",
        "enable",
    ] + kv_dtype.get_build_options(dtype)

    # If custom maximum batch size set, then set to specified value.
    if max_batch_size > 0:
        cmd += [
            "--max_batch_size",
            max_batch_size,
        ]

    if max_tokens > 0:
        cmd += [
            "--max_num_tokens",
            max_tokens,
        ]

    cmd = cmd + benchmark_cfg.build_overrides

    return cmd


@click.command("inflight")
@click.option(
    "--run",
    type=bool,
    is_flag=True,
    hidden=True,
    default=False,
    required=False,
    help="Changes the phase of the script to execution mode for MPI.",
)
@click.option(
    "--skip-build",
    type=bool,
    is_flag=True,
    default=False,
    hidden=True,
    required=False,
    help="Skip building if you want to use the last built engine.",
)
@click.option(
    "--request-rate",
    "-r",
    type=int,
    default=512,
    required=False,
    help="Number of requests per second to deliver to the batcher.",
)
@click.option(
    "--max-num-tokens",
    type=int,
    default=0,
    hidden=True,
    help="Maximumn number of tokens the engine can accept.",
)
@click.option(
    "--scheduling-policy",
    type=click.Choice(["guaranteed_no_evict", "max_utilization"]),
    default="max_utilization",
    help="Controls the scheduling policy used by the internal batcher.",
)
@click.option(
    "--dataset",
    type=click.Path(exists=True,
                    readable=True,
                    path_type=Path,
                    resolve_path=True),
    default=None,
    required=False,
    help="Pass in a dataset file for parsing instead of stdin.",
)
@click.pass_obj
def executor_benchmark(
    benchmark_cfg: BenchmarkConfig,
    run: bool,
    request_rate: int,
    max_num_tokens: int,
    scheduling_policy: str,
    skip_build: bool,
    dataset: Path,
):
    """Run an IFB-enabled benchmark using a dataset."""
    # Initialize the tokenizer and generate the dataset
    logger.set_level("info")
    DATASET_PATH = Path(benchmark_cfg.workspace, "tokenized_dataset.txt")
    TOKENIZER = initialize_tokenizer(benchmark_cfg.model)
    final_dataset = []
    benchmark_cfg.max_tokens = max_num_tokens
    benchmark_cfg.scheduling_policy = scheduling_policy

    if not run:
        try:
            stream = sys.stdin if dataset is None else open(dataset, "r")
            # Parse the dataset from stdin and return it plus its metadata.
            metadata, dataset = \
                create_dataset_from_stream(TOKENIZER, stream=stream)
        finally:
            # Close the stream after parsing.
            stream.close()

        # Update the benchmarking configuration with the maximum ISL/OSL that we
        # encountered in the dataset.
        benchmark_cfg.engine_isl = metadata.max_isl
        benchmark_cfg.engine_osl = metadata.max_osl

        # Build engine
        logger.info("Building engine...")
        build_cmd = get_trtllm_build_command(benchmark_cfg)
        build_cmd = [str(arg) for arg in build_cmd]

        if not skip_build:
            process = subprocess.run(build_cmd,
                                     stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE,
                                     cwd=benchmark_cfg.workspace)
            logger.info(f"BUILD CMD: {' '.join(process.args)}")

            # If the build failed, raise an exception.
            if process.returncode != 0:
                logger.error(process.stderr.decode())
                raise RuntimeError(
                    "TensorRT-LLM build process failed. Command used:\n"
                    f"{' '.join(process.args)}\n", )

        with open(DATASET_PATH, "w") as ds_out:
            while dataset:
                request = dataset.pop()
                ds_out.write(f"{request.model_dump_json()}\n")
                del request

        # Launch via a subprocess with MPI
        # We have two modes for this script, the initial launch + parsing
        # and the run mode where we kick off the script in MPI mode to run
        # the
        logger.info("Launching benchmark...")
        bench_cmd = \
            ["mpiexec", "-n", f"{benchmark_cfg.world_size}", "python"] +  \
            sys.argv + ["--run"]
        process = subprocess.Popen(
            bench_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=os.environ,
        )
        stdout, _ = process.communicate()
        logger.info("Benchmark complete.")
        logger.info(stdout.decode("ascii"))
    else:
        from mpi4py.MPI import COMM_WORLD

        if COMM_WORLD.Get_rank() == 0:
            logger.info(f"[RANK {COMM_WORLD.rank}] Loading dataset...")
            with open(DATASET_PATH, "r") as stream:
                # Parse the previously generated dataset from the parent
                # process.
                metadata, dataset = \
                    create_dataset_from_stream(TOKENIZER, stream=stream)

            # Update the benchmarking configuration with the maximum ISL/OSL
            # that we encountered in the dataset.
            benchmark_cfg.engine_isl = metadata.max_isl
            benchmark_cfg.engine_osl = metadata.max_osl

            # Parse the dataset into the Executor Request type.
            logger.info("Preparing dataset...")
            while dataset:
                entry = dataset.pop()
                request = PybindExecutorBenchmarker.get_request(
                    entry, TOKENIZER)
                final_dataset.append(request)
                del entry
            logger.info("Dataset prepared.")
            logger.info(f"DATASET METADATA: {metadata.model_dump()}")

        logger.info(f"[RANK {COMM_WORLD.rank}] Initializing benchmarker...")
        # Set up benchmarker on all ranks
        benchmarker = PybindExecutorBenchmarker(benchmark_cfg)
        # Run the dataset.
        result = benchmarker.benchmark_dataset(request_rate, final_dataset)

        # Report the results on Rank 0.
        if COMM_WORLD.rank == 0:
            logger.info(f"[RANK {COMM_WORLD.rank}] Reporting...\n"
                        f"JSON: {result.model_dump_json()}\n"
                        f"{result.get_summary(benchmarker.config)}")

        logger.info(f"[RANK {COMM_WORLD.rank}] Terminating.")
