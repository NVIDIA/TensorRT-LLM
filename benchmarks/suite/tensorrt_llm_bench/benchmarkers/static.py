import platform
from pathlib import Path
from subprocess import CompletedProcess
from typing import Dict, List

from utils import command_logger, process_error_check, run_process
from utils.dataclasses import BenchmarkConfig, BenchmarkResults
from utils.trtllm_config import TRTLLMConfig


class gptSessionBenchmarker:
    """Utility class for running static benchmarks with gptSessionBenchmark."""

    def __init__(
        self,
        config: BenchmarkConfig,
        benchmark_binary: Path,
        batch_size: int,
        isl: int,
        osl: int,
        warm_up_runs: int,
        num_runs: int,
        duration: int,
        kv_cache_free_fraction: float = .9,
    ):
        """Initialize a gptSessionBenchmark instance.

        Args:
            config (BenchmarkConfig): Benchmark configuration for build/run.
            benchmark_binary (Path): Path to the benchmarking binary.
            batch_size (int): Batch size to configure the build with.
            isl (int): Input sequence length to configure the build with.
            osl (int): Output sequence length to configure the build with.
            kv_cache_free_fraction (float, optional): The amount of remaining
            GPU memory after model loading to save for the KV Cache. Defaults
            to .9.
        """
        self.config: BenchmarkConfig = config
        self.gpt_session_path = Path(benchmark_binary).absolute()
        self.batch_size = batch_size
        self.input_length = isl
        self.output_length = osl
        self.warm_up = warm_up_runs
        self.num_runs = num_runs
        self.duration = duration
        self.kv_cache_mem = kv_cache_free_fraction
        self.result = None

    def get_build_command(self) -> List[str]:
        """Build the engine command for TRT-LLM.

        Returns:
            List[str]: A list of command line arguments to run a build command.
        """
        model = self.config.model
        tp = self.config.tensor_parallel
        pp = self.config.pipeline_parallel
        dtype = self.config.dtype.value
        kv_dtype = self.config.cache_dtype
        quant_algo = self.config.quantization.value
        output_dir = self.config.engine_path
        max_batch_size = self.batch_size
        max_isl = self.input_length
        max_osl = self.output_length
        workspace = self.config.workspace

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
            self.config.world_size,
            # Define the maximums the engine can accept.
            "--max_batch_size",
            max_batch_size,
            "--max_input_len",
            max_isl,
            "--max_seq_len",
            max_osl + max_isl,
            "--context_fmha",
            "enable",
            # Set the attention plugin data type.
            "--gpt_attention_plugin",
            dtype,
            # Disable paged cache since we aren't batching on the fly.
            "--paged_kv_cache",
            "disable",
        ] + kv_dtype.get_build_options(dtype)

        return [str(arg) for arg in cmd]

    @command_logger(prefix="BUILD COMMAND: ")
    @process_error_check
    def _run_build(self, cmd: List[str]) -> CompletedProcess:
        """Wrapper for calling the build for TRT-LLM.

        Purpose of this wrapper is so that we can decorate it/log it.

        Args:
            cmd (List[str]): List of command line arguments for running.

        Returns:
            CompletedProcess: Completed process information for parsing and
            reporting.
        """
        return run_process(
            cmd,
            self.config.workspace,
        )

    def build(self) -> None:
        """Build the engine for benchmarking."""
        self._run_build(self.get_build_command())

    @command_logger(prefix="BENCHMARK COMMAND: ")
    @process_error_check
    def _run_benchmark(self, cmd: List[str]) -> CompletedProcess:
        """Run the benchmark command in the configured workspace.

        Args:
            cmd (List[str]): List of command line arguments to run via
            subprocess.

        Returns:
            CompletedProcess: Completed process information for reporting.
        """
        return run_process(cmd, run_dir=self.config.workspace, use_environ=True)

    @staticmethod
    def parse_benchmark_result(benchmark_line: str) -> Dict[str, str]:
        pass

    def benchmark(self):
        """Benchmarks a TRT-LLM for a configured instance."""

        # Compile the command for running
        cmd = ["mpiexec", "-n", self.config.world_size]
        cmd += ["-allow-run-as-root"] if platform.system() != "Windows" else ""
        cmd += [
            self.gpt_session_path,
            "--engine_dir",
            self.config.engine_path,
            "--batch_size",
            self.batch_size,
            "--log_level",
            "info",
            "--kv_cache_free_gpu_mem_fraction",
            self.kv_cache_mem,
            "--beam_width",
            "1",
            "--warm_up",
            self.warm_up,
            "--num_runs",
            self.num_runs,
            "--duration",
            self.duration,
            "--input_output_len",
            f"{self.input_length},{self.output_length};{self.input_length},1",
        ]
        cmd = [str(arg) for arg in cmd]
        # Run the benchmark using the provided gptSession benchmark binary.
        bench_return = self._run_benchmark(cmd)
        results = [
            x.split(" ") for x in bench_return.stdout.split("\n")
            if "[BENCHMARK]" in x
        ]

        ttft = float(results[1][8])
        gen_time = float(results[0][8]) - ttft
        total_out = int(results[0][2]) * int(results[0][6])
        total_in = int(results[0][2]) * int(results[0][4])
        batch_size = int(results[0][2])

        bench_result = BenchmarkResults(
            model=self.config.model,
            dtype=self.config.dtype.value,
            quantization=str(self.config.quantization.value),
            max_batch_size=batch_size,
            total_input_tokens=total_in,
            total_output_tokens=total_out,
            tp_size=self.config.tensor_parallel,
            pp_size=self.config.pipeline_parallel,
            kv_mem_fraction=self.kv_cache_mem,
            scheduler="Static",
            inflight_batching=False,
            total_latency=results[0][8],
            first_token_latency=ttft,
            time_per_output_token=gen_time / (total_out - batch_size),
            latency_units="ms",
            throughput=results[0][10],
            throughput_units="tokens/second",
            peak_gpu_mem=results[0][16],
            peak_gpu_mem_units="GB",
            binary=str(self.gpt_session_path),
            build_cmd=" ".join(self.get_build_command()),
            benchmark_cmd=" ".join(cmd))

        return bench_result
