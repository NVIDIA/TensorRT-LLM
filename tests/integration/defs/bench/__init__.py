import os
import tempfile
from pathlib import Path
from typing import Any, Optional

from ..conftest import llm_models_root
from ..trt_test_alternative import check_call, check_output


# TODO replace the trtllm_bench_prolog
class BenchRunner:

    def __init__(self,
                 llm_root: str,
                 llm_venv: Any,
                 model_subdir: str,
                 model_name: str,
                 streaming: bool,
                 tp_size: int,
                 use_pytorch_backend: bool = False,
                 skip_engine_build: bool = False,
                 quant: Optional[str] = None,
                 extra_llm_api_options: Optional[str] = None,
                 use_mpirun: bool = False):

        llm_models = llm_models_root()
        assert llm_models is not None
        self.llm_root = llm_root
        self.llm_venv = llm_venv
        self.model_path = Path(llm_models, model_subdir).absolute()
        self.model_name = model_name
        self.quant = quant
        self.streaming = streaming
        self.skip_engine_build = skip_engine_build
        self.use_pytorch_backend = use_pytorch_backend
        self.use_mpirun = use_mpirun
        self.tp_size = tp_size
        self.quant_name = self.quant if self.quant is not None else "FP16"
        self.extra_llm_api_options = extra_llm_api_options

        self.work_dir = Path(tempfile.TemporaryDirectory().name)

        self.dataset_path = os.path.join(self.work_dir, f"data.txt")
        if self.use_mpirun:
            self.mpirun_cmd = f"mpirun --allow-run-as-root -n {self.tp_size} trtllm-llmapi-launch"
        else:
            self.mpirun_cmd = ""
        self.engine_path = None

    def __call__(self):
        self.prepare_dataset()
        if not (self.skip_engine_build or self.use_pytorch_backend):
            self.build_engine()
        self.run_bench()

    def prepare_dataset(self):
        dataset_tool = Path(self.llm_root, "benchmarks", "cpp",
                            "prepare_dataset.py")

        # Generate a small dataset to run a test.
        self.work_dir.mkdir(parents=True)
        command = [
            f"{dataset_tool.resolve()}",
            "--stdout",
            "--tokenizer",
            f"{self.model_path}",
            "token-norm-dist",
            "--input-mean",
            "128",
            "--output-mean",
            "128",
            "--input-stdev",
            "0",
            "--output-stdev",
            "0",
            "--num-requests",
            "10",
        ]
        print(f"Running command: {' '.join(command)}")
        dataset_output = self.llm_venv.run_cmd(
            command,
            caller=check_output,
        )
        # Grab the stdout and write it to a dataset file for passing to suite.
        with open(self.dataset_path, "w") as dataset:
            dataset.write(dataset_output)

    def build_engine(self):
        if self.skip_engine_build:
            return

        build_cmd = \
            f"{self.mpirun_cmd} " \
            f"trtllm-bench " \
            f"--model {self.model_name} " \
            f"--model_path {self.model_path} " \
            f"--workspace {self.work_dir} " \
            f"build --tp_size {self.tp_size}"

        if self.quant is not None:
            build_cmd = f"{build_cmd} --quantization {self.quant}"

        build_cmd = f"{build_cmd} --dataset {self.dataset_path}"
        build_output = check_output(build_cmd,
                                    shell=True,
                                    env=self.llm_venv._new_env)

        for line in build_output.split("\n")[::-1]:
            if line.startswith("ENGINE SAVED:"):
                self.engine_path = Path(line.split(":")[1])
                break

    def run_bench(self):
        streaming = "--streaming" if self.streaming else ""
        benchmark_cmd = \
            f"{self.mpirun_cmd} " \
            f"trtllm-bench --model {self.model_name} --model_path {self.model_path} " \
            f"throughput " \
            f"--tp {self.tp_size} "
        if self.engine_path:
            benchmark_cmd += f"--engine_dir {self.engine_path} "
        benchmark_cmd += f" --dataset {self.dataset_path} {streaming}"

        if self.use_pytorch_backend:
            benchmark_cmd += " --backend pytorch"

        if self.extra_llm_api_options:
            benchmark_cmd += f" --extra_llm_api_options {self.extra_llm_api_options}"
        check_call(benchmark_cmd, shell=True, env=self.llm_venv._new_env)
