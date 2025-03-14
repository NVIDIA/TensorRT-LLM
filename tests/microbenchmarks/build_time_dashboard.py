import argparse
import os
import subprocess
import traceback
from dataclasses import KW_ONLY, dataclass
from datetime import datetime, timedelta
from enum import Enum, auto
from pathlib import Path
from subprocess import check_output
from typing import Optional

import pandas
from build_time_benchmark import models_name_to_path

script_path = os.path.dirname(os.path.abspath(__file__))


class Tricks(Enum):
    default = auto()
    gemm_plugin = auto()
    managed_weights = auto()
    fp8 = auto()
    fp8_managed_weights = auto()


@dataclass
class Metrics:
    model: str
    _: KW_ONLY

    trick: Optional[Tricks] = None

    build_command: Optional[str] = None
    logfile: Optional[str] = None
    # trt_llm_engine_build_time + trt_llm_convert_time
    trt_llm_e2e_time: Optional[float] = None

    # AutoModelForCausalLM.from_hugging_face, or  Model.quantize() -> write -> load -> Model.from_checkpoint
    trt_llm_convert_time: Optional[float] = 0
    trt_llm_engine_build_time: Optional[float] = None  # tensorrt_llm.build time

    # TRT INetwork -> ICudaEngine -> IHostMemory
    build_serialized_network_time: Optional[float] = None

    # TRT INetwork -> ICudaEngine
    engine_generation_time: Optional[float] = None

    def common_fields(self):
        return [
            'build_command', 'logfile', 'trt_llm_e2e_time',
            'trt_llm_convert_time', 'trt_llm_engine_build_time',
            'build_serialized_network_time', 'engine_generation_time'
        ]


def time_to_seconds(time_str):
    dt = datetime.strptime(time_str, "%H:%M:%S")
    delta = timedelta(hours=dt.hour, minutes=dt.minute, seconds=dt.second)
    return delta.total_seconds()


def sanity():
    out_bytes = check_output(
        ['python3', f'{script_path}/build_time_benchmark.py',
         '--help']).decode('utf-8')
    print(out_bytes)


def run_model(model, logfile, timeout=3600, extral_options=[]) -> Metrics:
    '''Spawn a process to run buld_time_benchmark.py and then grep the results
    '''
    metrics = Metrics(model)

    out_log = ""
    build_command = [
        'python3', f'{script_path}/build_time_benchmark.py', '--model', model
    ] + extral_options
    metrics.build_command = " ".join(build_command)

    # easier to resume the job
    if not os.path.exists(logfile):
        try:
            out_log = check_output(build_command,
                                   timeout=timeout).decode('utf-8')
        except subprocess.TimeoutExpired as e:
            out_log = f"Benchmark {model} timeout\n"
            out_log += str(e)
        except subprocess.CalledProcessError as e:
            out_log = e.output.decode('utf-8')
            code = e.returncode
            out_log += f"Benchmark {model} errored with return code {code}"

        with open(logfile, 'w') as log:
            log.write(out_log)

    try:
        # yapf: disable
        logfile = str(logfile)
        metrics.engine_generation_time = float(check_output(["grep", "generation", logfile]).decode('utf-8').split(" ")[7])
        build_serialized_network_time = check_output(["grep", "Total time of building Unnamed Network", logfile]).decode('utf-8').strip().split(" ")[-1]
        metrics.build_serialized_network_time = time_to_seconds(build_serialized_network_time)
        metrics.trt_llm_engine_build_time = float(check_output(['grep', "===BuildTime==== build_engine", logfile]).decode('utf-8').split(" ")[-2])

        try:  # this one is optional, some configs dont load the weights
            metrics.trt_llm_convert_time = float(check_output(['grep', "===BuildTime==== load_and_convert", logfile]).decode('utf-8').split(" ")[-2])
        except Exception:
            pass

        metrics.trt_llm_e2e_time = metrics.trt_llm_engine_build_time + metrics.trt_llm_convert_time
        # yapf: enable
    except Exception as e:
        print(
            f"Error in found metrics for {model} in {logfile}, pls check the log"
        )
        traceback.print_exc()
        print(e)

    return metrics


def main(args):

    models = [
        'llama2-7b', 'llama3-70b.TP4', 'mixtral-8x22b.TP8', 'mixtral-8x7b.TP4'
    ]
    for model in models:
        assert model in models_name_to_path, f"{model} not supported by the benchmark script"

    tricks_and_options = [
        (Tricks.gemm_plugin, ['--gemm', 'plugin', '--strong_type']),
        (Tricks.default, ['--load', '--strong_type']),
        (Tricks.managed_weights,
         ['--load', '--managed_weights', '--strong_type']),
        (Tricks.fp8, ['--quant', 'fp8', '--strong_type']),
        (Tricks.fp8_managed_weights,
         ['--quant', 'fp8', '--strong_type', '--managed_weights']),
    ]

    if args.minimal:  # just sanity check the flow
        models = [models[0]]
        tricks_and_options = [tricks_and_options[0]]

    logdir = Path(args.logdir)
    if not logdir.exists():
        logdir.mkdir(parents=True, exist_ok=True)

    dashboard_records = []
    dashboard_combined = []

    for model in models:
        model_records = []
        for (trick, options) in tricks_and_options:
            print(f"Running model {model} {trick.name} {options} start")

            logfile = logdir / f"{model}.{trick.name}.log"
            metrics = Metrics(model)
            try:
                metrics = run_model(model, logfile, extral_options=options)
            except Exception:
                print(
                    f"Exception during run model {model} with {trick} and options {options}, pls check the logfile {logfile}"
                )
            metrics.trick = trick.name
            metrics.logfile = os.path.abspath(logfile)
            model_records.append(metrics)
            dashboard_records.append(metrics)

            # refresh the results every iteration, in cases the job is timeout or killed
            df = pandas.DataFrame(dashboard_records)
            df.to_csv(f"{logdir}/results.csv", index=False)

            print(f"Running model {model} {trick.name} {options} end")

        # combine the different tricks of the same method to same row
        combined_fields = {}
        combined_fields['model'] = model
        for r in model_records:
            for field in metrics.common_fields():
                combined_fields[f"{field}-{r.trick}"] = getattr(r, field)
        dashboard_combined.append(combined_fields)
        df = pandas.DataFrame(dashboard_combined)
        df.to_csv(f"{logdir}/results_combined.csv", index=False)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate the dashboard of the build time benchmark")
    parser.add_argument("logdir", type=str, help="The dir to save the logs")
    parser.add_argument(
        "--minimal",
        '-m',
        default=False,
        action='store_true',
        help="Just run one model and one config and then return!")
    return parser.parse_args()


if __name__:
    main(parse_args())
