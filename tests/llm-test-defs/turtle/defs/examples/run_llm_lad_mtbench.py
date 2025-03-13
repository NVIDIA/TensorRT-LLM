import argparse
import os
import uuid

from defs.trt_test_alternative import check_call
from evaltool.constants import *

LLM_GATE_WAY_CLIENT_ID = os.environ.get("LLM_GATE_WAY_CLIENT_ID")
LLM_GATE_WAY_TOKEN = os.environ.get("LLM_GATE_WAY_TOKEN")
GITLAB_API_USER = os.environ.get("GITLAB_API_USER")
GITLAB_API_TOKEN = os.environ.get("GITLAB_API_TOKEN")
EVALTOOL_REPO_URL = os.environ.get("EVALTOOL_REPO_URL")


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine_dir',
                        type=str,
                        required=True,
                        help='tokenizer path')
    parser.add_argument('--hf_model_dir', '--model_dir', type=str, default=None)
    parser.add_argument(
        '--tokenizer_dir',
        default=None,
        help='tokenizer path; defaults to hf_model_dir if left unspecified')
    parser.add_argument('--workspace', default=None, help='workspace directory')
    parser.add_argument("--lookahead_config", type=str, default=None)
    parser.add_argument("--device_count", type=int, default=1)
    args = parser.parse_args()
    return args


def prepare_evaltool(workspace):
    assert GITLAB_API_USER is not None and GITLAB_API_TOKEN is not None, "run_llm_lad_mtbench needs a gitlab token."
    assert EVALTOOL_REPO_URL is not None, "EVALTOOL_REPO_URL is not set."
    clone_dir = os.path.join(workspace, "eval-tool")
    repo_url = f"https://{GITLAB_API_USER}:{GITLAB_API_TOKEN}@{EVALTOOL_REPO_URL}"
    branch_name = "dev/0.9"

    from evaltool.constants import EVALTOOL_SETUP_SCRIPT
    evaltool_setup_cmd = [
        EVALTOOL_SETUP_SCRIPT, "-b", branch_name, "-d", clone_dir, "-r",
        repo_url
    ]
    check_call(" ".join(evaltool_setup_cmd), shell=True)
    return clone_dir


def run_lad_mtbench(engine_dir,
                    hf_model_dir,
                    workspace,
                    device_count=1,
                    tokenizer_dir=None,
                    lookahead_config=None):
    hf_model_dir = os.path.normpath(hf_model_dir)
    tokenizer_dir = hf_model_dir if tokenizer_dir is None else os.path.normpath(
        tokenizer_dir)

    # prepare evaltool
    evaltool_root = prepare_evaltool(workspace)

    # start inference server
    start_inference_server = [
        EVALTOOL_INFERENCE_SERVER_STARTUP_SCRIPT, "-e", engine_dir, "-t",
        tokenizer_dir, "-d", evaltool_root, "-m", "1024", "-c",
        str(device_count)
    ]
    if lookahead_config is not None:
        start_inference_server += ["-l", lookahead_config]
    check_call(" ".join(start_inference_server), shell=True)

    try:
        project_id = str(uuid.uuid4())
        config_file = EVALTOOL_MTBENCH_CONFIG
        result_file = EVALTOOL_MTBENCH_RESULT_FILE
        model_name = os.path.basename(hf_model_dir)

        # Update config dynamically
        import yaml
        with open(config_file, 'r') as f:
            mt_bench_config = yaml.safe_load(f)
            mt_bench_config['model']['llm_name'] = model_name
            mt_bench_config['model']['tokenizer_path'] = tokenizer_dir
            mt_bench_config['evaluations'][0]['judge_model'][
                'client_id'] = LLM_GATE_WAY_CLIENT_ID
            mt_bench_config['evaluations'][0]['judge_model'][
                'client_secret'] = LLM_GATE_WAY_TOKEN
            mt_bench_config['evaluations'][0]['inference_params'][
                'temperature'] = 1.0
            mt_bench_config['evaluations'][0]['inference_params']['top_p'] = 0.0

        config_file = os.path.join(workspace,
                                   f"{model_name}_mtbench_config.yaml")
        with open(config_file, 'w') as f:
            yaml.dump(mt_bench_config, f)

        # Update resource config
        run_cmd = [
            f"cd {evaltool_root}",
            "&&",
            "source .venv/bin/activate",
            "&&",
            "python3",
            "evaltool/interfaces/cli/main.py",
            "config",
            "resource",
            "--resource_config_file examples/resource_configs/resource_local.yaml",
        ]
        check_call(" ".join(run_cmd), shell=True, executable="/bin/bash")

        # launch evaluation
        run_cmd = [
            f"cd {evaltool_root}",
            "&&",
            "source .venv/bin/activate",
            "&&",
            "python3",
            f"evaltool/interfaces/cli/main.py",
            "project",
            "launch",
            f"--eval_project_config_file '{config_file}'",
            "--infra_name local",
            f"--output_dir '{workspace}'",
            f"--project_id {project_id}",
        ]
        check_call(" ".join(run_cmd), shell=True, executable="/bin/bash")

    finally:
        # stop the server
        check_call(f"{EVALTOOL_INFERENCE_SERVER_STOP_SCRIPT}", shell=True)

    # process result
    result_path = f"{workspace}/{project_id}/{result_file}/{model_name}.csv"
    check_call(f"cat {result_path}", shell=True)
    return result_path


if __name__ == '__main__':
    args = parse_arguments()
    run_lad_mtbench(engine_dir=args.engine_dir,
                    hf_model_dir=args.hf_model_dir,
                    workspace=args.workspace,
                    tokenizer_dir=args.tokenizer_dir,
                    lookahead_config=args.lookahead_config,
                    device_count=args.device_count)
