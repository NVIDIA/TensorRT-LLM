import os

EVALTOOL_REPO_URL = os.environ.get("EVALTOOL_REPO_URL")

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)

EVALTOOL_SETUP_SCRIPT = os.path.join(
    parent_directory, f"setup.sh -r https://{EVALTOOL_REPO_URL}")
EVALTOOL_INFERENCE_SERVER_STARTUP_SCRIPT = os.path.join(
    parent_directory, "run_trtllm_inference_server.sh")
EVALTOOL_INFERENCE_SERVER_STOP_SCRIPT = os.path.join(
    parent_directory, "stop_inference_server.sh")

EVALTOOL_HUMAN_EVAL_CONFIG = os.path.join(parent_directory, "eval_configs",
                                          "human_eval.yml")
EVALTOOL_MMLU_CONFIG = os.path.join(parent_directory, "eval_configs",
                                    "mmlu_str.yml")
EVALTOOL_WIKILINGUA_CONFIG = os.path.join(parent_directory, "eval_configs",
                                          "wikilingua.yml")
EVALTOOL_MTBENCH_CONFIG = os.path.join(parent_directory, "eval_configs",
                                       "mt_bench.yml")

EVALTOOL_HUMAN_EVAL_RESULT_FILE = "automatic/bigcode_latest/results/bigcode-aggregate_scores.json"
EVALTOOL_MMLU_RESULT_FILE = "automatic/lm_eval_harness/results/lm-harness-mmlu_str.json"
EVALTOOL_WIKILINGUA_RESULT_FILE = "automatic/lm_eval_harness/results/lm-harness.json"
EVALTOOL_MTBENCH_RESULT_FILE = "llm_as_a_judge/mtbench/results/"
