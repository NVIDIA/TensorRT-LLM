import gc
import glob
import json
import os
from unittest import mock

import lm_eval_ad  # noqa: F401
import pytest
from _dist_test_utils import param_with_device_count
from _model_test_utils import _hf_model_dir_or_hub_id
from _torch_test_utils import fp4_compatible, fp8_compatible
from lm_eval.__main__ import cli_evaluate
from lm_eval.api.task import TaskConfig
from utils.llm_data import llm_models_root

_task_config_post_init = TaskConfig.__post_init__


def _patched_post_init(self: TaskConfig, *args, **kwargs):
    # Call the original __post_init__
    ret = _task_config_post_init(self, *args, **kwargs)

    # Redirect dataset_path to the local directory if it exists.
    # This is to enable the test works in offline mode when we are running on CI.
    dataset_path_lookup = {
        "hails/mmlu_no_train": f"{llm_models_root()}/datasets/hails/mmlu_no_train",
        "gsm8k": f"{llm_models_root()}/datasets/openai/gsm8k",
    }

    # If the dataset key isn't in dataset_path_lookup, or the local path doesn't exist,
    # the test fallbacks to downloading dataset from huggingface hub; it will fail if
    # offline mode is enabled by HF_DATASETS_OFFLINE=1.
    if self.dataset_path in dataset_path_lookup:
        local_path = dataset_path_lookup[self.dataset_path]
        if os.path.exists(local_path):
            self.dataset_path = local_path

    return ret


def _cli_evaluate_with_mocks(args):
    # Mock post-init and argparser's reference to sys.argv
    with (
        mock.patch.object(TaskConfig, "__post_init__", new=_patched_post_init),
        mock.patch("argparse._sys.argv", [""] + args),
    ):
        # set up args via argparser
        cli_evaluate()


# LM eval limit setting has a bug: https://github.com/EleutherAI/lm-evaluation-harness/issues/2324
# So we run each task individually.
@pytest.mark.parametrize(
    "world_size,model_args,tasks,score_keys,scores",
    [
        param_with_device_count(
            2,
            {
                "model": _hf_model_dir_or_hub_id(
                    f"{llm_models_root()}/llama-3.1-model/Llama-3.1-8B-Instruct",
                    "meta-llama/Meta-Llama-3.1-8B-Instruct",
                ),
                "attn_backend": "FlashInfer",
                "max_batch_size": 32,
            },
            ["gsm8k", "mmlu"],
            ["exact_match,strict-match", "acc,none"],
            [0.75, 0.675],
            marks_extra=[
                pytest.mark.skip(
                    reason="https://nvbugspro.nvidia.com/bug/5123940; failed and timeout"
                )
            ],
        ),
        param_with_device_count(
            2,
            {
                "model": _hf_model_dir_or_hub_id(
                    f"{llm_models_root()}/llama-3.1-model/Llama-3.1-8B-Instruct-FP8",
                    "nvidia/Llama-3.1-8B-Instruct-FP8",
                )
            },
            ["gsm8k", "mmlu"],
            ["exact_match,strict-match", "acc,none"],
            [0.70, 0.67],
            marks_extra=[
                pytest.mark.skipif(not fp8_compatible(), reason="Requires fp8 support"),
            ],
        ),
        param_with_device_count(
            2,
            {
                "model": "nvidia/Llama-3.1-8B-Instruct-FP4",
            },
            ["gsm8k", "mmlu"],
            ["exact_match,strict-match", "acc,none"],
            [0.70, 0.64],
            marks_extra=[
                pytest.mark.skipif(not fp4_compatible(), reason="Requires fp4 support"),
                pytest.mark.skip(
                    reason="https://nvbugspro.nvidia.com/bug/5095416; to add ckpt on llm-models"
                ),
            ],
        ),
        param_with_device_count(
            4,
            {
                "model": _hf_model_dir_or_hub_id(
                    f"{llm_models_root()}/Mixtral-8x7B-Instruct-v0.1",
                    "mistralai/Mixtral-8x7B-Instruct-v0.1",
                ),
                "compile_backend": "torch-simple",
            },
            ["gsm8k", "mmlu"],
            ["exact_match,strict-match", "acc,none"],
            [0.583, 0.67],
            marks_extra=[
                pytest.mark.skip(reason="https://nvbugspro.nvidia.com/bug/5095416; timeout")
            ],
        ),
    ],
)
def test_lm_eval(world_size: int, model_args, tasks, score_keys, scores, tmp_path):
    # setup base args for our testing
    args = ["--model", "autodeploy", "--batch_size", "4", "--limit", "0.1"]

    # set up model args with world size
    model_args["world_size"] = world_size
    args += ["--model_args", ",".join([f"{k}={v}" for k, v in model_args.items()])]
    gen_kwargs = {}
    # Greedy to avoid variance
    gen_kwargs["temperature"] = 1e-4
    gen_kwargs["top_k"] = 1
    args += ["--gen_kwargs", ",".join([f"{k}={v}" for k, v in gen_kwargs.items()])]

    # set up output path and tasks
    args += ["--output_path", str(tmp_path)]
    args += ["--tasks", ",".join(tasks) if len(tasks) > 1 else tasks[0]]

    # run the cli with CI patches
    _cli_evaluate_with_mocks(args)

    json_files = glob.glob(os.path.join(tmp_path, "**", "*.json"), recursive=True)
    assert json_files
    with open(json_files[0], "r") as f:
        results = json.load(f)["results"]

        for task, score_key, score in zip(tasks, score_keys, scores):
            assert results[task][score_key] >= score, (
                f"{task}:{score_key} blow expected threshold. Details: {results[task]}"
            )

    # Terminate and free the executor resource.
    gc.collect()
