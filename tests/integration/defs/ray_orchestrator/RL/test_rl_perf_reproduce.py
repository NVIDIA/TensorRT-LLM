import json
import tempfile
from pathlib import Path

import pytest
from defs.common import venv_check_call
from defs.conftest import integration_path, llm_models_root
from transformers import AutoTokenizer


@pytest.mark.skip_less_device(4)
@pytest.mark.parametrize(
    "tp_size, num_instances", [(2, 2), (1, 4)], ids=["tp2_2instances", "tp1_4instances"]
)
def test_rl_perf_reproduce(llm_venv, tp_size, num_instances):
    script_path = (
        integration_path() / "defs" / "ray_orchestrator" / "RL" / "run_rl_perf_reproduce.py"
    )
    model_dir = f"{llm_models_root()}/Qwen2-7B-Instruct"

    if tp_size == 2:
        max_batch_size = 512
    else:
        max_batch_size = 256

    with tempfile.TemporaryDirectory() as tmpdir:
        prompt_text = "The president of the United States is"

        tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        token_ids = tokenizer.encode(prompt_text, add_special_tokens=False)

        # Replicate to create batch of 1024 prompts
        batch_size = 1024
        prompts = [token_ids for _ in range(batch_size)]

        data_path = Path(tmpdir) / "prompts.json"
        with open(data_path, "w") as f:
            json.dump(prompts, f)

        venv_check_call(
            llm_venv,
            [
                str(script_path),
                "--model_dir",
                model_dir,
                "--data_path",
                str(data_path),
                "--num_instances",
                str(num_instances),
                "--tp_size",
                str(tp_size),
                "--logprobs",
                "1",
                "--max_batch_size",
                str(max_batch_size),
                "--enable_block_reuse",
                "--enable_cuda_graph_padding",
            ],
        )
