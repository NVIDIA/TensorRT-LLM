# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json

import click
import pytest


def _write_config(tmp_path, architecture):
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "qwen3", "architectures": [architecture]})
    )
    return str(tmp_path)


def test_qwen3_causal_lm_is_remapped_to_reranker(tmp_path):
    from tensorrt_llm.commands.serve import _resolve_rerank_architecture_override

    model = _write_config(tmp_path, "Qwen3ForCausalLM")
    override = _resolve_rerank_architecture_override(model, trust_remote_code=False)

    assert override == {"architectures": ["Qwen3ForTextReranking"]}


def test_unknown_architecture_is_left_alone(tmp_path):
    from tensorrt_llm.commands.serve import _resolve_rerank_architecture_override

    model = _write_config(tmp_path, "BertForSequenceClassification")
    override = _resolve_rerank_architecture_override(model, trust_remote_code=False)

    assert override is None


def test_user_supplied_architecture_override_wins(monkeypatch):
    from tensorrt_llm.commands import serve
    from tensorrt_llm.llmapi.disagg_utils import ServerRole

    captured = {}
    run_value = object()

    monkeypatch.setattr(
        serve,
        "_resolve_rerank_architecture_override",
        lambda *args, **kwargs: {"architectures": ["Qwen3ForTextReranking"]},
    )

    def fake_llm(**kwargs):
        captured["llm_kwargs"] = kwargs
        return object()

    class FakeServer:
        def __init__(self, **kwargs):
            captured["server_kwargs"] = kwargs

        def __call__(self, host, port):
            captured["address"] = (host, port)
            return run_value

    monkeypatch.setattr(serve, "PyTorchLLM", fake_llm)
    monkeypatch.setattr(serve, "OpenAIServer", FakeServer)
    monkeypatch.setattr(serve.asyncio, "run", lambda value: captured.setdefault("run", value))

    llm_args = {
        "model": "reranker",
        "model_kwargs": {"architectures": ["CustomReranker"]},
    }
    serve.launch_rerank_server("localhost", 8000, llm_args, 0.005, 2048)

    assert captured["llm_kwargs"]["encode_only"] is True
    assert captured["llm_kwargs"]["model_kwargs"]["architectures"] == ["CustomReranker"]
    assert captured["server_kwargs"]["server_role"] is ServerRole.RERANK
    assert captured["address"] == ("localhost", 8000)
    assert captured["run"] is run_value


@pytest.mark.parametrize(
    "parallelism_key",
    ["tensor_parallel_size", "pipeline_parallel_size", "context_parallel_size"],
)
def test_encode_only_server_rejects_multi_gpu_config(monkeypatch, parallelism_key):
    from tensorrt_llm.commands import serve

    llm_args = {"model": "reranker", parallelism_key: 2}
    monkeypatch.setattr(serve, "collect_explicit_cli_keys", lambda **kwargs: set())
    monkeypatch.setattr(serve, "get_llm_args", lambda **kwargs: (llm_args, None))
    monkeypatch.setattr(
        serve._command_telemetry,
        "apply_raw_config_telemetry_opt_out",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        serve,
        "update_llm_args_with_extra_dict",
        lambda args, *unused_args, **unused_kwargs: args,
    )
    monkeypatch.setattr(
        serve,
        "_apply_effective_telemetry_config",
        lambda *args, **kwargs: None,
    )

    with pytest.raises(click.BadParameter, match="single-GPU only"):
        serve._prepare_encode_only_llm_args(
            model="reranker",
            max_batch_size=8,
            max_num_tokens=4096,
            trust_remote_code=False,
            revision=None,
            extra_llm_api_options=None,
            telemetry=False,
            server_name="rerank",
        )
