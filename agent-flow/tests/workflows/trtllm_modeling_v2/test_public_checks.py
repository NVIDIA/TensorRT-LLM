# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public entry boundaries that change a check's verdict or workload."""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pytest

from agent_flow.workflows.trtllm_modeling_v2.checks.harness import reuse_probe, workload_io


def test_normalization_preserves_source_skip_template_order_and_image_requests(tmp_path):
    data = tmp_path / "data.jsonl"
    data.write_text(
        "\n".join(
            json.dumps(r)
            for r in [
                {"q": "skip", "tag": "x"},
                {},
                {"q": "first", "tag": "a"},
                {"q": "second", "tag": "b"},
            ]
        )
    )
    image = {
        "id": "image",
        "prompt": "look",
        "images": ["/local/image.png"],
        "render": "raw",
        "max_tokens": 3,
    }
    spec = {
        "requests": [image],
        "dataset": {
            "source": str(data),
            "text_field": "q",
            "skip": 1,
            "num": 2,
            "template": "{tag}: {text}",
        },
    }
    requests = workload_io.normalize_requests(spec)
    assert requests[0] == image
    assert [r["prompt"] for r in requests[1:]] == ["a: first", "b: second"]
    spec["dataset"]["skip"] = 99
    with pytest.raises(workload_io.DatasetError, match="no usable prompts"):
        workload_io.normalize_requests(spec)


def test_repeat_probe_completes_before_resubmission_and_pins_attention_rank(monkeypatch):
    events = []
    monkeypatch.setitem(
        sys.modules,
        "tensorrt_llm.scheduling_params",
        SimpleNamespace(SchedulingParams=lambda **kw: kw),
    )

    class LLM:
        tokenizer = SimpleNamespace(encode=lambda *args, **kwargs: [1, 2, 3])

        def get_kv_cache_capacity(self):
            return {"tokensPerBlock": 4, "maxNumTokens": 128}

        def generate_async(self, prompt, params, **kwargs):
            events.append(("submit", kwargs))
            return SimpleNamespace(
                result=lambda: events.append(("complete", {})) or SimpleNamespace(cached_tokens=4)
            )

    result = reuse_probe.probe(LLM(), SimpleNamespace(), {"enable_attention_dp": True})
    assert result["status"] == "OBSERVED" and result["reused"] == 4
    assert [e[0] for e in events] == ["submit", "complete", "submit", "complete"]
    assert events[0][1] == events[2][1]
    assert events[0][1]["scheduling_params"] == {
        "attention_dp_rank": 0,
        "attention_dp_relax": False,
    }
    assert reuse_probe.probe(SimpleNamespace(), SimpleNamespace(), {})["status"] == "UNKNOWN"
