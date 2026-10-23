# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fixed-token scoring across chunk and prefix-cache boundaries."""

import json
from pathlib import Path

import pytest
import torch

from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm.sampling_params import LogitsProcessor

from .test_prefill_decode_consistency_gpu import _configure_determinism, _make_checkpoint

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")


class _CaptureAndForceTokens(LogitsProcessor):
    def __init__(self, prompt: list[int], response: list[int], output: Path) -> None:
        self.prompt = prompt
        self.response = response
        self.output = str(output)

    def __call__(self, req_id, logits, token_ids, stream_ptr, client_id) -> None:
        assert len(token_ids) == 1
        position = len(token_ids[0]) - len(self.prompt)
        assert 0 <= position < len(self.response)
        assert token_ids[0] == self.prompt + self.response[:position]
        target = self.response[position]
        stream = None if stream_ptr is None else torch.cuda.ExternalStream(stream_ptr)
        with torch.cuda.stream(stream):
            assert logits.numel() == logits.shape[-1]
            raw = logits.detach().reshape(-1).to(device="cpu", dtype=torch.float32)
            score = float(torch.log_softmax(raw, dim=-1)[target])
            assert torch.isfinite(torch.tensor(score))
            with open(self.output, "a", encoding="utf-8") as handle:
                handle.write(json.dumps({"position": position, "score": score}) + "\n")
            logits.fill_(float("-inf"))
            logits[..., target] = 0


@pytest.mark.parametrize("cuda_graph", [False, True])
@pytest.mark.parametrize("deterministic", [False, True], ids=["default", "deterministic"])
def test_fixed_tokens_across_cache_boundaries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cuda_graph: bool,
    deterministic: bool,
) -> None:
    _configure_determinism(monkeypatch, deterministic)
    model = tmp_path / "model"
    _make_checkpoint(model)
    response = [41, 52, 63, 74, 85, 96, 107, 118]
    baseline: dict[int, torch.Tensor] = {}
    # FORCE_DETERMINISTIC disables prefix reuse in the executor creator.
    # Exercise exact chunk/graph arithmetic under that policy; the default
    # mode separately verifies actual prefix hits and history preservation.
    modes = ((False, False), (True, False))
    if not deterministic:
        modes += ((False, True), (True, True))
    for chunked, reuse in modes:
        mode = f"chunk{int(chunked)}-reuse{int(reuse)}"
        with LLM(
            model=str(model),
            backend="pytorch",
            attn_backend="TRTLLM" if deterministic else "FlashInfer",
            env_overrides={
                "FORCE_DETERMINISTIC": "1" if deterministic else "0",
                "FORCE_ATTENTION_KERNEL_DETERMINISTIC": "0",
            },
            max_batch_size=1,
            max_seq_len=512,
            max_num_tokens=128 if chunked else 512,
            enable_chunked_prefill=chunked,
            disable_overlap_scheduler=True,
            cuda_graph_config={"batch_sizes": [1]} if cuda_graph else None,
            kv_cache_config={
                "enable_block_reuse": reuse,
                "tokens_per_block": 64,
                "use_kv_cache_manager_v2": True,
                "max_tokens": 2048,
            },
        ) as llm:
            for length in (63, 64, 65, 127, 128, 129, 255, 256, 257):
                prompt = ([17, 63, 129, 12, 71] * (1 + length // 5))[:length]
                salt = f"{mode}-{length}"
                for phase in ("cold", "warm", "after_unrelated"):
                    if phase == "after_unrelated":
                        llm.generate(
                            [[42, 99, 123] * 43],
                            sampling_params=SamplingParams(
                                max_tokens=4,
                                temperature=0,
                                ignore_eos=True,
                                add_special_tokens=False,
                            ),
                            cache_salt=f"other-{salt}",
                            use_tqdm=False,
                        )
                    output = tmp_path / f"{mode}-{length}-{phase}.jsonl"
                    result = llm.generate(
                        [prompt],
                        sampling_params=SamplingParams(
                            max_tokens=len(response),
                            temperature=0,
                            ignore_eos=True,
                            add_special_tokens=False,
                            return_perf_metrics=True,
                            logits_processor=_CaptureAndForceTokens(prompt, response, output),
                        ),
                        cache_salt=salt,
                        use_tqdm=False,
                    )[0]
                    assert list(result.prompt_token_ids) == prompt
                    assert list(result.outputs[0].token_ids) == response
                    metrics = result.outputs[0].request_perf_metrics
                    assert metrics is not None
                    records = [json.loads(line) for line in output.read_text().splitlines()]
                    contexts = sum(row["position"] == 0 for row in records)
                    reused = metrics.kv_cache_metrics.num_reused_blocks
                    assert contexts >= 1
                    if not reuse or phase == "cold":
                        assert reused == 0
                    elif length > 64:
                        assert reused > 0
                    if phase == "cold":
                        expected_contexts = (length + 127) // 128 if chunked else 1
                        assert contexts == expected_contexts
                    # Intermediate chunks do not emit a token. Their callbacks
                    # precede the final prefill callback that scores response[0].
                    assert [row["position"] for row in records] == [0] * contexts + list(
                        range(1, len(response))
                    )
                    scores = torch.tensor([row["score"] for row in records[contexts - 1 :]])
                    if not chunked and not reuse and phase == "cold":
                        baseline[length] = scores
                    torch.testing.assert_close(
                        scores, baseline[length], rtol=0, atol=0 if deterministic else 0.01
                    )
                if not chunked and not reuse and length == 257:
                    # Keep the last prompt token and continuation fixed, but change
                    # the cached prefix. This checkpoint must distinguish its history.
                    other_prompt = ([201, 202, 203, 204, 205] * 52)[:length]
                    other_prompt[-1] = prompt[-1]
                    output = tmp_path / "different-prefix.jsonl"
                    llm.generate(
                        [other_prompt],
                        sampling_params=SamplingParams(
                            max_tokens=len(response),
                            temperature=0,
                            ignore_eos=True,
                            add_special_tokens=False,
                            logits_processor=_CaptureAndForceTokens(other_prompt, response, output),
                        ),
                        cache_salt="different-prefix",
                        use_tqdm=False,
                    )
                    rows = [json.loads(line) for line in output.read_text().splitlines()]
                    assert [row["position"] for row in rows] == list(range(len(response)))
                    other_scores = torch.tensor([row["score"] for row in rows])
                    assert (other_scores - baseline[length]).abs().max() > 0.01
