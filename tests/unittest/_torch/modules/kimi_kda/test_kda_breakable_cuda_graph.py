# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The KDA mixer under the breakable prefill CUDA graph: padding rows, the split core, replay."""

import weakref
from types import SimpleNamespace

import pytest
import torch
from torch import nn

pytest.importorskip("fla")

from tensorrt_llm._torch.attention_backend import AttentionMetadata  # noqa: E402
from tensorrt_llm._torch.model_config import ModelConfig  # noqa: E402
from tensorrt_llm._torch.modules.kimi_kda import (  # noqa: E402
    KimiKDALinearAttention,
    kimi_kda_mixer,
)
from tensorrt_llm._torch.modules.kimi_kda._kda_flashinfer_prefill import (  # noqa: E402
    run_prefill_core,
)
from tensorrt_llm._torch.pyexecutor.breakable_cuda_graph_runner import (  # noqa: E402
    BreakableCUDAGraphRunner,
)
from tensorrt_llm._torch.utils import model_extra_attrs  # noqa: E402
from tensorrt_llm.models.modeling_utils import QuantConfig  # noqa: E402
from tensorrt_llm.quantization import QuantAlgo  # noqa: E402
from tests.unittest._torch.modules.kimi_kda.kda_module_test_utils import (  # noqa: E402
    clone_pool,
    compare,
    make_attention,
    make_batch,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() not in ((10, 0), (10, 3), (10, 7)),
    reason="FlashInfer KDA requires SM100/SM103/SM107",
)

HEADS, HIDDEN = 6, 768
BUCKET = 128


@pytest.fixture(scope="module")
def attention() -> KimiKDALinearAttention:
    return make_attention(HEADS, hidden_size=HIDDEN)


def _live_batch(batch, lengths):
    """Fresh cache pools and live attention metadata of ``batch``; one forward consumes them."""
    cache = SimpleNamespace(conv=batch.conv.clone(), temporal=clone_pool(batch.state))
    metadata = AttentionMetadata(max_num_requests=len(lengths), max_num_tokens=sum(lengths))
    metadata.seq_lens = torch.tensor(lengths, dtype=torch.int32)
    metadata.num_contexts = len(lengths)
    metadata.mamba_metadata = batch.metadata
    metadata.kv_cache_manager = SimpleNamespace(mamba_layer_cache=lambda _: cache)
    return metadata, cache


class _Body(nn.Module):
    """The model body the runner patches: one KDA layer on the batch of the moment."""

    def __init__(self, attention: KimiKDALinearAttention):
        super().__init__()
        self.attention = attention
        self.metadata = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.attention(hidden_states, self.metadata)


@torch.inference_mode()
def test_padded_eager_forward_matches_unpadded(attention) -> None:
    """Padding rows after the tokens leave as zeros and never reach the real rows or states."""
    lengths, pad = [17, 65], 13
    batch = make_batch(HEADS, lengths, hidden_size=HIDDEN, initial="mixed", padded=True)
    metadata, cache = _live_batch(batch, lengths)
    expected = attention(batch.hidden, metadata)

    padding = torch.randn(pad, HIDDEN, device="cuda", dtype=torch.bfloat16)
    metadata, padded_cache = _live_batch(batch, lengths)
    actual = attention(torch.cat([batch.hidden, padding]), metadata)

    assert actual.shape == (sum(lengths) + pad, HIDDEN)
    torch.testing.assert_close(actual[:-pad], expected, rtol=0, atol=0)
    assert not actual[-pad:].any()
    torch.testing.assert_close(padded_cache.conv, cache.conv, rtol=0, atol=0)
    torch.testing.assert_close(padded_cache.temporal, cache.temporal, rtol=0, atol=0)


@pytest.mark.parametrize("mode", ["split_core", "whole_mixer", "unfused_convolution"])
@pytest.mark.parametrize("lengths", [[64, 64], [17, 65]], ids=["full_bucket", "padded_bucket"])
@torch.inference_mode()
def test_graph_replay_matches_eager(attention, lengths, mode, monkeypatch) -> None:
    """A bucket captured on a dummy batch and replayed on a live one gives the eager result.

    The split core needs the fused token-major convolution: without it the whole mixer breaks.
    """
    monkeypatch.setattr(kimi_kda_mixer, "_KDA_BCG_WHOLE_MIXER", mode == "whole_mixer")
    monkeypatch.setattr(attention, "_kda_fi_fuse_conv", mode != "unfused_convolution")
    total = sum(lengths)
    batch = make_batch(HEADS, lengths, hidden_size=HIDDEN, initial="mixed", padded=True)
    metadata, expected_cache = _live_batch(batch, lengths)
    expected = attention(batch.hidden, metadata)

    # The engine captures a bucket on a dummy batch of exactly the bucket's tokens ...
    dummy_lengths = [BUCKET // 2, BUCKET // 2]
    dummy = make_batch(HEADS, dummy_lengths, hidden_size=HIDDEN, input_seed=5, padded=True)
    dummy_metadata, _ = _live_batch(dummy, dummy_lengths)
    body = _Body(attention)
    body.metadata = dummy_metadata
    runner = BreakableCUDAGraphRunner(body)
    static_input = dummy.hidden.clone()
    extra_attrs = {
        "attention_metadata": weakref.ref(dummy_metadata),
        "kda_layers": {attention.layer_idx_str: weakref.ref(attention)},
    }

    def engine_forward() -> torch.Tensor:
        if runner.is_capturing:
            return runner.capture_model_body(lambda: body(static_input))
        return body(static_input)

    try:
        with model_extra_attrs(extra_attrs):
            runner.capture(BUCKET, engine_forward)
            # ... and replays it on a live batch whose padding rows hold arbitrary values.
            live_metadata, live_cache = _live_batch(batch, lengths)
            extra_attrs["attention_metadata"] = weakref.ref(live_metadata)
            static_input.copy_(torch.randn_like(static_input))
            static_input[:total] = batch.hidden
            actual = runner.execute(BUCKET, lambda: body(static_input)).clone()
            torch.cuda.synchronize()
    finally:
        runner.clear()

    assert actual.shape == (BUCKET, HIDDEN)
    metrics = compare(actual[:total], expected)
    assert metrics["passed"], metrics
    assert not actual[total:].any()
    slots = batch.indices.long()
    for name in ("conv", "temporal"):
        metrics = compare(getattr(live_cache, name)[slots], getattr(expected_cache, name)[slots])
        assert metrics["passed"], (name, metrics)


def test_layers_sharing_a_model_config_register_under_distinct_names() -> None:
    """The registry the eager breaks resolve their layer from keeps every layer of the config."""
    config = ModelConfig(quant_config=QuantConfig(quant_algo=QuantAlgo.FP8_BLOCK_SCALES))
    first = make_attention(HEADS, hidden_size=HIDDEN, model_config=config)
    second = make_attention(HEADS, hidden_size=HIDDEN, model_config=config)

    assert first.layer_idx_str != second.layer_idx_str
    registry = config.extra_attrs["kda_layers"]
    assert registry[first.layer_idx_str]() is first
    assert registry[second.layer_idx_str]() is second


def test_prefill_core_rejects_an_inconsistent_token_count() -> None:
    """More tokens than rows, or padding rows without the token-major convolution, is an error."""
    g = torch.zeros(1, 8, 2, 4)
    metadata = SimpleNamespace(
        state_indices=torch.zeros(1, dtype=torch.int32),
        has_initial_states=torch.zeros(1, dtype=torch.bool),
    )
    unfused = SimpleNamespace(_kda_fi_fuse_conv=False)

    def run(num_tokens: int) -> None:
        run_prefill_core(
            unfused, None, g, None, None, metadata, 1, None, None, num_tokens=num_tokens
        )

    with pytest.raises(ValueError, match="exceed"):
        run(9)
    with pytest.raises(ValueError, match="fused"):
        run(7)


def test_split_core_rejects_a_batch_with_generation_requests(attention) -> None:
    """The engine never replays the split core for a mixed batch; reaching it is an error."""
    metadata = SimpleNamespace(
        num_contexts=2, seq_lens=torch.tensor([17, 65, 1], dtype=torch.int32)
    )
    with pytest.raises(RuntimeError, match="context-only"):
        attention._run_prefill_core_breakable(None, None, None, metadata, None)
