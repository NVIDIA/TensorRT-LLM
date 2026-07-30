# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Unit tests for ``MultimodalEncoderGraphRunner``.

The runner has two layers worth testing:

* Pure-Python (bucket selection, padding policy).
* CUDA graph capture/replay.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence
from unittest import mock

import pytest
import torch

from tensorrt_llm._torch.models.multimodal_encoder_graph import (
    EncoderGraphKey,
    EncoderGraphTensorSpec,
    MultimodalEncoderGraphRunner,
    _CapturedGraph,
)
from tensorrt_llm.llmapi.llm_args import MultimodalEncoderCudaGraphConfig

HIDDEN = 8
SCALE = 0.5


# ---------------------------------------------------------------------------
# Fixtures / helpers.
# ---------------------------------------------------------------------------


@dataclass
class _ToyMetadata:
    """Stand-in for AttentionMetadata used by the toy encoder.

    The runner only knows that metadata is an opaque object built by an `EncoderMetadataProvider`
    and refreshed in place by the same provider.
    The toy version stores per-context lengths in a fixed-size CUDA buffer so the captured graph
    can read them at replay time.
    """

    max_contexts: int
    seq_lens_cuda: torch.Tensor
    # `num_contexts` is intentionally not used by the captured kernel — the
    # graph instead reads `seq_lens_cuda` and the static input shape. This
    # mirrors how a real vision encoder is laid out: kernel boundaries are
    # baked into capture, not driven by runtime ints.
    num_contexts: int = 0


class _ToyMetadataProvider:
    """Concrete provider for the toy metadata used in CUDA tests."""

    graph_critical_attrs: Sequence[str] = ("seq_lens_cuda",)

    def __init__(self, device: torch.device) -> None:
        self._device = device

    def build(self, key: EncoderGraphKey) -> _ToyMetadata:
        return _ToyMetadata(
            max_contexts=key.num_contexts,
            seq_lens_cuda=torch.zeros(key.num_contexts, dtype=torch.int32, device=self._device),
        )

    def refresh_in_place(self, metadata: _ToyMetadata, padded_seq_lengths: Sequence[int]) -> None:
        n = len(padded_seq_lengths)
        if n > metadata.max_contexts:
            raise ValueError(
                f"padded_seq_lengths has {n} contexts but metadata buffer fits "
                f"only {metadata.max_contexts}."
            )
        host = torch.tensor(padded_seq_lengths, dtype=torch.int32)
        metadata.seq_lens_cuda[:n].copy_(host, non_blocking=True)
        metadata.num_contexts = n


class _NoopMetadataProvider:
    """No-op provider used in pure-Python tests that never reach capture."""

    graph_critical_attrs: Sequence[str] = ()

    def build(self, key: EncoderGraphKey):
        return None

    def refresh_in_place(self, metadata, padded_seq_lengths: Sequence[int]) -> None:
        return None


def _make_toy_encoder_fn(scale: float):
    """A deterministic, side-effect-free encoder for replay equivalence checks.

    The bias depends on `seq_lens_cuda` so the graph captures a real dependency on the metadata
    buffer rather than constant-folding it away.

    Importantly, the bias does not depend on the input shape, so eager and graph paths can be
    compared row-by-row after output slicing.
    """

    def encoder_fn(
        inputs: Dict[str, torch.Tensor], metadata: _ToyMetadata
    ) -> Dict[str, torch.Tensor]:
        x = inputs["x"]
        bias = metadata.seq_lens_cuda.sum().to(x.dtype)
        return {"y": x * scale + bias}

    return encoder_fn


def _eager_reference(
    x: torch.Tensor, padded_seq_lengths: Sequence[int], scale: float
) -> torch.Tensor:
    bias = float(sum(padded_seq_lengths))
    return x * scale + bias


class _NeverInvokedEncoder:
    """encoder_fn for tests where we expect maybe_run() to never invoke it."""

    def __init__(self):
        self.calls = 0

    def __call__(self, inputs, metadata):
        self.calls += 1
        return {}


class _FakeGraph:
    """CPU-only stand-in for torch.cuda.CUDAGraph."""

    def __init__(self):
        self.replay_calls = 0

    def replay(self) -> None:
        self.replay_calls += 1


def _make_logic_runner(
    *,
    buckets: Optional[List[EncoderGraphKey]] = None,
    enable_padding: bool = True,
    enable_replay_stats: bool = False,
    encoder_fn=None,
) -> MultimodalEncoderGraphRunner:
    """Build a pure-Python runner with default buckets."""
    encoder_fn = encoder_fn or _NeverInvokedEncoder()
    buckets = buckets or [
        EncoderGraphKey(num_contexts=1, total_tokens=256),
        EncoderGraphKey(num_contexts=1, total_tokens=512),
        EncoderGraphKey(num_contexts=1, total_tokens=1024),
    ]
    config = MultimodalEncoderCudaGraphConfig(
        buckets=[(bucket.total_tokens, bucket.num_contexts) for bucket in buckets],
        enable_padding=enable_padding,
        enable_replay_stats=enable_replay_stats,
    )
    return MultimodalEncoderGraphRunner(
        encoder_fn=encoder_fn,
        metadata_provider=_NoopMetadataProvider(),
        input_specs={"x": EncoderGraphTensorSpec(shape=(4,), dtype=torch.float32)},
        output_specs={"y": 0},
        config=config,
    )


@pytest.mark.parametrize(
    "enable_padding, real_contexts, real_tokens, expected_bucket",
    [
        pytest.param(
            True, 1, 512, EncoderGraphKey(num_contexts=1, total_tokens=512), id="exact-fit"
        ),
        pytest.param(
            True, 1, 300, EncoderGraphKey(num_contexts=1, total_tokens=512), id="smallest-fit"
        ),
        pytest.param(True, 1, 2048, None, id="too-big"),
        pytest.param(True, 2, 100, None, id="context-mismatch"),
        pytest.param(False, 1, 300, None, id="no-padding-partial-fit"),
        pytest.param(
            False,
            1,
            512,
            EncoderGraphKey(num_contexts=1, total_tokens=512),
            id="no-padding-exact-fit",
        ),
    ],
)
def test_select_bucket(enable_padding, real_contexts, real_tokens, expected_bucket):
    runner = _make_logic_runner(enable_padding=enable_padding)
    assert (
        runner._select_bucket(real_contexts=real_contexts, real_tokens=real_tokens)
        == expected_bucket
    )


def test_padded_seq_lengths_appends_dummy_context():
    runner = _make_logic_runner()
    bucket = EncoderGraphKey(num_contexts=1, total_tokens=512)
    padded = runner._padded_seq_lengths([300], bucket=bucket, real_tokens=300)
    # Slack (512 - 300) plus the runner's 1-token padding reservation.
    assert padded == [300, 213]


def test_padded_seq_lengths_reserves_one_token_on_exact_fit():
    # With padding on, even an exact-fit workload gets a 1-token dummy context because the attention
    # backend rejects empty contexts.
    runner = _make_logic_runner()
    bucket = EncoderGraphKey(num_contexts=1, total_tokens=512)
    padded = runner._padded_seq_lengths([512], bucket=bucket, real_tokens=512)
    assert padded == [512, 1]


def test_padded_seq_lengths_no_dummy_when_padding_disabled():
    runner = _make_logic_runner(enable_padding=False)
    bucket = EncoderGraphKey(num_contexts=1, total_tokens=512)
    padded = runner._padded_seq_lengths([512], bucket=bucket, real_tokens=512)
    assert padded == [512]


def test_maybe_run_returns_none_when_no_bucket():
    encoder = _NeverInvokedEncoder()
    runner = _make_logic_runner(encoder_fn=encoder)
    out = runner.maybe_run(seq_lengths=[9000], inputs={"x": torch.zeros(9000, 4)})
    assert out is None
    assert encoder.calls == 0


def _install_fake_captured_graph(
    runner: MultimodalEncoderGraphRunner, bucket: EncoderGraphKey
) -> _FakeGraph:
    key = runner._padded_key_for_bucket(bucket)
    graph = _FakeGraph()
    runner._captured[key] = _CapturedGraph(
        graph=graph,
        inputs={"x": torch.zeros(key.total_tokens, 4)},
        metadata=None,
        outputs={"y": torch.zeros(key.total_tokens, 4)},
        metadata_tensor_ptrs={},
    )
    return graph


def test_maybe_run_replays_captured_graph_and_slices_output():
    bucket = EncoderGraphKey(num_contexts=1, total_tokens=256)
    real_tokens = 200
    runner = _make_logic_runner(buckets=[bucket])
    graph = _install_fake_captured_graph(runner, bucket)

    out = runner.maybe_run(seq_lengths=[real_tokens], inputs={"x": torch.ones(real_tokens, 4)})

    assert out is not None
    assert tuple(out["y"].shape) == (real_tokens, 4)
    assert graph.replay_calls == 1


def test_replay_stats_disabled_does_not_log():
    runner = _make_logic_runner()
    assert runner._replay_stats_enabled is False

    with mock.patch("tensorrt_llm._torch.models.multimodal_encoder_graph.logger.info") as info:
        out = runner.maybe_run(seq_lengths=[9000], inputs={"x": torch.zeros(9000, 4)})

    assert out is None
    info.assert_not_called()


def test_replay_stats_enabled_logs_per_request_decision():
    bucket = EncoderGraphKey(num_contexts=1, total_tokens=256)
    runner = _make_logic_runner(buckets=[bucket], enable_replay_stats=True)
    _install_fake_captured_graph(runner, bucket)

    with mock.patch("tensorrt_llm._torch.models.multimodal_encoder_graph.logger.info") as info:
        runner.maybe_run(seq_lengths=[200], inputs={"x": torch.ones(200, 4)})  # hit
        runner.maybe_run(seq_lengths=[9000], inputs={"x": torch.zeros(9000, 4)})  # no-bucket miss

    messages = [call.args[0] for call in info.call_args_list]
    assert any("bucket hit" in m for m in messages)
    assert any("no bucket" in m for m in messages)


def test_empty_buckets_rejected():
    config = MultimodalEncoderCudaGraphConfig.model_construct(
        buckets=[],
        enable_padding=True,
        warmup_steps=2,
        enable_replay_stats=False,
    )
    with pytest.raises(ValueError, match="buckets is empty"):
        MultimodalEncoderGraphRunner(
            encoder_fn=_NeverInvokedEncoder(),
            metadata_provider=_NoopMetadataProvider(),
            input_specs={"x": EncoderGraphTensorSpec(shape=(4,), dtype=torch.float32)},
            output_specs={"y": 0},
            config=config,
        )


def test_metadata_buffer_snapshot_records_declared_attrs():
    md = _ToyMetadata(
        max_contexts=2,
        seq_lens_cuda=torch.zeros(2, dtype=torch.int32),
    )
    snapshot = MultimodalEncoderGraphRunner._snapshot_metadata_tensor_ptrs(md, ("seq_lens_cuda",))
    assert snapshot == {"seq_lens_cuda": md.seq_lens_cuda.data_ptr()}


def test_metadata_buffer_snapshot_rejects_non_tensor_attr():
    md = _ToyMetadata(
        max_contexts=2,
        seq_lens_cuda=torch.zeros(2, dtype=torch.int32),
    )
    with pytest.raises(TypeError, match="not a"):
        MultimodalEncoderGraphRunner._snapshot_metadata_tensor_ptrs(md, ("num_contexts",))


def test_metadata_buffer_stability_accepts_in_place_update():
    md = _ToyMetadata(
        max_contexts=2,
        seq_lens_cuda=torch.zeros(2, dtype=torch.int32),
    )
    snapshot = MultimodalEncoderGraphRunner._snapshot_metadata_tensor_ptrs(md, ("seq_lens_cuda",))

    # In-place copy_ keeps data_ptr stable.
    md.seq_lens_cuda.copy_(torch.tensor([7, 11], dtype=torch.int32))
    MultimodalEncoderGraphRunner._assert_metadata_buffers_stable(md, snapshot)


def test_metadata_buffer_stability_detects_realloc():
    md = _ToyMetadata(
        max_contexts=2,
        seq_lens_cuda=torch.zeros(2, dtype=torch.int32),
    )
    snapshot = MultimodalEncoderGraphRunner._snapshot_metadata_tensor_ptrs(md, ("seq_lens_cuda",))

    # Rebinding the attribute to a fresh tensor is the bug we want to catch.
    md.seq_lens_cuda = torch.zeros(2, dtype=torch.int32)
    with pytest.raises(RuntimeError, match="reallocated"):
        MultimodalEncoderGraphRunner._assert_metadata_buffers_stable(md, snapshot)


def test_metadata_buffer_stability_detects_type_change():
    md = _ToyMetadata(
        max_contexts=2,
        seq_lens_cuda=torch.zeros(2, dtype=torch.int32),
    )
    snapshot = MultimodalEncoderGraphRunner._snapshot_metadata_tensor_ptrs(md, ("seq_lens_cuda",))

    md.seq_lens_cuda = [0, 0]  # not a tensor anymore
    with pytest.raises(RuntimeError, match="was a tensor at capture time"):
        MultimodalEncoderGraphRunner._assert_metadata_buffers_stable(md, snapshot)


@pytest.mark.parametrize(
    "token_dim, expected_shape",
    [
        pytest.param(0, (128, 16), id="front"),
        pytest.param(1, (3, 128, 16), id="middle"),
    ],
)
def test_encoder_graph_tensor_spec_materialize(token_dim, expected_shape):
    base_shape = (16,) if token_dim == 0 else (3, 16)
    spec = EncoderGraphTensorSpec(shape=base_shape, dtype=torch.float32, token_dim=token_dim)
    t = spec.materialize(total_tokens=128, device=torch.device("cpu"))
    assert tuple(t.shape) == expected_shape


pytestmark_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="MultimodalEncoderGraphRunner requires CUDA",
)


@pytest.fixture
def cuda_device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device("cuda")


@pytest.fixture
def make_cuda_runner(cuda_device):
    """Factory fixture for CUDA end-to-end tests."""

    def _factory(
        *,
        buckets: List[EncoderGraphKey],
        enable_padding: bool = True,
    ) -> MultimodalEncoderGraphRunner:
        config = MultimodalEncoderCudaGraphConfig(
            buckets=[(bucket.total_tokens, bucket.num_contexts) for bucket in buckets],
            enable_padding=enable_padding,
        )
        return MultimodalEncoderGraphRunner(
            encoder_fn=_make_toy_encoder_fn(SCALE),
            metadata_provider=_ToyMetadataProvider(cuda_device),
            input_specs={
                "x": EncoderGraphTensorSpec(shape=(HIDDEN,), dtype=torch.float32, token_dim=0),
            },
            output_specs={"y": 0},
            config=config,
        )

    return _factory


@dataclass
class _PlanRunMetadata:
    """Metadata that models FlashInfer's plan/run split."""

    plan_cuda: torch.Tensor
    max_work_items: int = 0


class _PlanRunMetadataProvider:
    """Updates a graph-read plan in place and a Python launch extent."""

    graph_critical_attrs: Sequence[str] = ("plan_cuda",)

    def __init__(self, device: torch.device) -> None:
        self._device = device

    def build(self, key: EncoderGraphKey) -> _PlanRunMetadata:
        return _PlanRunMetadata(
            plan_cuda=torch.zeros(key.total_tokens, dtype=torch.float32, device=self._device),
        )

    def refresh_in_place(
        self,
        metadata: _PlanRunMetadata,
        padded_seq_lengths: Sequence[int],
    ) -> None:
        metadata.max_work_items = max(padded_seq_lengths)
        metadata.plan_cuda.zero_()
        metadata.plan_cuda[: metadata.max_work_items].fill_(1)


@pytestmark_cuda
class TestNoPaddingSkewedPlanRunReplay:
    """Regression tests for capture layouts that freeze a plan/run work extent."""

    @staticmethod
    def _encoder_fn(
        inputs: Dict[str, torch.Tensor],
        metadata: _PlanRunMetadata,
    ) -> Dict[str, torch.Tensor]:
        x = inputs["x"]
        y = torch.zeros_like(x)
        work_items = metadata.max_work_items
        y[:work_items].copy_(x[:work_items] * metadata.plan_cuda[:work_items].unsqueeze(-1))
        return {"y": y}

    def test_skewed_replay_matches_eager_plan_run(self, cuda_device):
        bucket = EncoderGraphKey(num_contexts=2, total_tokens=100)
        seq_lengths = [99, 1]
        real_tokens = sum(seq_lengths)
        config = MultimodalEncoderCudaGraphConfig(
            buckets=[(bucket.total_tokens, bucket.num_contexts)],
            enable_padding=False,
        )
        runner = MultimodalEncoderGraphRunner(
            encoder_fn=self._encoder_fn,
            metadata_provider=_PlanRunMetadataProvider(cuda_device),
            input_specs={
                "x": EncoderGraphTensorSpec(shape=(HIDDEN,), dtype=torch.float32, token_dim=0),
            },
            output_specs={"y": 0},
            config=config,
        )
        runner.capture_all(cuda_device)

        torch.manual_seed(4)
        x = torch.randn(real_tokens, HIDDEN, device=cuda_device, dtype=torch.float32)
        out = runner.maybe_run(seq_lengths=seq_lengths, inputs={"x": x})
        assert out is not None

        expected = torch.zeros_like(x)
        expected[: max(seq_lengths)].copy_(x[: max(seq_lengths)])
        torch.testing.assert_close(out["y"], expected, rtol=0, atol=0)


@pytestmark_cuda
def test_startup_capture_then_replay_with_padding(cuda_device, make_cuda_runner):
    bucket = EncoderGraphKey(num_contexts=1, total_tokens=512)
    real_tokens = 300
    padded_seq_lengths = [real_tokens, bucket.total_tokens - real_tokens + 1]
    runner = make_cuda_runner(buckets=[bucket])
    runner.capture_all(cuda_device)

    torch.manual_seed(0)
    x = torch.randn(real_tokens, HIDDEN, device=cuda_device, dtype=torch.float32)
    out = runner.maybe_run(seq_lengths=[real_tokens], inputs={"x": x})
    assert out is not None

    # Output token extent is sliced back to the real token count.
    assert tuple(out["y"].shape) == (real_tokens, HIDDEN)

    # Eager reference uses the same dummy-context layout.
    expected = _eager_reference(x, padded_seq_lengths=padded_seq_lengths, scale=SCALE)
    torch.testing.assert_close(out["y"], expected, rtol=0, atol=0)


@pytestmark_cuda
def test_startup_capture_misses_partial_fit_without_padding(cuda_device, make_cuda_runner):
    bucket = EncoderGraphKey(num_contexts=1, total_tokens=512)
    real_tokens = 300
    runner = make_cuda_runner(buckets=[bucket], enable_padding=False)
    runner.capture_all(cuda_device)

    # Without padding, only exact-fit requests replay.
    # Partial-fit misses rather than silently capturing a new variant on the fly.
    x = torch.randn(real_tokens, HIDDEN, device=cuda_device, dtype=torch.float32)
    assert runner.maybe_run(seq_lengths=[real_tokens], inputs={"x": x}) is None


@pytestmark_cuda
@pytest.mark.parametrize("real_tokens", [10, 64, 127, 300])
def test_replay_matches_eager_across_sizes(cuda_device, make_cuda_runner, real_tokens):
    buckets = [
        EncoderGraphKey(num_contexts=1, total_tokens=128),
        EncoderGraphKey(num_contexts=1, total_tokens=512),
    ]
    runner = make_cuda_runner(buckets=buckets)
    runner.capture_all(cuda_device)

    torch.manual_seed(2)
    x = torch.randn(real_tokens, HIDDEN, device=cuda_device, dtype=torch.float32)
    out = runner.maybe_run(seq_lengths=[real_tokens], inputs={"x": x})
    assert out is not None
    assert tuple(out["y"].shape) == (real_tokens, HIDDEN)

    # Eager reference uses the same dummy-context layout: slack plus the
    # runner's 1-token padding reservation.
    bucket = next(b for b in buckets if b.total_tokens >= real_tokens)
    padded = [real_tokens, (bucket.total_tokens - real_tokens) + 1]
    expected = _eager_reference(x, padded_seq_lengths=padded, scale=SCALE)
    torch.testing.assert_close(out["y"], expected, rtol=0, atol=0)


@pytestmark_cuda
def test_capture_uses_inference_mode_when_grad_enabled(cuda_device):
    bucket = EncoderGraphKey(num_contexts=1, total_tokens=128)
    grad_enabled_states = []

    def encoder_fn(
        inputs: Dict[str, torch.Tensor], metadata: _ToyMetadata
    ) -> Dict[str, torch.Tensor]:
        grad_enabled_states.append(torch.is_grad_enabled())
        x = inputs["x"]
        bias = metadata.seq_lens_cuda.sum().to(x.dtype)
        return {"y": x + bias}

    config = MultimodalEncoderCudaGraphConfig(
        buckets=[(bucket.total_tokens, bucket.num_contexts)],
        enable_padding=True,
        warmup_steps=2,
    )
    runner = MultimodalEncoderGraphRunner(
        encoder_fn=encoder_fn,
        metadata_provider=_ToyMetadataProvider(cuda_device),
        input_specs={
            "x": EncoderGraphTensorSpec(shape=(HIDDEN,), dtype=torch.float32, token_dim=0),
        },
        output_specs={"y": 0},
        config=config,
    )

    with torch.enable_grad():
        assert torch.is_grad_enabled()
        runner.capture_all(cuda_device)

    assert grad_enabled_states == [False, False, False]


class _ReallocatingProvider:
    graph_critical_attrs: Sequence[str] = ("seq_lens_cuda",)

    def __init__(self, device: torch.device) -> None:
        self._device = device

    def build(self, key: EncoderGraphKey) -> _ToyMetadata:
        return _ToyMetadata(
            max_contexts=key.num_contexts,
            seq_lens_cuda=torch.zeros(key.num_contexts, dtype=torch.int32, device=self._device),
        )

    def refresh_in_place(self, metadata: _ToyMetadata, padded_seq_lengths: Sequence[int]) -> None:
        # BUG: rebinds the tensor instead of copying into it.
        metadata.seq_lens_cuda = torch.tensor(
            list(padded_seq_lengths) + [0] * (metadata.max_contexts - len(padded_seq_lengths)),
            dtype=torch.int32,
            device=self._device,
        )
        metadata.num_contexts = len(padded_seq_lengths)


@pytestmark_cuda
def test_runner_rejects_provider_that_reallocates_metadata_tensor(cuda_device):
    """End-to-end: a refresh_in_place that rebinds a tensor is caught at replay."""

    bucket = EncoderGraphKey(num_contexts=1, total_tokens=128)
    config = MultimodalEncoderCudaGraphConfig(
        buckets=[(bucket.total_tokens, bucket.num_contexts)],
        enable_padding=True,
    )
    runner = MultimodalEncoderGraphRunner(
        encoder_fn=_make_toy_encoder_fn(SCALE),
        metadata_provider=_ReallocatingProvider(cuda_device),
        input_specs={
            "x": EncoderGraphTensorSpec(shape=(HIDDEN,), dtype=torch.float32, token_dim=0)
        },
        output_specs={"y": 0},
        config=config,
    )
    runner.capture_all(cuda_device)

    real_tokens = 64
    x = torch.randn(real_tokens, HIDDEN, device=cuda_device, dtype=torch.float32)
    with pytest.raises(RuntimeError, match="reallocated"):
        runner.maybe_run(seq_lengths=[real_tokens], inputs={"x": x})


@pytestmark_cuda
def test_dummy_padding_isolates_real_tokens(cuda_device, make_cuda_runner):
    """Padding context lives in its own row range, never blended into real tokens.

    After slicing, the real-token output must equal what eager would produce for the same padded
    sequence lengths - and it must not equal what eager would produce if we (incorrectly) extended
    the last real context to absorb the padding.
    """
    bucket = EncoderGraphKey(num_contexts=1, total_tokens=512)
    real_tokens = 300
    padded_seq_lengths = [real_tokens, bucket.total_tokens - real_tokens + 1]
    runner = make_cuda_runner(buckets=[bucket])
    runner.capture_all(cuda_device)

    torch.manual_seed(3)
    x = torch.randn(real_tokens, HIDDEN, device=cuda_device, dtype=torch.float32)
    out = runner.maybe_run(seq_lengths=[real_tokens], inputs={"x": x})
    assert out is not None

    # The toy encoder's bias is `sum(seq_lens)`, so the real-token output reflects whether the
    # padding lives in its own dummy context or was folded into the real context.
    # Correct: the dummy-context layout [300, 213] (sum 513).
    correct = _eager_reference(x, padded_seq_lengths=padded_seq_lengths, scale=SCALE)
    torch.testing.assert_close(out["y"], correct, rtol=0, atol=0)

    # Bad alternative: the last real context absorbs the padding, i.e. seq_lens == [512] (sum 512).
    # The real-token output must NOT match that.
    bad = _eager_reference(x, padded_seq_lengths=[bucket.total_tokens], scale=SCALE)
    assert not torch.allclose(out["y"], bad)

    # And the captured graph's metadata buffer holds the separate padding context.
    captured_key = EncoderGraphKey(
        num_contexts=len(padded_seq_lengths),
        total_tokens=sum(padded_seq_lengths),
    )
    captured_meta = runner._captured[captured_key].metadata
    assert captured_meta.seq_lens_cuda.tolist() == padded_seq_lengths
