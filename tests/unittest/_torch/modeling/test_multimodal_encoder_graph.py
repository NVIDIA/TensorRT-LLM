# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Unit tests for ``MultimodalEncoderGraphRunner``.

The runner has two layers worth testing:

* Pure-Python (bucket selection, padding policy).
* CUDA graph capture/replay.
"""

from __future__ import annotations

import contextlib
import weakref
from dataclasses import dataclass
from typing import Callable, Dict, List, Mapping, Optional, Sequence
from unittest import mock

import pytest
import torch

import tensorrt_llm._torch.models.multimodal_encoder_graph as encoder_graph_module
from tensorrt_llm._torch.models.multimodal_encoder_graph import (
    EncoderGraphIOAdapter,
    EncoderGraphKey,
    EncoderGraphTensorSpec,
    MultimodalEncoderGraphRunner,
    PackedEncoderGraphIOAdapter,
    _CapturedGraph,
)
from tensorrt_llm._torch.utils import is_torch_compiling, torch_compiling
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


class _BatchedFramesIOAdapter(EncoderGraphIOAdapter):
    """Test layout with a batch/time axis, validity masks, and downsampled outputs."""

    def allocate_inputs(
        self, key: EncoderGraphKey, device: torch.device
    ) -> dict[str, torch.Tensor]:
        frames = key.total_tokens // key.num_contexts
        return {
            "features": torch.zeros(key.num_contexts, frames, 2, device=device),
            "mask": torch.zeros(key.num_contexts, frames, dtype=torch.bool, device=device),
        }

    def copy_inputs(
        self,
        static_inputs: Mapping[str, torch.Tensor],
        inputs: Mapping[str, torch.Tensor],
        seq_lengths: Sequence[int],
    ) -> None:
        for name, dst in static_inputs.items():
            dst.zero_()
            src = inputs[name]
            dst[:, : src.shape[1]].copy_(src)

    def collect_outputs(
        self, outputs: Mapping[str, torch.Tensor], seq_lengths: Sequence[int]
    ) -> dict[str, torch.Tensor]:
        output_frames = (max(seq_lengths) + 1) // 2
        return {name: tensor[:, :output_frames] for name, tensor in outputs.items()}

    def can_replay(self, bucket: EncoderGraphKey, seq_lengths: Sequence[int]) -> bool:
        return max(seq_lengths) <= bucket.total_tokens // bucket.num_contexts

    def graph_key(self, bucket: EncoderGraphKey, *, enable_padding: bool) -> EncoderGraphKey:
        return bucket

    def padded_seq_lengths(
        self, seq_lengths: Sequence[int], *, bucket: EncoderGraphKey, enable_padding: bool
    ) -> list[int]:
        return list(seq_lengths)

    def capture_seq_lengths(self, bucket: EncoderGraphKey, key: EncoderGraphKey) -> list[int]:
        return [key.total_tokens // key.num_contexts] * key.num_contexts


def _batched_frames_encoder(
    inputs: Mapping[str, torch.Tensor], metadata: _ToyMetadata
) -> dict[str, torch.Tensor]:
    mask = inputs["mask"][:, ::2]
    bias = metadata.seq_lens_cuda.sum().to(inputs["features"].dtype)
    features = (inputs["features"][:, ::2] + bias) * mask.unsqueeze(-1)
    return {"features": features, "mask": mask}


def _make_batched_frames_runner(
    device: torch.device, *, automatic: bool = False
) -> MultimodalEncoderGraphRunner[_ToyMetadata]:
    return MultimodalEncoderGraphRunner(
        encoder_fn=_batched_frames_encoder,
        metadata_provider=_ToyMetadataProvider(device),
        io_adapter=_BatchedFramesIOAdapter(),
        config=(
            MultimodalEncoderCudaGraphConfig(max_batch_size=3, max_num_tokens=12)
            if automatic
            else MultimodalEncoderCudaGraphConfig(buckets=[(8, 2), (12, 2)])
        ),
    )


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
    lengths = [real_tokens - real_contexts + 1] + [1] * (real_contexts - 1)
    assert runner._select_bucket(real_contexts, real_tokens, lengths) == expected_bucket


@pytest.mark.cpu_only
def test_padded_lengths_preserve_real_segments_at_partial_occupancy():
    bucket = EncoderGraphKey(total_tokens=512, num_contexts=4)
    runner = _make_logic_runner(buckets=[bucket])
    assert runner._select_bucket(2, 300, [120, 180]) == bucket
    assert runner._padded_seq_lengths([120, 180], bucket=bucket) == [120, 180, 1, 1, 214]
    key = runner._padded_key_for_bucket(bucket)
    assert runner._dummy_padded_seq_lengths(bucket, key) == [1, 1, 1, 1, 512]
    assert key == (516, 5)
    assert runner._select_bucket(0, 0, []) is None
    assert runner._select_bucket(1, 0, [0]) is None


def test_padded_seq_lengths_reserves_one_token_on_exact_fit():
    # With padding on, even an exact-fit workload gets a 1-token dummy context because the attention
    # backend rejects empty contexts.
    runner = _make_logic_runner()
    bucket = EncoderGraphKey(num_contexts=1, total_tokens=512)
    padded = runner._padded_seq_lengths([512], bucket=bucket)
    assert padded == [512, 1]


def test_padded_seq_lengths_no_dummy_when_padding_disabled():
    runner = _make_logic_runner(enable_padding=False)
    bucket = EncoderGraphKey(num_contexts=1, total_tokens=512)
    padded = runner._padded_seq_lengths([512], bucket=bucket)
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


def test_maybe_run_refreshes_metadata_under_inference_mode():
    """Capture runs under inference_mode, so a backend may own inference tensors by replay time.
    Refreshing them in place outside inference_mode is rejected."""
    bucket = EncoderGraphKey(num_contexts=1, total_tokens=256)
    runner = _make_logic_runner(buckets=[bucket])
    _install_fake_captured_graph(runner, bucket)
    seen = []
    runner._metadata_provider.refresh_in_place = lambda metadata, padded_seq_lengths: seen.append(
        torch.is_inference_mode_enabled()
    )

    with torch.no_grad():
        runner.maybe_run(seq_lengths=[200], inputs={"x": torch.ones(200, 4)})

    assert seen == [True]


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


@pytest.mark.cpu_only
def test_automatic_buckets_cover_every_batch_with_bounded_layout_shapes() -> None:
    runner = _make_batched_frames_runner(torch.device("cpu"), automatic=True)
    assert runner._buckets == [EncoderGraphKey(12, batch) for batch in (1, 2, 3)]
    for lengths in ([7], [5, 1], [4, 2, 1]):
        bucket = runner._select_bucket(len(lengths), sum(lengths), lengths)
        assert bucket is not None
        assert bucket.total_tokens <= 12
        assert bucket.total_tokens % bucket.num_contexts == 0
    assert runner._select_bucket(4, 4, [1] * 4) is None
    assert runner._select_bucket(2, 8, [7, 1]) is None


@pytest.mark.cpu_only
def test_automatic_buckets_include_intermediate_token_capacities() -> None:
    buckets = _BatchedFramesIOAdapter().capture_buckets(3, 4096)
    assert [bucket.total_tokens for bucket in buckets if bucket.num_contexts == 1] == [
        256,
        512,
        1024,
        2048,
        4096,
    ]
    assert EncoderGraphKey(4096, 3) in buckets
    assert all(bucket.total_tokens <= 4096 for bucket in buckets)
    inputs = _BatchedFramesIOAdapter().allocate_inputs(
        EncoderGraphKey(4096, 3), torch.device("cpu")
    )
    assert inputs["features"].shape == (3, 1365, 2)
    assert _BatchedFramesIOAdapter().capture_buckets(8, 2) == [
        EncoderGraphKey(2, 1),
        EncoderGraphKey(2, 2),
    ]


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "cap,count,contexts",
    [(129, 1, 1), (2048, 13, 8), (8192, 21, 32), (16384, 25, 32), (16385, 26, 32)],
)
def test_packed_automatic_budgets(cap: int, count: int, contexts: int) -> None:
    config = MultimodalEncoderCudaGraphConfig(max_num_tokens=cap)
    runner = MultimodalEncoderGraphRunner(
        encoder_fn=_NeverInvokedEncoder(),
        metadata_provider=_NoopMetadataProvider(),
        input_specs={},
        output_specs={},
        config=config,
    )
    budgets = [bucket.total_tokens for bucket in runner._buckets]
    assert len(budgets) == count
    assert budgets[-1] == cap
    if cap >= 512:
        assert budgets[:5] == [256, 320, 384, 448, 512]
    assert {bucket.num_contexts for bucket in runner._buckets} == {contexts}


@pytest.mark.cpu_only
def test_exact_packed_budgets_keep_the_per_count_ladder() -> None:
    runner = MultimodalEncoderGraphRunner(
        encoder_fn=_NeverInvokedEncoder(),
        metadata_provider=_NoopMetadataProvider(),
        io_adapter=PackedEncoderGraphIOAdapter({}),
        config=MultimodalEncoderCudaGraphConfig(
            max_batch_size=3, max_num_tokens=512, enable_padding=False
        ),
    )
    assert runner._buckets == [(256, 1), (512, 1), (512, 2), (512, 3)]
    assert runner._select_bucket(2, 256, [128, 128]) is None
    assert runner._select_bucket(2, 512, [0, 512]) is None
    assert runner._select_bucket(3, 512, [0, 2, 510]) == (512, 3)
    assert runner._select_bucket(3, 512, [-1, 3, 510]) is None


@pytest.mark.cpu_only
def test_automatic_buckets_use_encoder_adapter_shape_rules() -> None:
    class AlignedAdapter(_BatchedFramesIOAdapter):
        def capture_buckets(
            self, max_batch_size: int, max_num_tokens: int
        ) -> list[EncoderGraphKey]:
            return [EncoderGraphKey(8, 2)]

    runner = MultimodalEncoderGraphRunner(
        encoder_fn=_batched_frames_encoder,
        metadata_provider=_ToyMetadataProvider(torch.device("cpu")),
        io_adapter=AlignedAdapter(),
        config=MultimodalEncoderCudaGraphConfig(max_batch_size=2, max_num_tokens=12),
    )
    assert runner._buckets == [EncoderGraphKey(8, 2)]


@pytest.mark.cpu_only
def test_packed_inputs_share_the_largest_physical_allocation() -> None:
    runner = MultimodalEncoderGraphRunner(
        encoder_fn=_NeverInvokedEncoder(),
        metadata_provider=_NoopMetadataProvider(),
        input_specs={"x": EncoderGraphTensorSpec((4,), torch.float32, token_dim=1)},
        output_specs={"y": 1},
        config=MultimodalEncoderCudaGraphConfig(buckets=[(1050, 1), (1024, 128)]),
    )
    buffers = []

    def allocate(key, lengths, device):
        buffers.append(runner._io_adapter.allocate_inputs(key, device)["x"])

    with mock.patch.object(runner, "_capture_key", side_effect=allocate):
        runner.capture_all(torch.device("cpu"))
    assert [tensor.shape for tensor in buffers] == [(4, 1152), (4, 1051)]
    assert all(tensor.is_contiguous() for tensor in buffers)
    assert buffers[0].data_ptr() == buffers[1].data_ptr()
    with pytest.raises(ValueError, match="dtype|float"):
        runner._io_adapter.copy_inputs({"x": buffers[0]}, {"x": torch.ones(4, 1).double()}, [1])


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


def test_clear_resets_graphs_before_releasing_pool_owner():
    runner = _make_logic_runner()
    events = []
    first_graph = mock.Mock()
    second_graph = mock.Mock()
    first_graph.reset.side_effect = lambda: events.append("reset-first")
    second_graph.reset.side_effect = lambda: events.append("reset-second")
    runner._captured = {
        EncoderGraphKey(total_tokens=8, num_contexts=1): mock.Mock(graph=first_graph),
        EncoderGraphKey(total_tokens=16, num_contexts=1): mock.Mock(graph=second_graph),
    }
    runner._memory_pool = (12345, 67890)

    with mock.patch(
        "tensorrt_llm._torch.models.multimodal_encoder_graph.release_nccl_window_graph_owner",
        side_effect=lambda pool: events.append(("release", pool)),
    ):
        runner.clear()
        runner.clear()

    assert events == [
        "reset-first",
        "reset-second",
        ("release", (12345, 67890)),
    ]
    assert runner._captured == {}
    assert runner._memory_pool is None


def test_uncoordinated_clear_does_not_release_pool_owner():
    runner = _make_logic_runner()
    graph = mock.Mock()
    runner._captured = {EncoderGraphKey(total_tokens=8, num_contexts=1): mock.Mock(graph=graph)}
    runner._memory_pool = (12345, 67890)

    with mock.patch(
        "tensorrt_llm._torch.models.multimodal_encoder_graph.release_nccl_window_graph_owner"
    ) as release_owner:
        runner.clear(release_nccl_window_owners=False)

    graph.reset.assert_called_once_with()
    release_owner.assert_not_called()
    assert runner._memory_pool is None


@pytest.mark.cpu_only
@pytest.mark.parametrize("completed", [0, 1])
def test_capture_failure_abandons_partial_graphs(monkeypatch, completed):
    from tensorrt_llm._torch import nccl_window_graph

    pool = (90001, 91)
    graphs = []

    def capture(graph, graph_pool):
        graphs.append(graph)
        return nccl_window_graph.nccl_window_graph_owner(graph_pool)

    monkeypatch.setattr(torch.cuda, "CUDAGraph", mock.Mock)
    monkeypatch.setattr(torch.cuda, "graph_pool_handle", lambda: pool)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(encoder_graph_module, "nccl_window_graph_capture", capture)
    monkeypatch.setattr(torch.ops.trtllm, "set_nccl_window_graph_owner", lambda owner: None)
    release = mock.Mock()
    monkeypatch.setattr(torch.ops.trtllm, "release_nccl_window_graph_owner", release)
    runner = _make_logic_runner()
    snapshots = [{}] * completed + [RuntimeError("snapshot failed")]
    monkeypatch.setattr(runner, "_snapshot_metadata_tensor_ptrs", mock.Mock(side_effect=snapshots))

    with pytest.raises(RuntimeError, match="snapshot failed"):
        runner.capture_all(torch.device("cpu"))
    assert len(graphs) == completed + 1
    for graph in graphs:
        graph.reset.assert_called_once_with()
    assert not runner._captured and runner._memory_pool is None
    assert not runner._io_adapter._static_inputs
    assert pool not in nccl_window_graph._pool_owners
    release.assert_not_called()


def test_capture_scopes_warmup_and_captured_encoder_calls(monkeypatch):
    """Load-time capture has no outer NCCL-window tensor scope, so each encoder call opens one."""
    events = []
    active = []
    scoped_inputs = []

    @contextlib.contextmanager
    def discard(inputs):
        events.append("scope")
        scoped_inputs.append(inputs)
        active.append("scope")
        try:
            yield
        finally:
            active.pop()

    @contextlib.contextmanager
    def capture(graph, pool, **kwargs):
        active.append("capture")
        try:
            yield
        finally:
            active.pop()

    def encoder_fn(inputs, metadata):
        events.append(("encoder", tuple(active)))
        return {"y": inputs["x"]}

    monkeypatch.setattr(encoder_graph_module, "discard_nccl_window_tensor_outputs", discard)
    monkeypatch.setattr(encoder_graph_module, "nccl_window_graph_capture", capture)
    monkeypatch.setattr(encoder_graph_module.torch.cuda, "CUDAGraph", mock.Mock)
    monkeypatch.setattr(encoder_graph_module.torch.cuda, "synchronize", lambda: None)

    runner = _make_logic_runner(encoder_fn=encoder_fn)
    runner._memory_pool = (12345, 67890)
    key = EncoderGraphKey(num_contexts=1, total_tokens=8)
    runner._capture_key(key, [8], torch.device("cpu"))

    warmup_steps = runner._config.warmup_steps
    assert events == ["scope", ("encoder", ("scope",))] * warmup_steps + [
        "scope",
        ("encoder", ("capture", "scope")),
    ]
    static_inputs = runner._captured[key].inputs
    assert all(inputs is static_inputs for inputs in scoped_inputs)


@pytest.mark.cpu_only
def test_custom_io_adapter_updates_masks_and_collects_downsampled_outputs() -> None:
    device = torch.device("cpu")
    runner = _make_batched_frames_runner(device)
    bucket = EncoderGraphKey(total_tokens=8, num_contexts=2)
    key = runner._padded_key_for_bucket(bucket)
    assert key == bucket
    assert runner._dummy_padded_seq_lengths(bucket, key) == [4, 4]

    static_inputs = runner._io_adapter.allocate_inputs(key, device)
    metadata = runner._metadata_provider.build(key)
    outputs = _batched_frames_encoder(static_inputs, metadata)
    graph = mock.Mock()

    def replay() -> None:
        for name, tensor in _batched_frames_encoder(static_inputs, metadata).items():
            outputs[name].copy_(tensor)

    graph.replay.side_effect = replay
    runner._captured[key] = _CapturedGraph(
        graph=graph,
        inputs=static_inputs,
        metadata=metadata,
        outputs=outputs,
        metadata_tensor_ptrs=runner._snapshot_metadata_tensor_ptrs(
            metadata, runner._metadata_provider.graph_critical_attrs
        ),
    )
    input_ptrs = {name: tensor.data_ptr() for name, tensor in static_inputs.items()}
    for lengths, frames in [([4, 2], 4), ([1, 2], 2)]:
        features = torch.arange(2 * frames * 2, dtype=torch.float32).reshape(2, frames, 2)
        mask = torch.arange(frames).unsqueeze(0) < torch.tensor(lengths).unsqueeze(1)
        result = runner.maybe_run(seq_lengths=lengths, inputs={"features": features, "mask": mask})
        assert result is not None
        expected = torch.zeros(2, (max(lengths) + 1) // 2, 2)
        for batch_idx, length in enumerate(lengths):
            for frame_idx in range(0, length, 2):
                expected[batch_idx, frame_idx // 2] = features[batch_idx, frame_idx] + sum(lengths)
        torch.testing.assert_close(result["features"], expected)
        torch.testing.assert_close(result["mask"], mask[:, ::2])
        assert {name: tensor.data_ptr() for name, tensor in static_inputs.items()} == input_ptrs
        assert metadata.seq_lens_cuda.tolist() == lengths

    assert not static_inputs["mask"][:, 2:].any()
    assert not static_inputs["features"][:, 2:].any()
    assert graph.replay.call_count == 2


@pytest.mark.cpu_only
def test_custom_layout_skips_bucket_whose_time_axis_is_too_small() -> None:
    runner = _make_batched_frames_runner(torch.device("cpu"))
    assert runner._select_bucket(1, 4, [4]) is None
    assert runner._select_bucket(3, 4, [1, 2, 1]) is None
    lengths = [5, 1]
    assert runner._select_bucket(2, sum(lengths), lengths) == EncoderGraphKey(12, 2)
    runner._buckets = [EncoderGraphKey(8, 2)]
    assert runner.maybe_run(seq_lengths=lengths, inputs={}) is None


pytestmark_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="MultimodalEncoderGraphRunner requires CUDA",
)


@pytest.fixture
def cuda_device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device("cuda")


@pytestmark_cuda
@pytest.mark.parametrize("automatic", [False, True], ids=["explicit", "automatic"])
def test_cuda_custom_io_adapter_replays_masked_downsampled_outputs(cuda_device, automatic) -> None:
    runner = _make_batched_frames_runner(cuda_device, automatic=automatic)
    runner.capture_all(cuda_device)
    cases = [([4, 2], 4), ([3, 1], 3), ([1, 2], 2)]
    if automatic:
        cases += [([7], 7), ([4, 2, 1], 4), ([1, 1, 2], 2)]
    for lengths, frames in cases:
        batch = len(lengths)
        features = torch.arange(batch * frames * 2, dtype=torch.float32).reshape(batch, frames, 2)
        mask = torch.arange(frames).unsqueeze(0) < torch.tensor(lengths).unsqueeze(1)
        result = runner.maybe_run(
            seq_lengths=lengths,
            inputs={"features": features.to(cuda_device), "mask": mask.to(cuda_device)},
        )
        assert result is not None
        expected = torch.zeros(batch, (max(lengths) + 1) // 2, 2)
        for batch_idx, length in enumerate(lengths):
            for frame_idx in range(0, length, 2):
                expected[batch_idx, frame_idx // 2] = features[batch_idx, frame_idx] + sum(lengths)
        torch.testing.assert_close(result["features"].cpu(), expected, rtol=0, atol=0)
        torch.testing.assert_close(result["mask"].cpu(), mask[:, ::2])


@pytest.fixture
def make_cuda_runner(cuda_device):
    """Factory fixture for CUDA end-to-end tests."""

    def _factory(
        *,
        buckets: List[EncoderGraphKey],
        enable_padding: bool = True,
        encoder_fn: Optional[Callable] = None,
    ) -> MultimodalEncoderGraphRunner:
        config = MultimodalEncoderCudaGraphConfig(
            buckets=[(bucket.total_tokens, bucket.num_contexts) for bucket in buckets],
            enable_padding=enable_padding,
        )
        return MultimodalEncoderGraphRunner(
            encoder_fn=encoder_fn or _make_toy_encoder_fn(SCALE),
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
def test_flashinfer_replay_across_shared_buckets(cuda_device):
    from tensorrt_llm._torch.attention.backends.flashinfer import FlashInferAttention
    from tensorrt_llm._torch.attention.backends.interface import PredefinedAttentionMask
    from tensorrt_llm._torch.models.modeling_multimodal_encoder import prepare_encoder_attn_metadata

    width = 8 * 64
    layer = FlashInferAttention(layer_idx=0, num_heads=8, num_kv_heads=8, head_dim=64)
    eager = layer.Metadata(max_num_requests=4, max_num_tokens=131, kv_cache_manager=None)
    eager.max_seq_len = 32

    def attention(inputs, metadata):
        x = inputs["x"]
        return {"y": layer.forward(x, x, x, metadata, attention_mask=PredefinedAttentionMask.FULL)}

    adapter = PackedEncoderGraphIOAdapter(
        {"x": EncoderGraphTensorSpec((width,), torch.bfloat16)}, attn_metadata=eager
    )
    runner = MultimodalEncoderGraphRunner(
        encoder_fn=attention,
        metadata_provider=adapter,
        io_adapter=adapter,
        config=MultimodalEncoderCudaGraphConfig(buckets=[(64, 3), (128, 3)]),
    )
    runner.capture_all(cuda_device)
    workspaces = {
        capture.metadata.workspace_buffer.data_ptr() for capture in runner._captured.values()
    }
    assert len(workspaces) == 1
    assert len({c.inputs["x"].data_ptr() for c in runner._captured.values()}) == 1
    input_buffer = weakref.ref(adapter._static_inputs["x"])
    workspace = weakref.ref(adapter._buffers.buffers["workspace_buffer"][0].buffer)
    runner.clear()
    assert input_buffer() is None and workspace() is None
    runner.capture_all(cuda_device)

    generator = torch.Generator(device=cuda_device).manual_seed(2)
    for lengths, capacity in [([49, 40], 128), ([19], 64), ([30, 32, 33], 128)]:
        padded = [*lengths, *([1] * (3 - len(lengths))), capacity - sum(lengths) + len(lengths)]
        x = torch.randn(
            capacity + 3, width, device=cuda_device, dtype=torch.bfloat16, generator=generator
        )
        prepare_encoder_attn_metadata(padded, eager, max_seq_len=128)
        expected = attention({"x": x}, eager)["y"][: sum(lengths)]
        actual = runner.maybe_run(seq_lengths=lengths, inputs={"x": x[: sum(lengths)]})
        assert actual is not None
        torch.testing.assert_close(actual["y"], expected, rtol=0, atol=0)


def _probing_encoder_fn(probe: Callable[[], bool], sink: List[bool]):
    """Toy encoder that records `probe()` on every call the runner makes.

    Used to assert what ambient state capture establishes for the encoder region,
    across the warmup steps and the capture itself.
    """

    def encoder_fn(
        inputs: Dict[str, torch.Tensor], metadata: _ToyMetadata
    ) -> Dict[str, torch.Tensor]:
        sink.append(probe())
        x = inputs["x"]
        bias = metadata.seq_lens_cuda.sum().to(x.dtype)
        return {"y": x + bias}

    return encoder_fn


@pytestmark_cuda
def test_capture_uses_inference_mode_when_grad_enabled(cuda_device, make_cuda_runner):
    bucket = EncoderGraphKey(num_contexts=1, total_tokens=128)
    observed: List[bool] = []
    runner = make_cuda_runner(
        buckets=[bucket],
        encoder_fn=_probing_encoder_fn(torch.is_grad_enabled, observed),
    )

    with torch.enable_grad():
        assert torch.is_grad_enabled()
        runner.capture_all(cuda_device)

    assert observed and not any(observed)


@pytestmark_cuda
def test_capture_lowers_torch_compiling_and_restores_it(cuda_device, make_cuda_runner):
    """Capture must not inherit a raised compile flag from a previous engine.

    `is_torch_compiling_flag` is a plain module global, so in a reused worker it
    still holds whatever the last engine set. Layers gate their custom-op path on
    it (`Attention.forward_impl`), and that op resolves attention metadata from
    `extra_attrs`, which nothing binds during `load_weights`-driven capture. The
    caller's value must be restored on exit so an engine mid-compile is unaffected.
    """
    bucket = EncoderGraphKey(num_contexts=1, total_tokens=128)
    observed: List[bool] = []
    runner = make_cuda_runner(
        buckets=[bucket],
        encoder_fn=_probing_encoder_fn(is_torch_compiling, observed),
    )

    with torch_compiling(True):
        assert is_torch_compiling()
        runner.capture_all(cuda_device)
        assert is_torch_compiling(), "capture must restore the caller's value"

    assert observed and not any(observed)
    assert not is_torch_compiling()


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
