# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Sol-Attn correctness tests: backend dispatch, config guards, step-context
dense_layers/disabled_until_timestep guards.

Mirrors test_attention_cute_dsl_vsa.py's structure and scope for its sibling
sparse-attention algorithm. GPU kernel-vs-dense numerical equivalence (the
analogue of VSA's test_cute_kernel_matches_dense_at_full_topk) is not yet
covered here -- see the TODO on test_cute_kernel_matches_dense_placeholder
below for what it needs and why it's deferred, not just missing.
"""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.visual_gen.attention_backend import CuTeDSLAttention
from tensorrt_llm._torch.visual_gen.attention_backend.cute_dsl.sol_attn import (
    SolAttnAttention,
    _parse_dense_layers,
    sol_attn_graph_phase,
)
from tensorrt_llm._torch.visual_gen.attention_backend.utils import create_attention
from tensorrt_llm._torch.visual_gen.config import (
    DiffusionModelConfig,
    create_attention_metadata_state,
)
from tensorrt_llm._torch.visual_gen.modules.attention import Attention, QKVMode
from tensorrt_llm.visual_gen.args import AttentionConfig, SolAttnAttentionConfig


def test_cute_dsl_factory_dispatches_dense_and_sol_attn() -> None:
    dense_config = AttentionConfig(backend="CUTEDSL")
    dense_attention = create_attention(
        backend="CUTEDSL",
        layer_idx=0,
        num_heads=8,
        head_dim=128,
        attention_config=dense_config,
    )

    sparse_config = SolAttnAttentionConfig(tau=2.0, disabled_until_timestep=0.9545)
    sol_attn_config = AttentionConfig(backend="CUTEDSL", sparse_attention_config=sparse_config)
    sol_attn_attention = create_attention(
        backend="CUTEDSL",
        layer_idx=0,
        num_heads=8,
        head_dim=128,
        attention_config=sol_attn_config,
    )

    assert isinstance(dense_attention, CuTeDSLAttention)
    assert isinstance(sol_attn_attention, SolAttnAttention)
    assert sol_attn_attention.tau == 2.0
    assert sol_attn_attention.disabled_until_timestep == 0.9545


def _make_config(
    hidden_size: int,
    num_heads: int,
    head_dim: int,
    backend: str,
    sol_attn_tau: "float | None" = None,
) -> DiffusionModelConfig:
    """Minimal DiffusionModelConfig for one Attention module."""
    pretrained_config = SimpleNamespace(
        hidden_size=hidden_size,
        num_attention_heads=num_heads,
        attention_head_dim=head_dim,
        eps=1e-6,
    )
    sparse_attention_config = (
        SolAttnAttentionConfig(tau=sol_attn_tau) if sol_attn_tau is not None else None
    )
    config = DiffusionModelConfig(
        pretrained_config=pretrained_config,
        attention=AttentionConfig(backend=backend, sparse_attention_config=sparse_attention_config),
        skip_create_weights_in_init=False,
    )
    config.attention_metadata_state = (
        create_attention_metadata_state() if backend == "TRTLLM" else None
    )
    return config


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Sol-Attn needs CUDA")
def test_sol_attn_cross_attention_uses_dense_cutedsl():
    """Cross-attention must stay on CuTeDSL, not drop to VANILLA.

    Sol-Attn is self-attention only, so SEPARATE_QKV modules fall back -- but to
    the dense kernel of the *configured* backend, not to torch SDPA. Falling back
    to VANILLA made a `backend: CUTEDSL` run and a `CUTEDSL + sol_attn` run differ
    in cross-attention in every block, regardless of any sparse setting, which is
    a backend difference masquerading as a sparsity difference in any A/B.
    """
    from tensorrt_llm._torch.visual_gen.attention_backend.cute_dsl.fmha import CuTeDSLAttention
    from tensorrt_llm._torch.visual_gen.attention_backend.cute_dsl.sol_attn import SolAttnAttention

    device = torch.device("cuda")
    dtype = torch.bfloat16
    cfg = _make_config(
        hidden_size=64, num_heads=4, head_dim=16, backend="CUTEDSL", sol_attn_tau=1.0
    )
    cross_attn = (
        Attention(64, 4, qkv_mode=QKVMode.SEPARATE_QKV, config=cfg)
        .to(device=device, dtype=dtype)
        .eval()
    )
    assert cross_attn.attn_backend == "CUTEDSL", (
        f"expected CUTEDSL cross-attention, got {cross_attn.attn_backend!r}"
    )
    assert isinstance(cross_attn.attn, CuTeDSLAttention), (
        f"expected the dense CuTeDSL kernel, got {type(cross_attn.attn).__name__}"
    )
    assert not isinstance(cross_attn.attn, SolAttnAttention), (
        "cross-attention must not re-select the sparse backend"
    )


def test_sol_attn_with_context_parallelism_raises():
    """Sol-Attn + Attention2D/Ring must error at construction (needs the full sequence per rank)."""
    pretrained_config = SimpleNamespace(
        hidden_size=64,
        num_attention_heads=4,
        attention_head_dim=16,
        eps=1e-6,
    )
    cfg = DiffusionModelConfig(
        pretrained_config=pretrained_config,
        attention=AttentionConfig(
            backend="CUTEDSL",
            sparse_attention_config=SolAttnAttentionConfig(tau=1.0),
        ),
        skip_create_weights_in_init=False,
    )
    cfg.visual_gen_mapping = SimpleNamespace(
        ring_size=1,
        ring_group=None,
        ulysses_size=1,
        ulysses_group=None,
        attn2d_row_size=2,
        attn2d_col_size=2,
        attn2d_row_group=None,
        attn2d_col_group=None,
        cp_size=4,
    )
    with pytest.raises(ValueError, match="incompatible with context parallelism"):
        Attention(64, 4, qkv_mode=QKVMode.FUSE_QKV, config=cfg)


def test_sol_attn_rejects_gqa_mqa():
    """Sol-Attn is MHA-only; num_kv_heads != num_heads must fail fast at construction."""
    with pytest.raises(AssertionError, match="MHA-only"):
        SolAttnAttention(layer_idx=0, num_heads=8, head_dim=128, num_kv_heads=2)


@pytest.mark.parametrize(
    "spec,expected",
    [
        (None, frozenset()),
        ("", frozenset()),
        ("0", frozenset({0})),
        ("0,2,4", frozenset({0, 2, 4})),
        ("0-3", frozenset({0, 1, 2, 3})),
        ("0-1,5,7-8", frozenset({0, 1, 5, 7, 8})),
        (" 0 , 2 ", frozenset({0, 2})),
    ],
    ids=["none", "empty", "single", "list", "range", "mixed", "whitespace"],
)
def test_parse_dense_layers(spec, expected):
    assert _parse_dense_layers(spec) == expected


@pytest.mark.parametrize(
    "timestep,expected",
    [
        (0.99, 0),  # early/noisy -> dense prefix
        (0.9545, 0),  # exactly at the cutoff -> still dense
        (0.95, 1),  # past the cutoff -> sparse
        (0.0, 1),  # final step -> sparse
        (None, None),  # no timestep -> no phase to distinguish
    ],
    ids=["early", "at-cutoff", "past-cutoff", "final", "missing"],
)
def test_graph_phase_matches_skip_softmax_sense(timestep, expected):
    """Phase 0 is the dense prefix, 1 the sparse phase, None when undecidable.

    Same contract as SkipSoftmaxScheduler.get_graph_phase_for_timestep.
    """
    assert sol_attn_graph_phase(timestep, disabled_until_timestep=0.9545) == expected


def test_graph_phase_none_when_prefix_unset():
    assert sol_attn_graph_phase(0.5, disabled_until_timestep=None) is None


def test_graph_phase_accepts_tensor_timestep():
    """Pipelines pass a tensor; a 0-d or 1-element tensor must work."""
    assert sol_attn_graph_phase(torch.tensor(0.99), disabled_until_timestep=0.95) == 0
    assert sol_attn_graph_phase(torch.tensor([0.10]), disabled_until_timestep=0.95) == 1


def test_dense_prefix_skips_kernel(monkeypatch):
    """Inside the dense prefix the sparse kernel must not be invoked at all.

    CPU tensors, so `_dense` takes its SDPA branch here; that the dense path
    routes to the CuTe kernel on CUDA is covered by
    `test_dense_paths_use_cutedsl_backend`.
    """
    import tensorrt_llm._torch.visual_gen.attention_backend.cute_dsl.sol_attn as sol_attn_mod

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("kernel must not run inside the dense prefix")

    monkeypatch.setattr(sol_attn_mod, "_sol_attn_run", _fail_if_called)

    attn = SolAttnAttention(layer_idx=0, num_heads=2, head_dim=16)
    attn.disabled_until_timestep = 0.9
    q = k = v = torch.randn(1, 4, 2, 16)
    out = attn.forward(q, k, v, timestep=torch.tensor(0.95))
    assert out.shape == q.shape
    assert torch.isfinite(out).all()


def test_missing_timestep_fails_open_to_sparse(monkeypatch):
    """Without a timestep the prefix cannot be applied; run sparse, do not raise.

    Matches the CuTeDSL skip-softmax path's fail-open choice.
    """
    import tensorrt_llm._torch.visual_gen.attention_backend.cute_dsl.sol_attn as sol_attn_mod

    called = {"n": 0}

    def _record(*args, **kwargs):
        called["n"] += 1
        return args[0]

    monkeypatch.setattr(sol_attn_mod, "_sol_attn_run", _record)

    attn = SolAttnAttention(layer_idx=0, num_heads=2, head_dim=16)
    attn.disabled_until_timestep = 0.9
    q = k = v = torch.randn(1, 4, 2, 16)
    attn.forward(q, k, v)  # no timestep kwarg
    assert called["n"] == 1, "expected the sparse kernel, not a silent dense fallback"


def test_dense_layers_guard_skips_kernel(monkeypatch):
    """A layer_idx in dense_layers must use the dense SDPA path and never invoke the kernel."""
    import tensorrt_llm._torch.visual_gen.attention_backend.cute_dsl.sol_attn as sol_attn_mod

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("kernel must not be invoked for a dense_layers-forced layer")

    monkeypatch.setattr(sol_attn_mod, "_sol_attn_run", _fail_if_called)

    attn = SolAttnAttention(layer_idx=3, num_heads=2, head_dim=16)
    attn.dense_layers = frozenset({3})
    q = k = v = torch.randn(1, 4, 2, 16)
    out = attn.forward(q, k, v)
    assert out.shape == q.shape
    assert torch.isfinite(out).all()


def _make_solattn_model(disabled_until_timestep=None, dense_layers=None):
    """Minimal BaseDiffusionModel carrying a Sol-Attn sparse config."""
    from tensorrt_llm._torch.visual_gen.models.modeling import BaseDiffusionModel

    pretrained_config = SimpleNamespace(
        hidden_size=64, num_attention_heads=4, attention_head_dim=16, eps=1e-6
    )
    config = DiffusionModelConfig(
        pretrained_config=pretrained_config,
        attention=AttentionConfig(
            backend="CUTEDSL",
            sparse_attention_config=SolAttnAttentionConfig(
                tau=2.0,
                disabled_until_timestep=disabled_until_timestep,
                dense_layers=dense_layers,
            ),
        ),
        skip_create_weights_in_init=False,
    )
    return BaseDiffusionModel(config)


def _graph_runner():
    from tensorrt_llm._torch.visual_gen.cuda_graph_runner import (
        CUDAGraphRunner,
        CUDAGraphRunnerConfig,
    )

    return CUDAGraphRunner(CUDAGraphRunnerConfig(use_cuda_graph=True))


def test_cuda_graph_key_separates_dense_prefix_from_sparse_phase():
    """The prefix swaps kernels without changing any tensor shape, so a graph
    captured in the dense prefix must not be replayed for the sparse phase."""
    model = _make_solattn_model(disabled_until_timestep=0.9)
    runner = _graph_runner()
    model.register_cuda_graph_extra_key_fns(runner)

    base = {"hidden_states": torch.empty(1, 8, 64)}
    key_dense = runner.get_graph_key(**base, timestep=torch.empty(1).fill_(0.95))
    key_sparse = runner.get_graph_key(**base, timestep=torch.empty(1).fill_(0.10))

    assert key_dense != key_sparse, (
        "dense-prefix and sparse phases share a CUDA graph key despite running "
        "different kernels; a graph captured in one phase would be replayed in "
        "the other"
    )


def test_cuda_graph_key_unregistered_without_prefix():
    """dense_layers alone is fixed per layer, so it needs no graph key."""
    model = _make_solattn_model(disabled_until_timestep=None, dense_layers="0,2")
    runner = _graph_runner()
    model.register_cuda_graph_extra_key_fns(runner)

    base = {"hidden_states": torch.empty(1, 8, 64)}
    key_a = runner.get_graph_key(**base, timestep=torch.empty(1).fill_(0.95))
    key_b = runner.get_graph_key(**base, timestep=torch.empty(1).fill_(0.10))
    assert key_a == key_b, "no phase key should be registered without a dense prefix"


# --- kernel-wrapper eligibility / strictness (sol_attn_backend.py) -----------
# These run on CPU: every path here is pure Python guard logic, and the CPU
# tensor is itself one of the ineligible cases.


def _backend_mod():
    from tensorrt_llm._torch.visual_gen.cute_dsl_kernels.blackwell import sol_attn_backend

    return sol_attn_backend


@pytest.mark.parametrize(
    "make,expect",
    [
        (lambda: torch.randn(1, 4, 2, 128), "not a CUDA tensor"),
        (lambda: torch.randn(1, 4, 2, 64), "not a CUDA tensor"),
        (lambda: torch.randn(1, 4, 128), "not a CUDA tensor"),
    ],
    ids=["cpu-ok-shape", "cpu-wrong-head-dim", "cpu-wrong-rank"],
)
def test_ineligible_reason_is_reported(make, expect):
    """Ineligibility must name a reason, never fail silently."""
    reason = _backend_mod().sol_attn_ineligible_reason(make())
    assert reason is not None and expect in reason
    assert not _backend_mod().sol_attn_supported(make())


def test_strict_raises_on_ineligible_input(monkeypatch):
    """SOL_ATTN_STRICT=1 must cover the shape/dtype/arch path, not just kernel
    exceptions. Without this, an unsupported arch degrades to dense silently
    even under STRICT, and the counters the PR relies on cannot be trusted."""
    sab = _backend_mod()
    monkeypatch.setenv("SOL_ATTN_STRICT", "1")
    q = k = v = torch.randn(1, 4, 2, 128)  # CPU -> ineligible
    with pytest.raises(RuntimeError, match="cannot run the CuTe kernel"):
        sab._run_sol_attn_bthd(q, k, v)


def test_ineligible_falls_back_to_dense_and_counts(monkeypatch):
    """Without STRICT the same input degrades to dense and increments the counter."""
    sab = _backend_mod()
    monkeypatch.delenv("SOL_ATTN_STRICT", raising=False)
    sab.reset_sol_attn_stats()
    q = k = v = torch.randn(1, 4, 2, 128)
    out = sab._run_sol_attn_bthd(q, k, v)
    ref = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    ).transpose(1, 2)
    assert torch.allclose(out, ref), "dense fallback must be plain SDPA"
    assert sab.get_sol_attn_stats()["dense_fallback_calls"] == 1
    assert sab.get_sol_attn_stats()["kernel_calls"] == 0


def test_supported_archs_matches_kernel_dispatch_map():
    """SUPPORTED_ARCHS is a hand-copy of interface.py's _CUTE_BACKENDS. If they
    drift, eligibility silently rejects an arch the kernel actually supports."""
    from tensorrt_llm._torch.visual_gen.cute_dsl_kernels.blackwell.sol_attn import interface

    assert _backend_mod().SUPPORTED_ARCHS == frozenset(interface._CUTE_BACKENDS)


def test_quant_attention_config_rejected_with_sol_attn():
    """Sol-Attn replaces the dense CuTeDSL path, so quantized attention cannot
    compose with it; accepting the pair would silently ignore the quant request."""
    from tensorrt_llm.visual_gen.args import QuantAttentionConfig

    with pytest.raises(ValueError, match="mutually exclusive"):
        AttentionConfig(
            backend="CUTEDSL",
            quant_attention_config=QuantAttentionConfig(),
            sparse_attention_config=SolAttnAttentionConfig(tau=2.0),
        )


def test_zero_cutoff_rejected():
    """0.0 is the natural thing to type for 'no prefix', but it would run dense
    on every step and turn Sol-Attn off entirely. Must be rejected, not silent."""
    with pytest.raises(ValueError):
        SolAttnAttentionConfig(tau=2.0, disabled_until_timestep=0.0)
    assert SolAttnAttentionConfig(tau=2.0).disabled_until_timestep is None


@pytest.mark.skip(
    reason=(
        "TODO(sol-attn): numerical equivalence vs dense SDPA at zero/near-zero routing "
        "(the analogue of VSA's test_cute_kernel_matches_dense_at_full_topk) needs the "
        "exact tau/thresh_type combination that guarantees full (non-sparse) block "
        "routing, which is not simply tau=0 because Sol-Attn's routing is score-derived "
        "rather than a plain top-k like VSA's. Deriving it requires reading "
        "cute_dsl_kernels/blackwell/sol_attn/interface.py's routing math and calibrating "
        "rtol/atol on real sm100 hardware. This would be a unit-level complement to the "
        "end-to-end accuracy evidence recorded in the pull request, not a replacement."
    )
)
def test_cute_kernel_matches_dense_placeholder():
    pass


def test_kv_splits_rejects_unsupported_value():
    """kv_splits is constrained at the config layer: an out-of-range value is
    otherwise rejected deep inside the kernel and caught by the blanket
    except, silently degrading the entire run to dense attention."""
    with pytest.raises(ValueError):
        SolAttnAttentionConfig(tau=2.0, kv_splits="4")
    assert SolAttnAttentionConfig(tau=2.0).kv_splits == "auto"


def _is_dynamo_disabled(fn) -> bool:
    """True if `fn` is wrapped by torch.compiler.disable / torch._dynamo.disable."""
    target = getattr(fn, "__func__", fn)
    return bool(getattr(target, "_torchdynamo_disable", False))


def test_kernel_launch_is_opaque_to_dynamo():
    """The CuTe DSL launch boundary must be @torch.compiler.disable'd.

    Without it Dynamo traces into the CuTe DSL JIT builder and retraces on every
    call: near two orders of magnitude slower on B200 (2496.9 s mean denoise
    without it), and silent -- it looks like torch.compile simply not paying off.
    """
    assert _is_dynamo_disabled(_backend_mod()._run_sol_attn_bthd), (
        "_run_sol_attn_bthd must be decorated with @torch.compiler.disable"
    )


def test_timestep_scalar_read_is_opaque_to_dynamo():
    """The dense-prefix `.item()` must stay in eager.

    Otherwise it graph-breaks the enclosing block once per attention layer.
    """
    assert _is_dynamo_disabled(SolAttnAttention._dense_by_step), (
        "SolAttnAttention._dense_by_step must be decorated with @torch.compiler.disable"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_dense_paths_use_cutedsl_backend(monkeypatch):
    """All three dense paths must reach the configured backend's dense kernel.

    Sol-Attn does dense attention on the `dense_layers` guard, the
    `disabled_until_timestep` prefix, and kernel-ineligibility fallback. If those
    call torch SDPA instead of `cute_dsl_fmha_fwd`, a `backend: CUTEDSL` run
    differs from a `backend: CUTEDSL` dense baseline on those steps, and any A/B
    against that baseline measures a backend swap rather than sparsity. Measured
    at LPIPS 0.214 on Wan2.2-T2V-A14B before this was fixed, against a 0.25 gate.

    The CPU-tensor tests above cannot see this: `_dense` falls back to SDPA when
    `q.is_cuda` is false, so they exercise the wrong branch by construction.
    """
    import tensorrt_llm._torch.visual_gen.attention_backend.cute_dsl.sol_attn as sol_attn_mod

    device = torch.device("cuda")
    q = k = v = torch.randn(1, 64, 2, 128, device=device, dtype=torch.bfloat16)

    def _make():
        a = SolAttnAttention(layer_idx=0, num_heads=2, head_dim=128)
        calls = {"n": 0}
        real = a._dense_backend.forward

        def _spy(*args, **kwargs):
            calls["n"] += 1
            return real(*args, **kwargs)

        monkeypatch.setattr(a._dense_backend, "forward", _spy)
        return a, calls

    # 1. dense prefix: timestep at/above the cutoff
    a, calls = _make()
    a.disabled_until_timestep = 0.9
    a.dense_layers = frozenset()
    a.forward(q, k, v, timestep=torch.tensor(0.95))
    assert calls["n"] == 1, "dense prefix did not use the CuTeDSL dense kernel"

    # 2. dense_layers guard
    a, calls = _make()
    a.disabled_until_timestep = None
    a.dense_layers = frozenset({0})
    a.forward(q, k, v)
    assert calls["n"] == 1, "dense_layers guard did not use the CuTeDSL dense kernel"

    # 3. ineligibility fallback, reached through `dense_fn`
    a, calls = _make()
    a.disabled_until_timestep = None
    a.dense_layers = frozenset()
    monkeypatch.setattr(
        sol_attn_mod,
        "_sol_attn_run",
        lambda *args, **kw: kw["dense_fn"](*args[:3]),
    )
    a.forward(q, k, v)
    assert calls["n"] == 1, "dense_fn did not route the fallback to the CuTeDSL dense kernel"
