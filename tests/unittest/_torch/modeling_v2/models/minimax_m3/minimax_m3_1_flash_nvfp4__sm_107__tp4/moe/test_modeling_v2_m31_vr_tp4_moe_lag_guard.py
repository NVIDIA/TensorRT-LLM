# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The m3_moe phase-lag guard on the sm_107 tp4 engine (op.moe_lag_config, the kernel module's lag check).

The kernel hands each group's FC1 completion to its FC2 B builder through fc1_done, an mbarrier ring with no empty
barrier. Lag <= 3 runs on 4 barriers and is provably safe. From lag 4 the kernel gives every group its own barrier (8
>= the groups of an 8-token call), so no barrier completes twice in a call. On a 4-barrier ring, lag >= 4 with more
than 16 distinct experts can complete a barrier twice before the builder waits on it (a race), and lag >= 5 with 21+
distinct experts hangs deterministically. Claims checked here:

- a state accepts lag 2 (served), 3 and 4 (the MOE_HANDOFF lag at 8 tokens, on the per-group ring), and rejects lag
  > 4 at build (a ValueError naming the fc1_done ring), before any allocation or compile;
- the kernel module picks the ring for its lag (4 barriers to lag 3, 8 from lag 4), refuses a 4-barrier ring with
  lag >= 4 and more than 16 distinct experts unless lag_unsafe is set, and runs the configured lag (the plan's);
- the EXT_ROUTE build builds at lag 2 with the fused build's geometry, and inherits the lag check (these checks skip
  on a tree without MOE_EXT_ROUTE).

CPU: the op's guard, plus the kernel module evaluated with its cute / cutlass imports stubbed (trace-time constants
only, nothing compiled). SM 10.7: the state build itself.
"""

import sys
import types
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__tp4.kernels.m3_moe import (  # noqa: E501
    op as moe_op,
)

_ON_GPU = torch.cuda.is_available() and torch.cuda.get_device_capability() == (10, 7)
needs_sm107 = pytest.mark.skipif(not _ON_GPU, reason="the engine runs on SM 10.7 only")
# MOE_EXT_ROUTE may land after this guard: its checks run once the op has it
_HAS_EXT = hasattr(moe_op, "ext_route_mode")
needs_ext = pytest.mark.skipif(not _HAS_EXT, reason="MOE_EXT_ROUTE is not in this tree")

# ---------------------------------------------------------------- the kernel module, imports stubbed


class _Stub:
    """Any cute / cutlass name: a decorator returns its function, anything else another stub."""

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        return _Stub()

    def __getattr__(self, name):
        return _Stub()

    def __getitem__(self, key):
        return _Stub()

    def __or__(self, other):
        return _Stub()

    __ror__ = __or__


_STUBBED = (
    "cuda",
    "cuda.bindings",
    "cuda.bindings.driver",
    "cutlass",
    "cutlass.cute",
    "cutlass.experimental",
    "cutlass.experimental.cuda",
    "cutlass._mlir",
    "cutlass._mlir.dialects",
    "cutlass._mlir.dialects.llvm",
    "cutlass._mlir.extras",
    "cutlass._mlir.extras.types",
    "cutlass.cutlass_dsl",
    "cutlass.experimental.primitives",
)


def _stub_module(name: str) -> types.ModuleType:
    mod = types.ModuleType(name)
    mod.__getattr__ = lambda attr: _Stub()
    return mod


def _kernel(config: dict) -> SimpleNamespace:
    """m3_moe_kernel.py's int / bool module constants for ``config`` (its M3_CONFIG), evaluated with stubbed imports."""
    stubs = {name: _stub_module(name) for name in _STUBBED}
    stubs["cutlass.cutlass_dsl"].dsl_user_op = lambda fn: fn
    scope = {"M3_CONFIG": dict(config), "__name__": "m3_moe_kernel_stub"}
    with open(moe_op._ENGINE_PATH) as f:
        code = compile(f.read(), moe_op._ENGINE_PATH, "exec")
    with mock.patch.dict(sys.modules, stubs):
        exec(code, scope)
    return SimpleNamespace(**{k: v for k, v in scope.items() if type(v) in (int, bool)})


def _served_config(m: int) -> dict:
    """The kernel config a shared-expert state for ``m`` tokens builds at the served TP4 cell (188 routed CTAs)."""
    cfg = {
        "m_max": m,
        "num_ctas": 188,
        "nb2_cap": 2,
        "fused_router": 1,
        "timing": 0,
        "i_tp": moe_op.INTER,
        "sh_i": moe_op.SHARED_INTER,
        "shared": 1,
        "sfa_cp": moe_op.SFA_CP_BY_M[m - 1],
        "sfa_bulk": 1,
        "k128": 1,
        "b2img": moe_op.B2IMG_BY_M[m - 1],
        "ar_pre": moe_op.AR_PRE_BY_M[m - 1],
        "out_skip": 1,
        "prod_trigger": 1,
    }
    if moe_op.STAGES_BY_M[m - 1] > 0:
        cfg["stages"] = moe_op.STAGES_BY_M[m - 1]
    if moe_op.B2_POLL_NS_BY_M[m - 1] > 0:
        cfg["b2_poll_ns"] = moe_op.B2_POLL_NS_BY_M[m - 1]
    if not moe_op.EARLY0_BY_M[m - 1]:
        cfg["early0"] = 0
    if moe_op.flat_applies(m):
        cfg.update(moe_op.FLAT_STACK)
    return cfg


def _ext_configs(cfg: dict, m: int) -> dict:
    """The EXT_ROUTE configs a state builds from ``cfg`` (op: dict(self.config, ext_route=..., ...))."""
    out = {
        "post_wait": dict(cfg, ext_route=1, route_repl=moe_op.EXT_ROUTE_REPL),
        "b1_x4": dict(cfg, ext_route=1, route_repl=moe_op.EXT_ROUTE_REPL, b1_x4=1),
    }
    if m >= moe_op.EXT_ROUTE_PREWAIT_MIN_TOKENS:
        out["pre_wait"] = dict(cfg, ext_route=2, route_repl=moe_op.EXT_ROUTE_REPL, early0=1)
    return out


# ---------------------------------------------------------------- CPU: the op's guard


def test_lag_constants() -> None:
    """Served lag 2; the caps: 3 on the 4-barrier ring (its slots - 1), 4 on the per-group ring."""
    assert moe_op.M3_MOE_LAG == 2
    assert moe_op.M3_MOE_LAG_MAX == 3 and moe_op.FC1_DONE_SLOTS == 4
    assert moe_op.M3_MOE_LAG_MAX_WIDE == 4 and moe_op.FC1_DONE_SLOTS_WIDE == 8
    assert [moe_op.fc1_done_slots(lag) for lag in range(0, 9)] == [4, 4, 4, 4, 8, 8, 8, 8, 8]
    with open(moe_op._ENGINE_PATH) as f:
        src = f.read()
    # the ring the bounds are about: FC1D mbarriers, group g on barrier _fc1d_idx(g)
    assert "fc1_done = cutlass.Array(cutlass.Int64, FC1D," in src
    assert "fc1_done.subview(_fc1d_idx(g))" in src
    for lag in (2, 3, 4):
        k = _kernel(dict(_served_config(8), lag=lag))
        assert k.FC1D == k.FC1_DONE_SLOTS == moe_op.fc1_done_slots(lag), lag


@pytest.mark.parametrize("m", range(1, 9))
@pytest.mark.parametrize("lag", [2, 3, 4])
def test_lag_accepted(monkeypatch: pytest.MonkeyPatch, lag: int, m: int) -> None:
    cfg = moe_op.moe_lag_config(lag, m)
    # lag 2 is the kernel's default: the config (and the module key) stay as before the guard
    assert cfg == ({} if lag == 2 else {"lag": lag})


@pytest.mark.parametrize("m", [1, 4, 5, 8])
@pytest.mark.parametrize("lag", [5, 6, 8])
def test_lag_rejected(monkeypatch: pytest.MonkeyPatch, lag: int, m: int) -> None:
    with pytest.raises(ValueError, match="fc1_done") as err:
        moe_op.moe_lag_config(lag, m)
    assert "admitted up to lag 4" in str(err.value)


@pytest.mark.parametrize("lag", [-1, 2.0, True, "2"])
def test_bad_lag_values(lag) -> None:
    with pytest.raises(ValueError, match="non-negative int"):
        moe_op.moe_lag_config(lag, 8)


def test_hazard_condition() -> None:
    """A 4-barrier ring can lose a phase (a race) only with lag >= 4 and a group g + 4 (> 16 distinct experts: 5+
    tokens); it hangs whatever the timing (a CTA without FC1 work) with lag >= 5 and 6+ groups (21+ experts). The
    kernel's own ring at each lag (4 barriers to lag 3, one per group from lag 4) is outside both."""
    for m in range(1, 9):
        for lag in range(0, 9):
            assert moe_op.moe_lag_hazard(lag, m, slots=4) == (lag >= 4 and m >= 5), (lag, m)
            assert not moe_op.moe_lag_hazard(lag, m), (lag, m)
    for ng in range(1, 9):
        for lag in range(0, 9):
            # the happens-before model: some group g >= 1 with g + 4 <= min(ng - 1, lag) in the no-FC1 burst
            burst = any(g + 4 <= min(ng - 1, lag) for g in range(1, ng))
            assert (
                moe_op.moe_lag_deterministic(lag, ng, slots=4) == burst == (lag >= 5 and ng >= 6)
            ), (
                lag,
                ng,
            )
            assert not moe_op.moe_lag_deterministic(lag, ng), (lag, ng)


# ---------------------------------------------------------------- CPU: the kernel module (imports stubbed)


@pytest.mark.parametrize("m", range(1, 9))
@pytest.mark.parametrize("lag", [2, 3, 4, 5, 6])
@pytest.mark.parametrize("unsafe", [False, True])
def test_kernel_lag_check(m: int, lag: int, unsafe: bool) -> None:
    """The kernel module runs the configured lag on its default ring (one barrier per group from lag 4). On a forced
    4-barrier ring it refuses LAG >= 4 with more than 16 distinct experts unless lag_unsafe."""
    cfg = dict(_served_config(m), lag=lag)
    if unsafe:
        cfg["lag_unsafe"] = 1
    k = _kernel(cfg)
    assert (
        k.LAG == lag and k.FC1D == moe_op.fc1_done_slots(lag) and (lag <= 3 or k.FC1D >= k.NG_CAP)
    )
    forced = dict(cfg, fc1d=4)
    if lag > 3 and m >= 5 and not unsafe:
        with pytest.raises(ValueError, match="fc1_done 4-barrier ring"):
            _kernel(forced)
    else:
        assert _kernel(forced).LAG == lag


@needs_ext
@pytest.mark.parametrize("m", range(1, 9))
def test_ext_route_builds_at_lag2(monkeypatch: pytest.MonkeyPatch, m: int) -> None:
    """At the served lag the op's lag config adds nothing, and every EXT_ROUTE build has LAG 2 and passes the
    geometry gate against the fused build."""
    cfg = dict(_served_config(m), **moe_op.moe_lag_config(moe_op.M3_MOE_LAG, m))
    assert "lag" not in cfg and "lag_unsafe" not in cfg
    base = _kernel(cfg)
    assert base.LAG == 2
    for name, ext_cfg in _ext_configs(cfg, m).items():
        ext = _kernel(ext_cfg)
        assert ext.LAG == 2 and ext.EXT_ROUTE in (1, 2), name
        assert moe_op.ext_route_geometry_diff(base, ext) == [], name


@needs_ext
@pytest.mark.parametrize("m", [5, 8])
@pytest.mark.parametrize("lag", [5, 6])
def test_ext_route_inherits_lag_check(monkeypatch: pytest.MonkeyPatch, m: int, lag: int) -> None:
    """The EXT_ROUTE config copies the state's config, so it carries the lag and the same refusal."""
    with pytest.raises(ValueError, match="fc1_done"):
        moe_op.moe_lag_config(lag, m)
    raw = dict(_served_config(m), lag=lag, fc1d=4)  # a lag forced past the op onto a 4-barrier ring
    for name, ext_cfg in _ext_configs(raw, m).items():
        with pytest.raises(ValueError, match="fc1_done"):
            _kernel(ext_cfg)


# ---------------------------------------------------------------- SM 10.7: the state build


def _state(monkeypatch: pytest.MonkeyPatch, lag: int, m: int = 8, ext: bool = False):
    """A state at phase lag ``lag``: M3_MOE_LAG with the MOE_HANDOFF tables off (their lag would override it)."""
    monkeypatch.setattr(moe_op, "M3_MOE_LAG", lag)
    monkeypatch.setattr(
        moe_op, "m31_enabled", lambda item: ext and item == "MOE_EXT_ROUTE", raising=False
    )
    monkeypatch.setattr(moe_op, "m31_disabled", lambda item: False)
    dev = torch.device("cuda")
    sms = torch.cuda.get_device_properties(dev).multi_processor_count
    return moe_op.M3MoeState(
        dev, num_tokens=m, num_ctas=sms - moe_op.SHARED_CTAS, shared=True, handoff=False
    )


@needs_sm107
@pytest.mark.parametrize("ext", [False, pytest.param(True, marks=needs_ext)])
@pytest.mark.parametrize("lag", [5, 6])
def test_state_rejects_lag_at_build(monkeypatch: pytest.MonkeyPatch, lag: int, ext: bool) -> None:
    modules, compiled = len(moe_op._modules), len(moe_op._compiled)
    with pytest.raises(ValueError, match="fc1_done"):
        _state(monkeypatch, lag, ext=ext)
    # rejected before the kernel module is even loaded (so before any compile)
    assert (len(moe_op._modules), len(moe_op._compiled)) == (modules, compiled)


@needs_sm107
@pytest.mark.parametrize("m", [8, 5])
def test_state_builds_lag4_on_the_group_ring(monkeypatch: pytest.MonkeyPatch, m: int) -> None:
    """Lag 4 (the MOE_HANDOFF lag at 8 tokens) builds without the override, on one fc1_done barrier per group."""
    st = _state(monkeypatch, 4, m=m)
    assert (
        st.lag == 4 and st.mod.LAG == 4 and st.config["lag"] == 4 and "lag_unsafe" not in st.config
    )
    assert int(st.mod.FC1D) == 8 >= int(st.mod.NG_CAP)


@needs_sm107
@needs_ext
@pytest.mark.parametrize("m", [8, 5, 1])
@pytest.mark.parametrize("lag", [2, 3])
def test_ext_route_state_builds_at_accepted_lag(
    monkeypatch: pytest.MonkeyPatch, m: int, lag: int
) -> None:
    st = _state(monkeypatch, lag, m=m, ext=True)
    assert st.lag == lag and st.ext_route and st.ext_mod is not None
    assert st.mod.LAG == st.ext_mod.LAG == lag
    assert st.config.get("lag", 2) == st.ext_config.get("lag", 2) == lag
    assert "lag_unsafe" not in st.config and "lag_unsafe" not in st.ext_config
