# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""PassManager unit tests (CPU-only, no checkpoint required).

Exercises the composition primitives + default-pipeline construction. The
manager only needs an `fx.GraphModule` argument to its passes, but for
ordering/composition tests a dummy stand-in suffices.
"""

import pytest

# ---------------------------------------------------------------------------
# Manager composition primitives
# ---------------------------------------------------------------------------


def test_basic_append_and_run():
    from tensorrt_llm._torch.visual_gen.auto.pass_manager import Pass, PassManager

    log: list[str] = []
    pm = PassManager()
    pm.append(Pass("a", lambda _gm: (log.append("a"), 1)[1]))
    pm.append(Pass("b", lambda _gm: (log.append("b"), 2)[1]))

    results = pm.run(_gm_sentinel())
    assert log == ["a", "b"]
    assert results == {"a": 1, "b": 2}
    assert pm.names() == ["a", "b"]
    assert len(pm) == 2
    assert "a" in pm and "z" not in pm


def test_insert_before_and_after():
    from tensorrt_llm._torch.visual_gen.auto.pass_manager import Pass, PassManager

    pm = PassManager(
        [
            Pass("alpha", lambda _gm: 0),
            Pass("gamma", lambda _gm: 0),
        ]
    )
    pm.insert_after("alpha", Pass("beta", lambda _gm: 0))
    pm.insert_before("alpha", Pass("first", lambda _gm: 0))
    assert pm.names() == ["first", "alpha", "beta", "gamma"]


def test_replace():
    from tensorrt_llm._torch.visual_gen.auto.pass_manager import Pass, PassManager

    pm = PassManager(
        [
            Pass("a", lambda _gm: "old"),
            Pass("b", lambda _gm: 0),
        ]
    )
    pm.replace("a", Pass("a", lambda _gm: "new"))
    results = pm.run(_gm_sentinel())
    assert results["a"] == "new"


def test_remove():
    from tensorrt_llm._torch.visual_gen.auto.pass_manager import Pass, PassManager

    pm = PassManager(
        [
            Pass("a", lambda _gm: 0),
            Pass("b", lambda _gm: 0),
            Pass("c", lambda _gm: 0),
        ]
    )
    pm.remove("b")
    assert pm.names() == ["a", "c"]


def test_skip_via_run_arg():
    from tensorrt_llm._torch.visual_gen.auto.pass_manager import Pass, PassManager

    log: list[str] = []
    pm = PassManager(
        [
            Pass("a", lambda _gm: log.append("a")),
            Pass("b", lambda _gm: log.append("b")),
            Pass("c", lambda _gm: log.append("c")),
        ]
    )
    pm.run(_gm_sentinel(), skip=["b"])
    assert log == ["a", "c"]


def test_unknown_anchor_raises_key_error():
    from tensorrt_llm._torch.visual_gen.auto.pass_manager import Pass, PassManager

    pm = PassManager([Pass("a", lambda _gm: 0)])
    with pytest.raises(KeyError, match=r"no pass named 'nope'"):
        pm.insert_after("nope", Pass("new", lambda _gm: 0))
    with pytest.raises(KeyError):
        pm.remove("nope")
    with pytest.raises(KeyError):
        pm.replace("nope", Pass("new", lambda _gm: 0))


def test_duplicate_name_on_append_raises():
    from tensorrt_llm._torch.visual_gen.auto.pass_manager import Pass, PassManager

    pm = PassManager([Pass("a", lambda _gm: 0)])
    with pytest.raises(ValueError, match=r"already present"):
        pm.append(Pass("a", lambda _gm: 0))
    with pytest.raises(ValueError, match=r"already present"):
        pm.insert_before("a", Pass("a", lambda _gm: 0))
    with pytest.raises(ValueError, match=r"already present"):
        pm.insert_after("a", Pass("a", lambda _gm: 0))


def test_pass_exception_propagates_and_logs():
    from tensorrt_llm._torch.visual_gen.auto.pass_manager import Pass, PassManager

    def boom(_gm):
        raise RuntimeError("simulated pass failure")

    pm = PassManager(
        [
            Pass("ok1", lambda _gm: 0),
            Pass("bad", boom),
            Pass("ok2", lambda _gm: 0),  # must NOT run after the bad one
        ]
    )
    log: list[str] = []
    pm._passes[0] = Pass("ok1", lambda _gm: log.append("ok1"))
    pm._passes[2] = Pass("ok2", lambda _gm: log.append("ok2"))
    with pytest.raises(RuntimeError, match="simulated"):
        pm.run(_gm_sentinel())
    # ok1 ran; ok2 did NOT (bad raised first).
    assert log == ["ok1"]


def test_chaining_returns_self():
    from tensorrt_llm._torch.visual_gen.auto.pass_manager import Pass, PassManager

    pm = PassManager()
    out = (
        pm.append(Pass("a", lambda _gm: 0))
        .append(Pass("c", lambda _gm: 0))
        .insert_before("c", Pass("b", lambda _gm: 0))
    )
    assert out is pm
    assert pm.names() == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# Default pipeline construction from RewritePolicy
# ---------------------------------------------------------------------------


def test_default_pipeline_full_policy():
    """All fusions enabled → all 6 passes registered in canonical order."""
    from tensorrt_llm._torch.visual_gen.auto.policy import RewritePolicy
    from tensorrt_llm._torch.visual_gen.auto.rewrite import _build_default_pass_manager

    pm = _build_default_pass_manager(RewritePolicy())  # defaults: all True
    assert pm.names() == [
        "sdpa_rewrite",
        "fuse_qkv",
        "qk_rope_single",
        "qk_rope_dual",
        "contig_for_fp8_quant",
        "strip_assert_metadata",
    ]


def test_default_pipeline_qkv_off():
    """`fuse_qkv=False` removes both QKV and qk_rope passes (qk_rope depends on QKV)."""
    from tensorrt_llm._torch.visual_gen.auto.policy import RewritePolicy
    from tensorrt_llm._torch.visual_gen.auto.rewrite import _build_default_pass_manager

    pm = _build_default_pass_manager(RewritePolicy(fuse_qkv=False))
    assert pm.names() == [
        "sdpa_rewrite",
        "contig_for_fp8_quant",
        "strip_assert_metadata",
    ]


def test_default_pipeline_qk_rope_off():
    """`fuse_qk_rope=False` keeps `fuse_qkv` but drops both qk_rope passes."""
    from tensorrt_llm._torch.visual_gen.auto.policy import RewritePolicy
    from tensorrt_llm._torch.visual_gen.auto.rewrite import _build_default_pass_manager

    pm = _build_default_pass_manager(RewritePolicy(fuse_qk_rope=False))
    assert pm.names() == [
        "sdpa_rewrite",
        "fuse_qkv",
        "contig_for_fp8_quant",
        "strip_assert_metadata",
    ]


def test_default_pipeline_env_override(monkeypatch):
    """`VISGEN_AUTO_DISABLE_QKROPE` env var disables qk_rope without code change."""
    from tensorrt_llm._torch.visual_gen.auto.policy import RewritePolicy
    from tensorrt_llm._torch.visual_gen.auto.rewrite import _build_default_pass_manager

    monkeypatch.setenv("VISGEN_AUTO_DISABLE_QKROPE", "1")
    pm = _build_default_pass_manager(RewritePolicy())
    assert "qk_rope_single" not in pm
    assert "qk_rope_dual" not in pm
    assert "fuse_qkv" in pm  # still present


# ---------------------------------------------------------------------------
# Adapter hook
# ---------------------------------------------------------------------------


def test_customize_passes_hook_fires(monkeypatch):
    """Adapter's `customize_passes(pm)` is called between pipeline construction
    and run, and adapter changes are reflected at run time.
    """
    from tensorrt_llm._torch.visual_gen.auto.pass_manager import Pass
    from tensorrt_llm._torch.visual_gen.auto.policy import RewritePolicy

    # Don't depend on importing a real adapter — fake one with just the
    # hook + a sentinel pass.
    class _Fake:
        def customize_passes(self, pm):
            pm.insert_after("sdpa_rewrite", Pass("custom", lambda _gm: "ran"))

    # Build the default pipeline + fire the hook + capture order.
    from tensorrt_llm._torch.visual_gen.auto.rewrite import _build_default_pass_manager

    pm = _build_default_pass_manager(RewritePolicy())
    _Fake().customize_passes(pm)
    assert pm.names() == [
        "sdpa_rewrite",
        "custom",  # ← spliced in
        "fuse_qkv",
        "qk_rope_single",
        "qk_rope_dual",
        "contig_for_fp8_quant",
        "strip_assert_metadata",
    ]


def test_adapter_default_customize_is_noop():
    """Default `VisGenFamilyAdapter.customize_passes` does nothing — adapters
    that don't override see the canonical pipeline unchanged.
    """
    from tensorrt_llm._torch.visual_gen.auto.families import FluxAdapter
    from tensorrt_llm._torch.visual_gen.auto.pass_manager import PassManager

    pm_before = PassManager()
    FluxAdapter().customize_passes(pm_before)
    assert pm_before.names() == []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _gm_sentinel():
    """Return any object — the composition tests above never inspect it."""
    return object()
