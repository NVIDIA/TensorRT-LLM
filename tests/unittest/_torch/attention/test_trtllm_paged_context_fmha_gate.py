# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kernel-availability gate for paged-context FMHA in the TRTLLM backend.

``TrtllmAttentionMetadata.__post_init__`` enables ``use_paged_context_fmha``
whenever chunked prefill, KV block reuse or speculative draft tokens are
configured. If the fused context FMHA kernel is absent for the SM/head-size
combination (head_dim 64 on SM 103 is a proven case), attentionOp.cpp falls
back to unfused MHA whose context path builds K/V from the current chunk only:
the cached prefix is dropped from attention and then overwritten by the
chunk's write-back. The same greedy prompt sent twice then returns different,
both-coherent answers with no error or warning at request time.

The gate lives in ``FallbackFmha.validate_metadata`` -- the FMHA library that
owns the thop attention op -- and is invoked through the
``Fmha.validate_metadata`` hook for every enabled library at metadata
construction. These tests pin the gate that turns that silent wrong answer
into a construction-time error. Most run on CPU: SM version, the native
kernel lookup and buffer allocation are mocked, the manager is a stub. The
last two need a GPU -- they run the real kernel lookup, which reads the
kernels loaded for the current device.
"""

from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from tensorrt_llm._torch.attention.backends.fmha.fallback import FallbackFmha
from tensorrt_llm._torch.attention.backends.fmha.interface import Fmha
from tensorrt_llm._torch.attention.backends.interface import AttentionRuntimeFeatures
from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata
from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.bindings import DataType
from tensorrt_llm.bindings.internal import thop

# Sentinel: build a manager stub WITHOUT the attribute, mirroring managers
# that predate the exemption path.
_ABSENT = object()

_FALLBACK_SM_VERSION_TARGET = "tensorrt_llm._torch.attention.backends.fmha.fallback.get_sm_version"


def _make_metadata(
    head_dim,
    sm_version,
    features,
    kv_dtype=_ABSENT,
    tokens_per_block=32,
    kernel_exists=True,
):
    """Construct real metadata with mocked SM version, a stubbed native kernel
    lookup and no CUDA buffers.

    ``create=True`` on the lookup stub: the binding may be absent from a
    build whose bindings predate it, and the gate treats that as unverifiable
    (fail closed), which ``test_missing_lookup_binding_fails_closed`` covers
    explicitly.
    """
    manager = SimpleNamespace(head_dim=head_dim)
    if kv_dtype is not _ABSENT:
        manager.dtype = kv_dtype
    if tokens_per_block is not _ABSENT:
        manager.tokens_per_block = tokens_per_block
    with (
        mock.patch(_FALLBACK_SM_VERSION_TARGET, return_value=sm_version),
        mock.patch.object(
            thop, "fused_context_fmha_kernel_exists", return_value=kernel_exists, create=True
        ),
        mock.patch.object(TrtllmAttentionMetadata, "_post_init_with_buffers"),
    ):
        return TrtllmAttentionMetadata(
            max_num_requests=4,
            max_num_tokens=1024,
            kv_cache_manager=manager,
            runtime_features=features,
        )


ALL_FEATURES = AttentionRuntimeFeatures(
    chunked_prefill=True, cache_reuse=True, has_speculative_draft_tokens=True
)


def test_default_validate_metadata_hook_accepts_everything():
    """The interface hook is a no-op unless a library declares a constraint."""
    assert Fmha.validate_metadata(mock.Mock()) is None


@pytest.mark.parametrize(
    "features",
    [
        AttentionRuntimeFeatures(cache_reuse=True),
        AttentionRuntimeFeatures(chunked_prefill=True),
        AttentionRuntimeFeatures(has_speculative_draft_tokens=True),
        ALL_FEATURES,
    ],
    ids=["cache_reuse", "chunked_prefill", "spec_draft", "all"],
)
def test_kernel_absent_configuration_is_refused(features):
    """head_dim 64 on SM 103 has no fused context FMHA kernel: raise, do not
    enable use_paged_context_fmha and silently drop the cached prefix."""
    with pytest.raises(RuntimeError) as exc:
        _make_metadata(head_dim=64, sm_version=103, features=features)
    message = str(exc.value)
    assert "64" in message and "103" in message
    assert "FlashInfer" in message  # the remedy must be actionable


def test_kernel_absent_head_dim_in_list_is_refused():
    """VSWA managers may carry a per-window head_dim list."""
    with pytest.raises(RuntimeError, match="64"):
        _make_metadata(head_dim=[128, 64], sm_version=103, features=ALL_FEATURES)


def test_supported_head_dim_is_admitted():
    metadata = _make_metadata(head_dim=128, sm_version=103, features=ALL_FEATURES)
    assert metadata.use_paged_context_fmha


def test_other_sm_versions_are_not_blocked():
    """The blocklist is per-SM: nothing is proven absent elsewhere."""
    metadata = _make_metadata(head_dim=64, sm_version=90, features=ALL_FEATURES)
    assert metadata.use_paged_context_fmha


def test_no_reuse_features_no_gate():
    """Without reuse/chunking/spec the paged-context path is never taken, so
    the configuration stays legal even where the kernel is absent."""
    metadata = _make_metadata(head_dim=64, sm_version=103, features=AttentionRuntimeFeatures())
    assert not metadata.use_paged_context_fmha


def test_disabled_fallback_library_disables_gate(monkeypatch):
    """The gate belongs to the fallback library. With the fallback disabled
    the thop attention path cannot run, so its kernel-presence constraint
    must not fire."""
    monkeypatch.setenv("TLLM_FMHA_LIBS", "-fallback")
    metadata = _make_metadata(head_dim=64, sm_version=103, features=ALL_FEATURES)
    assert metadata.use_paged_context_fmha


@pytest.mark.parametrize(
    "kv_dtype",
    [DataType.FP8, DataType.NVFP4, DataType.BF16, DataType.HALF],
    ids=["fp8", "nvfp4", "bf16", "half"],
)
def test_exempt_kv_dtypes_admit_blocked_combination(kv_dtype):
    """A full build carries the fused kernel for FP8, NVFP4 and matched 16-bit
    KV caches on SM 103 / head_dim 64 (the SM103 L0 suites run paged-context
    attention with a BF16 KV cache): the dtype exemption must admit the
    otherwise-blocked configuration once the kernel lookup confirms it."""
    metadata = _make_metadata(head_dim=64, sm_version=103, features=ALL_FEATURES, kv_dtype=kv_dtype)
    assert metadata.use_paged_context_fmha


@pytest.mark.parametrize("kv_dtype", [DataType.FLOAT, DataType.INT8], ids=["float", "int8"])
def test_unlisted_kv_dtypes_stay_refused(kv_dtype):
    """The exemption is a per-dtype allowlist, not a bypass: a KV dtype with
    no proven-present kernel must still be refused."""
    with pytest.raises(RuntimeError, match="64"):
        _make_metadata(head_dim=64, sm_version=103, features=ALL_FEATURES, kv_dtype=kv_dtype)


def test_exemption_requires_all_absent_head_dims_covered():
    """With a per-window head_dim list, every blocked head_dim must be exempt
    for the given dtype. head_dim 64 is exempt for FP8 on SM 103; a second
    blocked head_dim without an exemption entry must keep the refusal."""
    with mock.patch.dict(FallbackFmha.CONTEXT_FMHA_ABSENT_HEAD_DIMS, {103: (64, 72)}):
        with pytest.raises(RuntimeError, match="72"):
            _make_metadata(
                head_dim=[64, 72],
                sm_version=103,
                features=ALL_FEATURES,
                kv_dtype=DataType.FP8,
            )


def test_exemption_is_keyed_by_sm_and_head_dim():
    """The (103, 64) exemption must not leak to other blocked combinations."""
    with mock.patch.dict(FallbackFmha.CONTEXT_FMHA_ABSENT_HEAD_DIMS, {90: (64,)}):
        with pytest.raises(RuntimeError, match="64"):
            _make_metadata(head_dim=64, sm_version=90, features=ALL_FEATURES, kv_dtype=DataType.FP8)


def test_unblocked_combination_needs_no_exemption():
    """Off the blocklist, any KV dtype is admitted without an exemption."""
    metadata = _make_metadata(
        head_dim=128, sm_version=103, features=ALL_FEATURES, kv_dtype=DataType.BF16
    )
    assert metadata.use_paged_context_fmha


def test_manager_without_dtype_fails_closed():
    """A manager exposing no ``dtype`` cannot prove the kernel present; the
    blocked combination must stay refused rather than assume the exemption."""
    with pytest.raises(RuntimeError, match="64"):
        _make_metadata(head_dim=64, sm_version=103, features=ALL_FEATURES)


def test_manager_without_tokens_per_block_fails_closed():
    """The native lookup needs the page size the engine will run with. A
    manager that does not expose one cannot be checked, so the exemption is
    not honoured."""
    with pytest.raises(RuntimeError, match="64"):
        _make_metadata(
            head_dim=64,
            sm_version=103,
            features=ALL_FEATURES,
            kv_dtype=DataType.FP8,
            tokens_per_block=_ABSENT,
        )


def test_exemption_queries_the_native_kernel_lookup():
    """The exemption path must ask the build what it contains, with the
    combination the table claims and the page size the engine will use."""
    with (
        mock.patch(_FALLBACK_SM_VERSION_TARGET, return_value=103),
        mock.patch.object(
            thop, "fused_context_fmha_kernel_exists", return_value=True, create=True
        ) as lookup,
        mock.patch.object(TrtllmAttentionMetadata, "_post_init_with_buffers"),
    ):
        metadata = TrtllmAttentionMetadata(
            max_num_requests=4,
            max_num_tokens=1024,
            kv_cache_manager=SimpleNamespace(head_dim=64, dtype=DataType.FP8, tokens_per_block=32),
            runtime_features=ALL_FEATURES,
        )
    assert metadata.use_paged_context_fmha
    # output_dtype follows the binding's probe convention: FP8 output for the
    # FP8/NVFP4 KV kernels, matched 16-bit output otherwise.
    lookup.assert_called_once_with(
        head_size=64,
        kv_cache_dtype=DataType.FP8,
        tokens_per_block=32,
        output_dtype=DataType.FP8,
    )


def test_exemption_is_refused_when_the_build_has_no_kernel():
    """The exemption table is a hand-maintained assertion about the kernel
    set. When the build no longer has that kernel the guard must fail at
    construction, naming the combination and the table to update -- not pass
    the claim through to a runtime that will silently fall back."""
    with pytest.raises(RuntimeError) as exc:
        _make_metadata(
            head_dim=64,
            sm_version=103,
            features=ALL_FEATURES,
            kv_dtype=DataType.FP8,
            kernel_exists=False,
        )
    message = str(exc.value)
    assert "64" in message and "103" in message  # the claimed combination
    assert str(DataType.FP8) in message or "FP8" in message  # the KV dtype
    assert "CONTEXT_FMHA_PRESENT_KV_DTYPES" in message  # what to update
    assert "32" in message  # the page size that was queried


def test_missing_lookup_binding_fails_closed():
    """A build whose bindings predate the kernel lookup cannot verify the
    exemption, so the blocked combination stays refused, with the error
    naming the missing binding rather than crashing on an attribute error."""
    manager = SimpleNamespace(head_dim=64, dtype=DataType.FP8, tokens_per_block=32)
    with (
        mock.patch(_FALLBACK_SM_VERSION_TARGET, return_value=103),
        mock.patch.object(TrtllmAttentionMetadata, "_post_init_with_buffers"),
        mock.patch.object(thop, "fused_context_fmha_kernel_exists", new=None, create=True),
    ):
        with pytest.raises(RuntimeError, match="fused_context_fmha_kernel_exists"):
            TrtllmAttentionMetadata(
                max_num_requests=4,
                max_num_tokens=1024,
                kv_cache_manager=manager,
                runtime_features=ALL_FEATURES,
            )


# Paged-KV context FMHA kernels are built for 32-token pages; a page size with
# no kernel is a real absence, not a test artifact.
_PAGED_CONTEXT_TOKENS_PER_BLOCK = 32
# A head width the SM100-family dispatcher has no context kernel for -- the
# forced miss for the live lookup. Other architectures route to FMHA-v2
# kernels with a different supported set (SM90 does carry head size 96), so
# this absence holds only within the SM100 family, mirroring
# test_context_fmha_kernel_presence.py.
_UNSUPPORTED_HEAD_SIZE = 96

_LIVE_LOOKUP_UNAVAILABLE = not torch.cuda.is_available() or not hasattr(
    thop, "fused_context_fmha_kernel_exists"
)
_LIVE_LOOKUP_SKIP_REASON = "needs a GPU and bindings built with fused_context_fmha_kernel_exists"


@pytest.mark.skipif(_LIVE_LOOKUP_UNAVAILABLE, reason=_LIVE_LOOKUP_SKIP_REASON)
def test_native_lookup_reports_absent_for_an_unbuilt_head_size():
    """The live lookup must be able to say no: without that, the exemption
    check above can only ever pass."""
    if not 100 <= get_sm_version() < 110:
        pytest.skip("the unsupported head-size case targets the SM100-family dispatcher")
    assert not thop.fused_context_fmha_kernel_exists(
        head_size=_UNSUPPORTED_HEAD_SIZE,
        kv_cache_dtype=DataType.BF16,
        tokens_per_block=_PAGED_CONTEXT_TOKENS_PER_BLOCK,
        output_dtype=DataType.BF16,
    )


@pytest.mark.skipif(_LIVE_LOOKUP_UNAVAILABLE, reason=_LIVE_LOOKUP_SKIP_REASON)
def test_exemption_table_matches_the_kernels_this_build_has():
    """Every exemption claimed for this device's SM must hold against the
    live kernel lookup. This is the regression that the check exists for: if
    a kernel-set change drops one, this fails here rather than in flight.

    "The kernels this build has" is literal. cuda_configuration.cmake stamps
    ``-DEXCLUDE_SM_<arch>`` for every architecture that the build's
    ``--cuda_architectures`` does not name, and that macro compiles the
    matching block of the trtllm-gen cubin table out. So a build whose
    architecture list omits the SM of the device it is then run on has no
    kernels for that SM and fails here -- correctly, because on such a build
    the runtime falls back to unfused MHA and the exemption must not be
    honoured. Check the build's architecture list before suspecting the
    kernel set."""
    sm = get_sm_version()
    claimed = {
        (claimed_sm, head_dim): dtypes
        for (
            claimed_sm,
            head_dim,
        ), dtypes in FallbackFmha.CONTEXT_FMHA_PRESENT_KV_DTYPES.items()
        if claimed_sm == sm
    }
    if not claimed:
        pytest.skip(f"no exemptions claimed for SM {sm}")
    for (_, head_dim), dtypes in claimed.items():
        for kv_dtype in dtypes:
            # Probe with the output precision the kernel table pairs with this
            # KV precision, mirroring FallbackFmha.validate_metadata.
            output_dtype = DataType.FP8 if kv_dtype in (DataType.FP8, DataType.NVFP4) else kv_dtype
            assert thop.fused_context_fmha_kernel_exists(
                head_size=head_dim,
                kv_cache_dtype=kv_dtype,
                tokens_per_block=_PAGED_CONTEXT_TOKENS_PER_BLOCK,
                output_dtype=output_dtype,
            ), (
                f"CONTEXT_FMHA_PRESENT_KV_DTYPES claims a fused context FMHA "
                f"kernel for head_dim {head_dim} / {kv_dtype} on SM {sm}, but "
                f"this build has none. Either the kernel set changed and the "
                f"entry must go, or this build's --cuda_architectures does "
                f"not name SM {sm} and so carries no kernels for it."
            )


def test_unknown_head_dim_is_not_guessed():
    """A manager that exposes no head_dim (or no manager at all) cannot be
    judged; the gate must not block it on a guess."""
    with (
        mock.patch(_FALLBACK_SM_VERSION_TARGET, return_value=103),
        mock.patch.object(TrtllmAttentionMetadata, "_post_init_with_buffers"),
    ):
        metadata = TrtllmAttentionMetadata(
            max_num_requests=4,
            max_num_tokens=1024,
            kv_cache_manager=None,
            runtime_features=ALL_FEATURES,
        )
    assert metadata.use_paged_context_fmha
