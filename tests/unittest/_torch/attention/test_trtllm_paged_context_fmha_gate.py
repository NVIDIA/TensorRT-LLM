# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Paged-context refusals of the native attention op.

``TrtllmAttentionMetadata.__post_init__`` enables ``use_paged_context_fmha``
whenever chunked prefill, KV block reuse or speculative draft tokens are
configured. Without a fused context FMHA kernel for the configuration, the
context phase would reach the unfused library, whose context path builds K/V
from the current chunk only: the cached prefix would be dropped from attention
and then overwritten by the chunk's write-back. ``unfused_is_supported``
refuses that call with an error instead. A page size that only the trtllm-gen
kernels outside the op read is refused the same way for every native library.
"""

import pytest
import torch
from fmha_test_utils import native_batch, native_forward_args

from tensorrt_llm._torch.attention.backends.fmha.fallback import FallbackFmha
from tensorrt_llm._torch.attention.backends.fmha.trtllm_gen import TrtllmGenFmha
from tensorrt_llm._torch.attention.backends.fmha.unfused import UnfusedFmha

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

# FP32 attention has no fused context kernel, so its context phase runs unfused.
_UNFUSED_DTYPE = torch.float32


def _is_supported(fmha_cls, batch, **overrides) -> bool:
    attn, metadata, qkv = batch
    forward_args = native_forward_args(attn, metadata, qkv, **overrides)
    return fmha_cls(attn).is_supported(qkv, None, None, metadata, forward_args)


def test_paged_context_without_fused_kernel_is_refused():
    with native_batch(
        _UNFUSED_DTYPE, num_contexts=1, num_generations=0, chunked_prefill=True
    ) as batch:
        with pytest.raises(RuntimeError, match="requires a fused context FMHA kernel"):
            _is_supported(FallbackFmha, batch)
        with pytest.raises(RuntimeError, match="requires a fused context FMHA kernel"):
            _is_supported(UnfusedFmha, batch)


def test_unfused_context_without_paged_context_is_admitted():
    with native_batch(_UNFUSED_DTYPE, num_contexts=1, num_generations=0) as batch:
        assert _is_supported(UnfusedFmha, batch)


def test_generation_only_batch_skips_the_paged_context_refusal():
    with native_batch(
        _UNFUSED_DTYPE, num_contexts=0, num_generations=2, chunked_prefill=True
    ) as batch:
        assert _is_supported(FallbackFmha, batch)


@pytest.mark.parametrize("fmha_cls", [FallbackFmha, TrtllmGenFmha, UnfusedFmha])
def test_trtllm_gen_only_page_size_refuses_paged_prefill(fmha_cls):
    with native_batch(
        torch.bfloat16,
        num_contexts=1,
        num_generations=0,
        chunked_prefill=True,
        extra_page_sizes=frozenset({32}),
    ) as batch:
        with pytest.raises(RuntimeError, match="32-token KV pages"):
            _is_supported(fmha_cls, batch)


<<<<<<< HEAD
def test_trtllm_gen_only_page_size_admits_generation():
    with native_batch(
        torch.bfloat16,
        num_contexts=0,
        num_generations=2,
        chunked_prefill=True,
        extra_page_sizes=frozenset({32}),
    ) as batch:
        assert _is_supported(FallbackFmha, batch)
=======
def test_metadata_construction_does_not_run_the_gate():
    """The gate moved from metadata construction to FMHA dispatch: building
    metadata for a blocked combination must succeed without consulting the
    SM version or the kernel lookup, with the paged-context flag set. The
    fallback refuses the configuration when a context batch reaches it."""
    with mock.patch.object(thop, "fused_context_fmha_kernel_exists", create=True) as lookup:
        metadata = _make_metadata(head_dim=64, features=ALL_FEATURES)
    assert metadata.use_paged_context_fmha
    lookup.assert_not_called()


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
    serve the batch and silently drop the cached prefix."""
    with pytest.raises(RuntimeError) as exc:
        _gate(head_dim=64, sm_version=103, features=features)
    message = str(exc.value)
    assert "64" in message and "103" in message
    assert "FlashInfer" in message  # the remedy must be actionable


def test_kernel_absent_head_dim_in_list_is_refused():
    """VSWA managers may carry a per-window head_dim list."""
    with pytest.raises(RuntimeError, match="64"):
        _gate(head_dim=[128, 64], sm_version=103, features=ALL_FEATURES)


def test_supported_head_dim_is_admitted():
    metadata = _gate(head_dim=128, sm_version=103, features=ALL_FEATURES)
    assert metadata.use_paged_context_fmha


def test_other_sm_versions_are_not_blocked():
    """The blocklist is per-SM: nothing is proven absent elsewhere."""
    metadata = _gate(head_dim=64, sm_version=90, features=ALL_FEATURES)
    assert metadata.use_paged_context_fmha


def test_no_reuse_features_no_gate():
    """Without reuse/chunking/spec the paged-context path is never taken, so
    the configuration stays legal even where the kernel is absent."""
    metadata = _gate(head_dim=64, sm_version=103, features=AttentionRuntimeFeatures())
    assert not metadata.use_paged_context_fmha


def _make_dispatch_inputs(num_contexts):
    """Stub metadata/forward_args pairs for exercising the real
    ``is_supported`` entry point on a blocked configuration."""
    metadata = SimpleNamespace(
        use_paged_context_fmha=True,
        kv_cache_manager=SimpleNamespace(head_dim=64),
        runtime_features=ALL_FEATURES,
        num_contexts=num_contexts,
        num_generations=0 if num_contexts else 2,
        num_ctx_tokens=num_contexts * 4,
        helix_position_offsets=None,
        _helix_spec_tokens_valid=False,
        is_cross=False,
    )
    forward_args = SimpleNamespace(
        sparse_runtime_params=SimpleNamespace(block_sparse_inputs=None),
        attention_mask=PredefinedAttentionMask.CAUSAL,
        variable_window_token_starts=None,
        variable_window_token_ends=None,
        update_kv_cache=True,
        kv_append=None,
    )
    q = torch.zeros(num_contexts * 4 + metadata.num_generations, 8, dtype=torch.float16)
    return q, metadata, forward_args


def test_dispatch_refuses_context_batch_in_blocked_cell():
    """A batch with a context phase reaching the fallback in a blocked,
    unverifiable configuration must raise through ``is_supported`` -- the
    dispatch entry point -- not return False into the generic no-library
    error."""
    q, metadata, forward_args = _make_dispatch_inputs(num_contexts=2)
    attn = mock.Mock()  # held: the library keeps only a weak reference
    fmha = FallbackFmha(attn)
    with mock.patch(_FALLBACK_SM_VERSION_TARGET, return_value=103):
        with pytest.raises(RuntimeError) as exc:
            fmha.is_supported(q, None, None, metadata, forward_args)
    assert "FlashInfer" in str(exc.value)


def test_dispatch_admits_generation_only_batch_in_blocked_cell():
    """Generation-only batches never run the context path, so they must pass
    without consulting the gate. Whether a batch has a context phase is part
    of the FMHA selection cache key, so this admission cannot be replayed for
    a context batch."""
    q, metadata, forward_args = _make_dispatch_inputs(num_contexts=0)
    attn = mock.Mock()  # held: the library keeps only a weak reference
    fmha = FallbackFmha(attn)
    with mock.patch(_FALLBACK_SM_VERSION_TARGET, return_value=103) as sm_lookup:
        assert fmha.is_supported(q, None, None, metadata, forward_args)
    sm_lookup.assert_not_called()


@pytest.mark.parametrize(
    "kv_dtype",
    [DataType.FP8, DataType.NVFP4, DataType.BF16, DataType.HALF],
    ids=["fp8", "nvfp4", "bf16", "half"],
)
def test_probe_present_kv_dtypes_admit_blocked_combination(kv_dtype):
    """Inside a blocklisted cell the native kernel lookup decides per KV
    dtype: a present kernel admits the configuration. The four dtypes here
    are the ones a full build carries for SM 103 / head_dim 64 (the SM103
    L0 suites run paged-context attention with a BF16 KV cache)."""
    metadata = _gate(head_dim=64, sm_version=103, features=ALL_FEATURES, kv_dtype=kv_dtype)
    assert metadata.use_paged_context_fmha


@pytest.mark.parametrize(
    "kv_dtype", [DataType.FP8, DataType.FLOAT, DataType.INT8], ids=["fp8", "float", "int8"]
)
def test_probe_absent_kv_dtypes_are_refused(kv_dtype):
    """A KV dtype the build has no kernel for (or that the lookup does not
    model at all -- it reports those absent) must be refused, with the error
    naming the combination, the page size that was queried and the remedy."""
    with pytest.raises(RuntimeError) as exc:
        _gate(
            head_dim=64,
            sm_version=103,
            features=ALL_FEATURES,
            kv_dtype=kv_dtype,
            kernel_exists=False,
        )
    message = str(exc.value)
    assert "64" in message and "103" in message  # the combination
    assert str(kv_dtype) in message or kv_dtype.name in message  # the KV dtype
    assert "32" in message  # the page size that was queried
    assert "--cuda_architectures" in message  # the partial-build cause
    assert "FlashInfer" in message  # the remedy must be actionable


def test_every_blocked_head_dim_needs_its_kernel():
    """With a per-window head_dim list, the kernel must be present for every
    blocked head_dim; one miss keeps the refusal."""
    metadata = _make_metadata(head_dim=[64, 72], features=ALL_FEATURES, kv_dtype=DataType.FP8)
    with mock.patch.dict(FallbackFmha.CONTEXT_FMHA_ABSENT_HEAD_DIMS, {103: (64, 72)}):
        with (
            mock.patch(_FALLBACK_SM_VERSION_TARGET, return_value=103),
            mock.patch.object(
                thop,
                "fused_context_fmha_kernel_exists",
                side_effect=lambda head_size, **kwargs: head_size == 64,
                create=True,
            ),
        ):
            with pytest.raises(RuntimeError, match="72"):
                FallbackFmha._validate_paged_context_fmha(metadata)


def test_blocklist_extension_on_another_sm_uses_the_probe():
    """A newly blocklisted cell on another SM needs no companion table: the
    probe decides there too, in both directions."""
    with mock.patch.dict(FallbackFmha.CONTEXT_FMHA_ABSENT_HEAD_DIMS, {90: (64,)}):
        with pytest.raises(RuntimeError, match="64"):
            _gate(
                head_dim=64,
                sm_version=90,
                features=ALL_FEATURES,
                kv_dtype=DataType.FP8,
                kernel_exists=False,
            )
        metadata = _gate(head_dim=64, sm_version=90, features=ALL_FEATURES, kv_dtype=DataType.FP8)
        assert metadata.use_paged_context_fmha


@pytest.mark.parametrize(
    "head_dim,kv_dtype",
    [(128, DataType.BF16), (576, DataType.FP8)],
    ids=["dense_128", "mla_576"],
)
def test_unblocked_combination_is_never_probed(head_dim, kv_dtype):
    """Off the blocklist, the gate must not consult the kernel lookup at
    all: the lookup is a fixed-convention diagnostic (dense causal
    Q_PAGED_KV, inferred Q/output precision) that does not model every
    configuration the op can run (MLA, cross attention), so probing
    unconditionally would refuse configurations the op serves correctly.
    head_dim 576 is the MLA KV-cache width (DeepSeek), which the lookup's
    dense-MHA convention does not model -- the probe must stay scoped to
    the blocklisted head dims even on the FP8-KV multi-variant path."""
    metadata = _make_metadata(head_dim=head_dim, features=ALL_FEATURES, kv_dtype=kv_dtype)
    with (
        mock.patch(_FALLBACK_SM_VERSION_TARGET, return_value=103),
        mock.patch.object(
            thop, "fused_context_fmha_kernel_exists", return_value=False, create=True
        ) as lookup,
    ):
        FallbackFmha._validate_paged_context_fmha(metadata)
    assert metadata.use_paged_context_fmha
    lookup.assert_not_called()


def test_manager_without_dtype_fails_closed():
    """A manager exposing no ``dtype`` cannot prove the kernel present; the
    blocked combination must stay refused rather than assume presence."""
    with pytest.raises(RuntimeError, match="64"):
        _gate(head_dim=64, sm_version=103, features=ALL_FEATURES)


def test_manager_without_tokens_per_block_fails_closed():
    """The native lookup needs the page size the engine will run with. A
    manager that does not expose one cannot be checked, so the blocked
    combination stays refused."""
    with pytest.raises(RuntimeError, match="64"):
        _gate(
            head_dim=64,
            sm_version=103,
            features=ALL_FEATURES,
            kv_dtype=DataType.FP8,
            tokens_per_block=_ABSENT,
        )


def test_gate_queries_the_native_kernel_lookup():
    """Inside a blocklisted cell the gate must ask the build what it
    contains, with the manager's KV dtype and the page size the engine will
    use. A 16-bit KV cache pairs with exactly one output precision (matched),
    so the gate asks exactly once."""
    metadata = _make_metadata(head_dim=64, features=ALL_FEATURES, kv_dtype=DataType.BF16)
    with (
        mock.patch(_FALLBACK_SM_VERSION_TARGET, return_value=103),
        mock.patch.object(
            thop, "fused_context_fmha_kernel_exists", return_value=True, create=True
        ) as lookup,
    ):
        FallbackFmha._validate_paged_context_fmha(metadata)
    assert metadata.use_paged_context_fmha
    lookup.assert_called_once_with(
        head_size=64,
        kv_cache_dtype=DataType.BF16,
        tokens_per_block=32,
        output_dtype=DataType.BF16,
    )


# For an FP8 KV cache the op quantizes Q to FP8 but keeps the output at the
# activation dtype (BF16/FP16) unless FP8 attention output is enabled, so any
# of these output precisions may be the one the model runs.
_FP8_KV_OUTPUT_VARIANTS = (DataType.BF16, DataType.HALF, DataType.FP8)


def _run_fp8_kv_gate(kernel_exists_stub):
    metadata = _make_metadata(head_dim=64, features=ALL_FEATURES, kv_dtype=DataType.FP8)
    with (
        mock.patch(_FALLBACK_SM_VERSION_TARGET, return_value=103),
        mock.patch.object(
            thop, "fused_context_fmha_kernel_exists", side_effect=kernel_exists_stub, create=True
        ) as lookup,
    ):
        FallbackFmha._validate_paged_context_fmha(metadata)
    return metadata, lookup


@pytest.mark.parametrize("present_output_dtype", _FP8_KV_OUTPUT_VARIANTS, ids=lambda d: d.name)
def test_fp8_kv_admits_any_present_output_variant(present_output_dtype):
    """With an FP8 KV cache, the activation dtype and the FP8-attention-output
    flag are not visible to this metadata-only check, so one present output
    variant admits the cell. In particular a build that carries only the
    activation-output cubin (BF16 model, no FP8 attention output) must not be
    refused just because the FP8-output cubin is absent."""
    metadata, _ = _run_fp8_kv_gate(lambda **kwargs: kwargs["output_dtype"] == present_output_dtype)
    assert metadata.use_paged_context_fmha


def test_fp8_kv_refused_only_after_every_output_variant_is_absent():
    """The FP8-KV refusal must be proven against every output precision the
    op can pair with the cache, not a single hardcoded one."""
    metadata = _make_metadata(head_dim=64, features=ALL_FEATURES, kv_dtype=DataType.FP8)
    with (
        mock.patch(_FALLBACK_SM_VERSION_TARGET, return_value=103),
        mock.patch.object(
            thop, "fused_context_fmha_kernel_exists", return_value=False, create=True
        ) as lookup,
    ):
        with pytest.raises(RuntimeError, match="64"):
            FallbackFmha._validate_paged_context_fmha(metadata)
    assert {call.kwargs["output_dtype"] for call in lookup.call_args_list} == set(
        _FP8_KV_OUTPUT_VARIANTS
    )


def test_nvfp4_kv_probes_fp8_output_only():
    """An NVFP4 KV cache is read by the FP8-output kernel set; the gate must
    not widen that probe to activation-dtype outputs."""
    metadata = _make_metadata(head_dim=64, features=ALL_FEATURES, kv_dtype=DataType.NVFP4)
    with (
        mock.patch(_FALLBACK_SM_VERSION_TARGET, return_value=103),
        mock.patch.object(
            thop, "fused_context_fmha_kernel_exists", return_value=True, create=True
        ) as lookup,
    ):
        FallbackFmha._validate_paged_context_fmha(metadata)
    assert metadata.use_paged_context_fmha
    lookup.assert_called_once_with(
        head_size=64,
        kv_cache_dtype=DataType.NVFP4,
        tokens_per_block=32,
        output_dtype=DataType.FP8,
    )


def test_missing_lookup_binding_fails_closed():
    """A build whose bindings predate the kernel lookup cannot verify kernel
    presence (and also predates the op-level refusal), so the blocked
    combination stays refused, with the error naming the missing binding
    rather than crashing on an attribute error."""
    metadata = _make_metadata(head_dim=64, features=ALL_FEATURES, kv_dtype=DataType.FP8)
    with (
        mock.patch(_FALLBACK_SM_VERSION_TARGET, return_value=103),
        mock.patch.object(thop, "fused_context_fmha_kernel_exists", new=None, create=True),
    ):
        with pytest.raises(RuntimeError, match="fused_context_fmha_kernel_exists"):
            FallbackFmha._validate_paged_context_fmha(metadata)


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
    """The live lookup must be able to say no: without that, the gate's
    probe inside a blocklisted cell could only ever admit. The lookup's
    present-case live behavior is covered by
    test_context_fmha_kernel_presence.py."""
    if not 100 <= get_sm_version() < 110:
        pytest.skip("the unsupported head-size case targets the SM100-family dispatcher")
    assert not thop.fused_context_fmha_kernel_exists(
        head_size=_UNSUPPORTED_HEAD_SIZE,
        kv_cache_dtype=DataType.BF16,
        tokens_per_block=_PAGED_CONTEXT_TOKENS_PER_BLOCK,
        output_dtype=DataType.BF16,
    )


def test_unknown_head_dim_is_not_guessed():
    """A manager that exposes no head_dim (or no manager at all) cannot be
    judged; the gate must not block it on a guess."""
    with mock.patch.object(TrtllmAttentionMetadata, "_post_init_with_buffers"):
        metadata = TrtllmAttentionMetadata(
            max_num_requests=4,
            max_num_tokens=1024,
            kv_cache_manager=None,
            runtime_features=ALL_FEATURES,
        )
    with mock.patch(_FALLBACK_SM_VERSION_TARGET, return_value=103):
        FallbackFmha._validate_paged_context_fmha(metadata)
    assert metadata.use_paged_context_fmha
>>>>>>> origin/main
