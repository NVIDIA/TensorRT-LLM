# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests for ``TRTLLM_FI_DECODE_TENSOR_CORES``.

``_use_tensor_cores`` decides whether a pure-decode plan gets a tensor-core
wrapper, and such a wrapper does not run FlashInfer's split-K decode kernel: it
builds a synthetic ``qo_indptr`` and dispatches the paged-prefill kernel. The
GQA-ratio arm fires for any model with at least 4 query heads per KV head,
which is most of them, so those models run the prefill kernel on every layer at
pure decode. That arm is FlashInfer's own perf heuristic rather than a
correctness requirement, and the knob exists to measure the alternative.

These tests pin what the knob may and may not touch, because the risk in a
switch like this is not that it fails to work -- it is that it silently changes
something else: the FP8 arm or the speculative-decode path, neither of which
has a split-K decode kernel to fall back to, or a model whose operator never
set the variable.

Pure dispatch logic, so no GPU is needed.
"""

import os
from unittest import mock

import pytest
import torch

from tensorrt_llm._torch.attention.backends.flashinfer import (
    FI_DECODE_TENSOR_CORES_ENV,
    FlashInferAttentionMetadata,
    PlanParams,
    decode_tensor_cores_override,
)
from tensorrt_llm.functional import AttentionMaskType


def plan(
    num_heads: int,
    num_kv_heads: int,
    kv_dtype=torch.bfloat16,
    q_len_per_req: int = 1,
) -> PlanParams:
    """A decode PlanParams; only the fields the decision reads vary."""
    return PlanParams(
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=64,
        q_dtype=torch.bfloat16,
        kv_dtype=kv_dtype,
        attention_mask_type=AttentionMaskType.causal,
        q_len_per_req=q_len_per_req,
    )


def use_tensor_cores(
    plan_params: PlanParams,
    *,
    flashinfer_backend: str = "fa2",
    use_graph_tensor_cores: bool = False,
) -> bool:
    """``_use_tensor_cores`` reads only its arguments, never ``self``."""
    return FlashInferAttentionMetadata._use_tensor_cores(
        None, plan_params, flashinfer_backend, use_graph_tensor_cores
    )


@pytest.fixture(autouse=True)
def _unset_env():
    """Never inherit the operator's setting: it would invert every assertion."""
    with mock.patch.dict(os.environ, {}, clear=False):
        os.environ.pop(FI_DECODE_TENSOR_CORES_ENV, None)
        yield


@pytest.mark.parametrize(
    "num_heads,num_kv_heads,expected",
    [
        (128, 16, True),  # ratio 8: a high-GQA decode shape
        (32, 8, True),  # ratio 4: exactly at the threshold
        (24, 8, False),  # ratio 3: below it
        (16, 16, False),  # MHA
    ],
)
@pytest.mark.parametrize("unset_as", [None, "", "auto", "AUTO"])
def test_default_is_unchanged(num_heads, num_kv_heads, expected, unset_as):
    """Unset, empty and ``auto`` are all exactly the pre-existing heuristic."""
    if unset_as is not None:
        os.environ[FI_DECODE_TENSOR_CORES_ENV] = unset_as
    assert decode_tensor_cores_override() is None
    assert use_tensor_cores(plan(num_heads, num_kv_heads)) is expected


@pytest.mark.parametrize(
    "value,num_heads,num_kv_heads,expected",
    [
        # "0" forces the split-K decode kernel, even on a high-GQA decode shape
        # the heuristic would have given tensor cores.
        ("0", 128, 16, False),
        (" 0 ", 128, 16, False),  # surrounding whitespace is tolerated
        # "1" is symmetric: it forces tensor cores below the ratio and keeps
        # them above it.
        ("1", 24, 8, True),
        ("1", 128, 16, True),
    ],
)
def test_an_explicit_override_selects_that_kernel(value, num_heads, num_kv_heads, expected):
    os.environ[FI_DECODE_TENSOR_CORES_ENV] = value
    assert use_tensor_cores(plan(num_heads, num_kv_heads)) is expected


# Only 0/1/auto are accepted; the old boolean spellings now raise like any typo.
@pytest.mark.parametrize("bad", ["yes", "2", "no", "disable", "None", "false", "off", "true", "on"])
def test_an_unrecognised_value_raises(bad):
    """Loud, not ignored. A typo that silently kept the heuristic would make
    both legs of the A/B measure the same thing and report a 0% difference."""
    os.environ[FI_DECODE_TENSOR_CORES_ENV] = bad
    with pytest.raises(ValueError, match=FI_DECODE_TENSOR_CORES_ENV):
        decode_tensor_cores_override()


@pytest.mark.parametrize("kv_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("value", ["0", "1"])
def test_fp8_kv_cache_always_keeps_tensor_cores(kv_dtype, value):
    """The FP8 arm is not part of the experiment and must not move.

    There is no non-tensor-core decode kernel for an FP8 KV cache, so honouring
    ``0`` here would not select a different kernel -- it would select a path
    that does not exist.
    """
    os.environ[FI_DECODE_TENSOR_CORES_ENV] = value
    assert use_tensor_cores(plan(128, 16, kv_dtype=kv_dtype)) is True
    # Also for a ratio that would not have asked for tensor cores anyway.
    assert use_tensor_cores(plan(16, 16, kv_dtype=kv_dtype)) is True


@pytest.mark.parametrize("q_len_per_req", [2, 4])
def test_multi_token_query_ignores_an_off_override(q_len_per_req):
    """Speculative decode has no split-K decode kernel either.

    This is the case the knob has to leave alone: with ``q_len_per_req > 1``
    the heuristic stands whatever the variable says.
    """
    os.environ[FI_DECODE_TENSOR_CORES_ENV] = "0"
    assert use_tensor_cores(plan(128, 16, q_len_per_req=q_len_per_req)) is True
    # And a shape the heuristic already declined stays declined -- the
    # multi-token guard restores the heuristic, it does not force cores on.
    assert use_tensor_cores(plan(24, 8, q_len_per_req=q_len_per_req)) is False


def test_decision_is_read_per_call_not_cached_at_import():
    """The A/B sets the variable per server launch, so an import-time read
    would give both legs the same answer and quietly measure nothing."""
    os.environ.pop(FI_DECODE_TENSOR_CORES_ENV, None)
    assert use_tensor_cores(plan(128, 16)) is True
    os.environ[FI_DECODE_TENSOR_CORES_ENV] = "0"
    assert use_tensor_cores(plan(128, 16)) is False
    os.environ.pop(FI_DECODE_TENSOR_CORES_ENV, None)
    assert use_tensor_cores(plan(128, 16)) is True


def _decision_warnings(plan_params, *, use_graph_tensor_cores, backend) -> list:
    """Run the tensor-core decision and return any forced-on warnings.

    The warning is now emitted inside ``_use_tensor_cores`` next to the choice
    it explains, so it is driven through the decision rather than called on its
    own.
    """
    with mock.patch(
        "tensorrt_llm._torch.attention.backends.flashinfer.logger.warning_once"
    ) as warn:
        use_tensor_cores(
            plan_params, flashinfer_backend=backend, use_graph_tensor_cores=use_graph_tensor_cores
        )
    return warn.call_args_list


# plan(24, 8): ratio 3, so the heuristic declines tensor cores and an off
# override is what the decision honours -- the shape whose override a later
# requirement can outrank.
_OVERRIDE_HONOURED = (24, 8)


@pytest.mark.parametrize(
    "use_graph_tensor_cores,backend,reason",
    [
        (True, "fa2", "CUDA graph with head_dim > 128"),
        (False, "trtllm-gen", "the trtllm-gen backend"),
    ],
)
def test_a_forced_override_warns_with_the_forcing_reason(use_graph_tensor_cores, backend, reason):
    """An off override that a requirement outranks must say so, and say why.

    The override names an A/B leg; a leg that silently runs as the other leg
    records "no difference", so removal or misrouting of this warning turns a
    voided measurement into a finding.
    """
    os.environ[FI_DECODE_TENSOR_CORES_ENV] = "0"
    calls = _decision_warnings(
        plan(*_OVERRIDE_HONOURED), use_graph_tensor_cores=use_graph_tensor_cores, backend=backend
    )
    assert len(calls) == 1, f"expected exactly one warning; got {calls}"
    message = calls[0].args[0]
    assert f"{FI_DECODE_TENSOR_CORES_ENV}=0 ignored" in message
    assert reason in message


@pytest.mark.parametrize(
    "env,plan_params,use_graph_tensor_cores,backend",
    [
        # No off override -> nothing was outranked, whatever the wrapper does.
        (None, plan(*_OVERRIDE_HONOURED), True, "fa2"),
        ("auto", plan(*_OVERRIDE_HONOURED), False, "trtllm-gen"),
        # The heuristic already forces tensor cores on (fp8 KV has no split-K
        # decode kernel), so the off override changed nothing for this plan.
        ("0", plan(128, 16, kv_dtype=torch.float8_e4m3fn), True, "fa2"),
        # Nothing forces tensor cores -> the off override is honoured silently.
        ("0", plan(*_OVERRIDE_HONOURED), False, "fa2"),
    ],
)
def test_no_warning_when_nothing_is_forced(env, plan_params, use_graph_tensor_cores, backend):
    """The warning is for one case only: an off override a requirement beat."""
    if env is not None:
        os.environ[FI_DECODE_TENSOR_CORES_ENV] = env
    calls = _decision_warnings(
        plan_params, use_graph_tensor_cores=use_graph_tensor_cores, backend=backend
    )
    assert calls == [], f"expected no warning; got {calls}"
