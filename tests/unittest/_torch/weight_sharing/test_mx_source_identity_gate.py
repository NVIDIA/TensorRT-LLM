# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise the real ModelExpress identity gate before RDMA mutation."""

import json
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock

import modelexpress.load_strategy as load_strategy
import pytest
from _source_identity_fakes import make_identity
from modelexpress.engines import trtllm
from modelexpress.load_strategy import rdma_strategy

from tensorrt_llm._torch.weight_sharing import (
    LLAMA_POST_TRANSFORM_LAYOUT_ABI_V1,
    SOURCE_IDENTITY_FORMAT_VERSION,
)


def _identity(*, transform_abi_id=LLAMA_POST_TRANSFORM_LAYOUT_ABI_V1):
    return replace(
        make_identity(model_name="meta-llama/Llama-3.1-8B-Instruct"),
        transform_abi_id=transform_abi_id,
    )


def _mx_modules():
    return trtllm, load_strategy, rdma_strategy


def test_real_adapter_serializes_authoritative_trt_identity():
    trtllm, _, _ = _mx_modules()
    identity = _identity()

    mx_identity = trtllm.build_mx_identity(
        identity,
        transform_protocol_version=1,
    )
    serialized = json.loads(mx_identity.extra_parameters["trtllm_source_identity"])

    assert serialized == {
        key: value for key, value in identity.to_dict().items() if key != "model_name"
    }
    assert serialized["format_version"] == SOURCE_IDENTITY_FORMAT_VERSION
    assert serialized["transform_abi_id"] == LLAMA_POST_TRANSFORM_LAYOUT_ABI_V1


@pytest.mark.parametrize(
    "incompatible_identity",
    [
        _identity(transform_abi_id="trtllm-llama-target-layout-v2"),
        replace(_identity(), format_version=SOURCE_IDENTITY_FORMAT_VERSION - 1),
        replace(_identity(), backend_fingerprint="different-backend"),
        replace(
            _identity(),
            artifact_identity=replace(_identity().artifact_identity, digest="1" * 64),
        ),
    ],
    ids=["transform-abi", "format", "backend", "artifact"],
)
def test_incompatible_identity_falls_back_before_receiver_mutation(
    incompatible_identity,
):
    trtllm, load_strategy, rdma_strategy = _mx_modules()
    target_identity = trtllm.build_mx_identity(
        _identity(),
        transform_protocol_version=1,
    )
    published_identity = trtllm.build_mx_identity(
        incompatible_identity,
        transform_protocol_version=1,
    )

    class _Client:
        def __init__(self):
            self.queried_identity = None

        def list_sources(self, *, identity, status_filter):
            self.queried_identity = identity
            instances = [MagicMock()] if identity == published_identity else []
            return SimpleNamespace(instances=instances)

    client = _Client()
    ctx = SimpleNamespace(
        mx_client=client,
        identity=target_identity,
        global_rank=0,
        worker_rank=0,
    )
    result = load_strategy.LoadResult(value=MagicMock(), model=MagicMock())

    with pytest.raises(load_strategy.StrategyFailed) as error:
        rdma_strategy.RdmaStrategy().load(result, ctx)

    assert error.value.mutated is False
    assert client.queried_identity == target_identity
    assert target_identity != published_identity
