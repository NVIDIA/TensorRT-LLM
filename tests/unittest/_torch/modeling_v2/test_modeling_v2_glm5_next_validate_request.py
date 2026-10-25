# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The GLM-5.3-Flash target's admission check fails a request its step cannot serve.

With MTP the worker's head samples without materializing the logits, so a logits
post-processor (a thinking budget, a request's own processor) has nothing to edit;
without MTP the forward computes each sequence's last-token logits only, not the
prompt's. Raised inside the step, either would fail every active request. The check
reads only host-side request fields, so these run on fake objects: no GPU or engine.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

__extra_import_path__ = [".."]  # noqa: F841 -- repo's file-scoped import hook

from tensorrt_llm._torch._experimental.modeling_v2.models.glm5_next.glm_5_3_flash_nvfp4__sm_10x__tp4 import (  # noqa: E501
    modeling as M,
)

pytestmark = pytest.mark.cpu_only

Shell = M.ModelingV2Glm53FlashNvfp4Sm10xTp4


def _admit(request, *, mtp: bool) -> None:
    Shell.validate_request(SimpleNamespace(spec_worker=object() if mtp else None), request)


def _request(post_processors=None, context_logits=False):
    return SimpleNamespace(
        py_logits_post_processors=post_processors, py_return_context_logits=context_logits
    )


def _processor(req_id, logits, token_ids, stream_ptr, client_id):
    pass


@pytest.mark.parametrize("mtp", [True, False])
def test_plain_requests_are_admitted(mtp):
    _admit(_request(), mtp=mtp)
    _admit(_request(post_processors=[]), mtp=mtp)


def test_mtp_refuses_logits_post_processors():
    with pytest.raises(ValueError, match="post-processors"):
        _admit(_request(post_processors=[_processor]), mtp=True)


def test_without_mtp_post_processors_run_on_the_logits():
    _admit(_request(post_processors=[_processor]), mtp=False)


def test_without_mtp_context_logits_are_refused():
    with pytest.raises(ValueError, match="return_context_logits"):
        _admit(_request(context_logits=True), mtp=False)


def test_with_mtp_context_logits_are_left_to_the_sampler():
    # SpecSampler.validate_request refuses them before the model's check runs.
    _admit(_request(context_logits=True), mtp=True)
