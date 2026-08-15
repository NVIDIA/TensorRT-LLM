# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from tensorrt_llm.serve.cluster_storage import jsonify

pytestmark = pytest.mark.cpu_only


@pytest.mark.asyncio
@pytest.mark.parametrize("result", ["", {}])
async def test_jsonify_returns_200_for_valid_empty_results(result):
    async def handler():
        return result

    response = await jsonify(handler)()

    assert response.status_code == 200
    assert json.loads(response.body)["result"] == result


@pytest.mark.asyncio
@pytest.mark.parametrize("result", [False, None])
async def test_jsonify_keeps_failure_status_for_false_and_none(result):
    async def handler():
        return result

    response = await jsonify(handler)()

    assert response.status_code == 400
