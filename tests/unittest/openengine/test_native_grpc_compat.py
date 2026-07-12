# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from tensorrt_llm.grpc.grpc_request_manager import GrpcRequestManager


class _Result:
    def __init__(self) -> None:
        self.aborted = False

    def abort(self) -> None:
        self.aborted = True

    def __aiter__(self):
        return self

    async def __anext__(self):
        raise StopAsyncIteration


class _Llm:
    def __init__(self) -> None:
        self.last_result = None

    def generate_async(self, *args, **kwargs):
        del args, kwargs
        self.last_result = _Result()
        return self.last_result


@pytest.mark.asyncio
async def test_native_grpc_duplicate_aborts_rejected_engine_result() -> None:
    llm = _Llm()
    manager = GrpcRequestManager(llm)
    manager._request_tracker.admit("duplicate", _Result())

    with pytest.raises(ValueError, match="already active"):
        _ = [
            result
            async for result in manager.generate(
                request_id="duplicate",
                prompt_token_ids=[1],
                sampling_params=object(),
            )
        ]

    assert llm.last_result.aborted
