# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import asyncio
from unittest import mock

import tensorrt_llm.serve.responses_utils as responses_utils


async def _empty_stream():
    return
    yield  # pragma: no cover - makes this an async generator


def test_process_streaming_events_handles_empty_stream() -> None:
    # An empty/aborted stream must not raise UnboundLocalError on final_res.
    async def run():
        processor = mock.MagicMock()
        processor.get_initial_responses.return_value = []
        processor.get_final_response = mock.AsyncMock()
        with mock.patch.object(
            responses_utils, "ResponsesStreamingProcessor", return_value=processor
        ):
            events = [
                event
                async for event in responses_utils.process_streaming_events(
                    _empty_stream(),
                    request=None,
                    sampling_params=None,
                    model_name="m",
                    conversation_store=None,
                )
            ]
        assert events == []
        # No output => no final response is built.
        processor.get_final_response.assert_not_called()

    asyncio.run(run())
