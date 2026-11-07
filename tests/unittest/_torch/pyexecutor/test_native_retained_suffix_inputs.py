# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Matching-main CPU test of real request cursor -> scheduler -> decoder packer.

No GPU allocation or model execution. Only the cache capacity/resize seam is a
fake; the actual native-backed LlmRequest, connector window/cursor helpers,
ScheduledRequests and text DecoderInputPreparer packing methods execute.
Actual GPU bytes/transport/semantic E2E remain separate required gates.
"""

import hashlib
import json
import struct
import unittest
from types import MethodType
from types import SimpleNamespace as NS
from unittest.mock import patch

import pytest

from tensorrt_llm._torch.pyexecutor import native_retained_prefix as retained
from tensorrt_llm._torch.pyexecutor.engine.runners.decoder.inputs import (
    DecoderInputPreparer,
    ExtraInputsCollector,
    HostInputs,
)
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest, SamplingConfig
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm.disaggregated_params import DisaggregatedParams, DisaggScheduleStyle

pytestmark = pytest.mark.cpu_only


class ResizeOnlyCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.history_length = 0

    def resize(self, capacity, history_length):
        self.capacity = capacity
        self.history_length = history_length
        return True


class ActualSuffixPacking(unittest.TestCase):
    def exercise(self, prompt, scheduled_chunk, prefix, expected_end, last):
        tokens = [1000 + index for index in range(prompt)]
        request = LlmRequest(
            request_id=71,
            max_new_tokens=4,
            input_tokens=tokens,
            sampling_config=SamplingConfig(1),
            is_streaming=False,
        )
        request.context_chunk_size = scheduled_chunk
        request.py_seq_slot = 3
        params = DisaggregatedParams(
            disagg_request_id=91,
            ctx_info_endpoint="trusted-normal-forward-context-endpoint",
            schedule_style=DisaggScheduleStyle.CONTEXT_FIRST,
            retain_prefix=True,
        )
        request.py_disaggregated_params = params
        scheduled = ScheduledRequests()
        scheduled.append_context_request(request)
        was_last = request.is_last_context_chunk
        manager = NS(tokens_per_block=32, _kv_alloc_extra_tokens=0)
        manager._resize_for_served_prefix = MethodType(
            KVCacheManagerV2._resize_for_served_prefix, manager
        )
        cache = ResizeOnlyCache(scheduled_chunk)
        fitted = KVCacheManagerV2._reserve_connector_prefix(manager, request, cache, 0, prefix)
        self.assertEqual((prefix, expected_end - prefix), fitted)
        # Production only commits this cursor after verified native completion.
        KVCacheManagerV2._mark_connector_prefix_populated(manager, request, *fitted)
        scheduled.reset_context_requests()
        self.assertEqual(last, request.is_last_context_chunk)
        self.assertEqual([request] if last else [], scheduled.context_requests_last_chunk)
        self.assertEqual([] if last else [request], scheduled.context_requests_chunking)
        self.assertEqual(prefix, cache.history_length)
        self.assertGreaterEqual(cache.capacity, expected_end)
        self.assertEqual(prefix, request.context_current_position)
        self.assertTrue(request.is_first_context_chunk)
        preparer = object.__new__(DecoderInputPreparer)
        preparer._config = NS(mm_encoder_cache_enabled=False)
        preparer._mm_item_scheduler = None
        preparer.model = NS()
        preparer.mapping = NS(rank=0)
        request.py_retained_loaded = True
        request.py_retained_imported_tokens = prefix
        request.py_retained_import_receipt = retained._RetainedImportReceipt(
            "a" * 64, 123, "b" * 32, "c" * 32, prefix
        )
        host = HostInputs(
            use_mrope=False,
            mrope_dummy_seq_slot=8,
            extra_inputs=ExtraInputsCollector(),
            context_prompt_lookahead=None,
        )
        # This executes the actual text multimodal no-content branch too.
        with patch.object(retained.logger, "info") as logged:
            preparer._pack_context_requests(host, scheduled)
        receipt = json.loads(logged.call_args.args[0].removeprefix("retained_prefix "))
        self.assertEqual("model_input", receipt["event"])
        self.assertEqual(123, receipt["native_claim_id"])
        self.assertEqual(71, receipt["request_id"])
        self.assertEqual(prefix, receipt["begin"])
        self.assertEqual(expected_end, receipt["end"])
        self.assertEqual(expected_end - prefix, receipt["actual_token_count"])
        self.assertEqual(
            hashlib.sha256(
                b"".join(struct.pack("<q", token) for token in tokens[prefix:expected_end])
            ).hexdigest(),
            receipt["token_ids_sha256"],
        )
        self.assertEqual(
            hashlib.sha256(
                b"".join(struct.pack("<q", position) for position in range(prefix, expected_end))
            ).hexdigest(),
            receipt["position_ids_sha256"],
        )
        self.assertEqual(tokens[prefix:expected_end], host.lists.input_ids)
        self.assertEqual(list(range(prefix, expected_end)), host.lists.position_ids)
        self.assertEqual([expected_end - prefix], host.lists.sequence_lengths)
        self.assertEqual([expected_end - prefix], host.lists.prompt_lengths)
        self.assertEqual([prefix], host.lists.num_cached_tokens_per_seq)
        self.assertEqual(expected_end - prefix, host.num_ctx_tokens)
        self.assertEqual(prefix, request.cached_tokens)
        self.assertEqual(3, request.py_batch_idx)
        # Ordinary P->D pairing metadata survives the cursor/packing transition.
        self.assertIs(params, request.py_disaggregated_params)
        self.assertEqual(91, params.disagg_request_id)
        self.assertEqual("trusted-normal-forward-context-endpoint", params.ctx_info_endpoint)
        return was_last

    def test_block_edge_leaves_exactly_one_suffix_token(self):
        self.assertFalse(
            self.exercise(prompt=33, scheduled_chunk=32, prefix=32, expected_end=33, last=True)
        )

    def test_prefix_past_previous_scheduler_chunk_reclassifies_as_final(self):
        self.assertFalse(
            self.exercise(prompt=96, scheduled_chunk=32, prefix=64, expected_end=96, last=True)
        )

    def test_prefix_within_chunk_preserves_nonfinal_aligned_end(self):
        self.assertFalse(
            self.exercise(prompt=129, scheduled_chunk=96, prefix=32, expected_end=96, last=False)
        )

    def test_prefix_beyond_chunk_preserves_remaining_chunk_budget(self):
        self.assertFalse(
            self.exercise(prompt=161, scheduled_chunk=32, prefix=64, expected_end=96, last=False)
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
