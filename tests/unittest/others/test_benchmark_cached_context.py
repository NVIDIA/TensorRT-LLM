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
"""Unit tests for the cached-context (--random-cached-len) benchmark mode.

Run tests with:
    python -m pytest tests/unittest/others/test_benchmark_cached_context.py -v
"""

import asyncio
import json
import unittest
from unittest.mock import patch

import aiohttp
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from tensorrt_llm.serve.scripts import benchmark_serving
from tensorrt_llm.serve.scripts.backend_request_func import (
    RequestFuncInput,
    RequestFuncOutput,
    async_request_openai_completions,
)
from tensorrt_llm.serve.scripts.benchmark_dataset import RandomDataset, SampleRequest

pytestmark = pytest.mark.cpu_only


class _Tokenizer:
    vocab_size = 1000


def _sample_conversations(**overrides):
    kwargs = dict(
        tokenizer=_Tokenizer(),
        num_requests=24,
        num_conversations=4,
        cached_len=500,
        input_len=20,
        output_len=3,
    )
    kwargs.update(overrides)
    return RandomDataset(
        sample_from_sharegpt=False, return_text=False, random_seed=0
    ).sample_conversations(**kwargs)


class TestRandomConversations(unittest.TestCase):
    def test_turns_extend_the_previous_prompt(self):
        conversations = _sample_conversations()
        self.assertEqual(conversations.num_conversations, 4)
        self.assertEqual(len(conversations.requests), 24)
        self.assertEqual(len(set(conversations.conversation_ids)), 4)

        cached_lens = []
        histories = set()
        for conversation in range(4):
            prompt = json.loads(conversations.priming_prompt_json(conversation))
            # 500 - (6 turns - 1) / 2 * 20 new tokens per turn.
            self.assertEqual(len(prompt), 450)
            histories.add(tuple(prompt))
            num_turns = 0
            while (turn := conversations.next_turn(conversation)) is not None:
                index, prompt_json = turn
                next_prompt = json.loads(prompt_json)
                self.assertEqual(next_prompt[: len(prompt)], prompt)
                self.assertEqual(len(next_prompt), len(prompt) + 20)
                request = conversations.requests[index]
                self.assertEqual(request.prompt_len, len(next_prompt))
                self.assertEqual(request.expected_output_len, 3)
                self.assertEqual(
                    request.conversation_id, conversations.conversation_ids[conversation]
                )
                cached_lens.append(len(prompt))
                prompt = next_prompt
                num_turns += 1
            self.assertEqual(num_turns, 6)
            with self.assertRaises(RuntimeError):
                conversations.priming_prompt_json(conversation)
        self.assertEqual(len(histories), 4)
        self.assertEqual(sum(cached_lens) / len(cached_lens), 500)

    def test_uneven_turns(self):
        conversations = _sample_conversations(num_requests=10)
        turns = [0] * 4
        for conversation in range(4):
            while conversations.next_turn(conversation) is not None:
                turns[conversation] += 1
        self.assertEqual(turns, [3, 3, 2, 2])

    def test_invalid_shapes(self):
        for overrides in (
            dict(num_conversations=0),
            dict(num_conversations=25),
            dict(cached_len=50),
            dict(range_ratio=1.0),
        ):
            with self.subTest(**overrides), self.assertRaises(ValueError):
                _sample_conversations(**overrides)

    def test_lengths_follow_seed_and_tokens_are_fresh(self):
        first = _sample_conversations(range_ratio=0.5)
        second = _sample_conversations(range_ratio=0.5)
        self.assertEqual(
            [r.prompt_len for r in first.requests], [r.prompt_len for r in second.requests]
        )
        self.assertEqual(
            [r.expected_output_len for r in first.requests],
            [r.expected_output_len for r in second.requests],
        )
        self.assertNotEqual(first.priming_prompt_json(0), second.priming_prompt_json(0))
        self.assertNotEqual(first.conversation_ids, second.conversation_ids)


class TestCachedTokenMetrics(unittest.TestCase):
    def _metrics(self, cached_tokens):
        input_requests = [
            SampleRequest(prompt=None, prompt_len=1000, expected_output_len=1)
            for _ in cached_tokens
        ]
        outputs = [
            RequestFuncOutput(
                success=True, latency=0.5, ttft=0.5, output_tokens=1, cached_tokens=cached
            )
            for cached in cached_tokens
        ]
        metrics, _ = benchmark_serving.calculate_metrics(
            input_requests=input_requests,
            outputs=outputs,
            dur_s=1.0,
            tokenizer=None,
            selected_percentile_metrics=["ttft"],
            selected_percentiles=[50.0],
            goodput_config_dict={},
        )
        return metrics

    def test_cache_hit_ratio(self):
        metrics = self._metrics([900, 980])
        self.assertEqual(metrics.total_cached_tokens, 1880)
        self.assertAlmostEqual(metrics.cache_hit_ratio, 0.94)
        self.assertEqual(metrics.mean_cached_tokens, 940)
        self.assertEqual(metrics.mean_computed_tokens, 60)

    def test_not_reported(self):
        metrics = self._metrics([None, None])
        self.assertIsNone(metrics.total_cached_tokens)
        self.assertIsNone(metrics.cache_hit_ratio)


class TestCompletionsRequest(unittest.TestCase):
    def test_prompt_json_and_cached_tokens(self):
        received = []

        async def completions(request):
            body = await request.json()
            received.append(body)
            usage = {
                "prompt_tokens": len(body["prompt"]),
                "completion_tokens": 1,
                "total_tokens": len(body["prompt"]) + 1,
                "prompt_tokens_details": {"cached_tokens": len(body["prompt"]) - 2},
            }
            if not body["stream"]:
                return web.json_response({"choices": [{"text": "a"}], "usage": usage})
            response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
            await response.prepare(request)
            for chunk in ({"choices": [{"text": "a"}]}, {"choices": [], "usage": usage}):
                await response.write(b"data: " + json.dumps(chunk).encode() + b"\n\n")
            await response.write(b"data: [DONE]\n\n")
            return response

        async def run(streaming):
            app = web.Application()
            app.router.add_post("/v1/completions", completions)
            async with TestServer(app) as server, aiohttp.ClientSession() as session:
                return await async_request_openai_completions(
                    RequestFuncInput(
                        prompt=None,
                        prompt_json=b"[5, 6, 7, 8, 9]",
                        api_url=str(server.make_url("/v1/completions")),
                        prompt_len=5,
                        output_len=1,
                        model="model",
                        extra_body={"conversation_params": {"conversation_id": "c0"}},
                    ),
                    streaming=streaming,
                    session=session,
                )

        for streaming in (True, False):
            with self.subTest(streaming=streaming):
                output = asyncio.run(run(streaming))
                self.assertTrue(output.success, output.error)
                self.assertEqual(output.cached_tokens, 3)
                body = received[-1]
                self.assertEqual(body["prompt"], [5, 6, 7, 8, 9])
                self.assertEqual(body["max_tokens"], 1)
                self.assertEqual(body["conversation_params"], {"conversation_id": "c0"})


class TestConversationScheduling(unittest.TestCase):
    def _run(self, num_conversations, max_concurrency):
        conversations = _sample_conversations(num_conversations=num_conversations)
        last_prompt = {}
        in_flight = set()
        max_in_flight = 0
        calls = []

        async def fake_request(request_func_input, streaming=True, pbar=None, session=None):
            nonlocal max_in_flight
            conversation_id = request_func_input.extra_body["conversation_params"][
                "conversation_id"
            ]
            prompt = json.loads(request_func_input.prompt_json)
            self.assertNotIn(conversation_id, in_flight)
            previous = last_prompt.get(conversation_id, [])
            self.assertEqual(prompt[: len(previous)], previous)
            in_flight.add(conversation_id)
            max_in_flight = max(max_in_flight, len(in_flight))
            await asyncio.sleep(0.001)
            in_flight.discard(conversation_id)
            last_prompt[conversation_id] = prompt
            calls.append(conversation_id)
            if pbar:
                pbar.update(1)
            return RequestFuncOutput(
                success=True,
                latency=0.01,
                ttft=0.01,
                output_tokens=1,
                prompt_len=len(prompt),
                cached_tokens=len(previous),
            )

        async def no_energy_metrics(base_url):
            return None

        with (
            patch.dict(benchmark_serving.ASYNC_REQUEST_FUNCS, {"fake": fake_request}),
            patch.object(benchmark_serving, "fetch_energy_metrics", no_energy_metrics),
        ):
            result = asyncio.run(
                benchmark_serving.benchmark(
                    backend="fake",
                    api_url="http://localhost/v1/completions",
                    base_url="http://localhost",
                    model_id="model",
                    model_name=None,
                    tokenizer=None,
                    input_requests=conversations.requests,
                    logprobs=None,
                    request_rate=float("inf"),
                    burstiness=1.0,
                    disable_tqdm=True,
                    profile=False,
                    selected_percentile_metrics=["ttft"],
                    selected_percentiles=[50.0],
                    ignore_eos=True,
                    goodput_config_dict={},
                    max_concurrency=max_concurrency,
                    lora_modules=None,
                    extra_body={"temperature": 0.0},
                    streaming=True,
                    no_test_input=True,
                    conversations=conversations,
                )
            )
        # Every conversation is primed before its first measured turn.
        self.assertEqual(sorted(calls[:num_conversations]), sorted(conversations.conversation_ids))
        self.assertEqual(len(calls), num_conversations + 24)
        self.assertLessEqual(max_in_flight, max_concurrency)
        return result

    def test_one_conversation_per_worker(self):
        result = self._run(num_conversations=4, max_concurrency=4)
        self.assertEqual(result["completed"], 24)
        self.assertEqual(result["num_conversations"], 4)
        self.assertEqual(result["turns_per_conversation"], 6)
        self.assertEqual(result["history_len"], 450)
        self.assertEqual(result["priming_cached_tokens"], 0)
        self.assertEqual(result["mean_cached_tokens"], 500)
        self.assertEqual(result["mean_computed_tokens"], 20)

    def test_more_conversations_than_workers(self):
        result = self._run(num_conversations=8, max_concurrency=3)
        self.assertEqual(result["completed"], 24)
        self.assertEqual(result["turns_per_conversation"], 3)
        self.assertEqual(result["mean_computed_tokens"], 20)


if __name__ == "__main__":
    unittest.main()
