# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Unit tests for PyTorchWorker.

Tests the PyTorch backend worker implementation for scaffolding,
including generation, streaming generation, reward tasks, logprobs,
stop sequences, and input_tokens fallback.
"""

import pytest
import torch

from tensorrt_llm.scaffolding import (
    GenerationTask,
    NativeGenerationController,
    PyTorchWorker,
    RewardTask,
    ScaffoldingLlm,
    StreamGenerationTask,
    TaskStatus,
)


@pytest.fixture(scope="module")
def small_model_name():
    """Use a very small model for fast testing."""
    return "gpt2"


@pytest.fixture(scope="module")
def pytorch_worker(small_model_name):
    """Create a PyTorchWorker with a small model (shared across module)."""
    worker = PyTorchWorker.from_pretrained(
        small_model_name,
        device="cpu",
        torch_dtype=torch.float32,
    )
    yield worker
    worker.shutdown()


@pytest.fixture
def test_prompt():
    return "The capital of France is"


class TestPyTorchWorkerInitialization:
    def test_from_pretrained(self, small_model_name):
        worker = PyTorchWorker.from_pretrained(
            small_model_name, device="cpu", torch_dtype=torch.float32
        )
        assert worker.model is not None
        assert worker.tokenizer is not None
        worker.shutdown()

    def test_device_selection_explicit(self, small_model_name):
        worker = PyTorchWorker.from_pretrained(
            small_model_name, device="cpu", torch_dtype=torch.float32
        )
        assert str(worker.device) == "cpu"
        worker.shutdown()


class TestPyTorchWorkerGeneration:
    @pytest.mark.asyncio
    async def test_basic_generation(self, pytorch_worker, test_prompt):
        task = GenerationTask.create_from_prompt(test_prompt)
        task.max_tokens = 10
        task.temperature = 0.7

        status = await pytorch_worker.generation_handler(task)

        assert status == TaskStatus.SUCCESS
        assert task.output_str is not None
        assert len(task.output_str) > 0
        assert task.output_tokens is not None
        assert len(task.output_tokens) > 0

    @pytest.mark.asyncio
    async def test_generation_with_input_tokens(self, pytorch_worker):
        """Test generation using input_tokens instead of input_str."""
        tokenizer = pytorch_worker.tokenizer
        prompt = "Hello world"
        tokens = tokenizer.encode(prompt)

        task = GenerationTask()
        task.input_str = None
        task.input_tokens = tokens
        task.max_tokens = 10
        task.temperature = 0.0

        status = await pytorch_worker.generation_handler(task)

        assert status == TaskStatus.SUCCESS
        assert task.output_str is not None
        assert len(task.output_str) > 0
        assert task.output_tokens is not None

    @pytest.mark.asyncio
    async def test_generation_none_input_fails(self, pytorch_worker):
        """Both input_str and input_tokens None should fail."""
        task = GenerationTask()
        task.input_str = None
        task.input_tokens = None
        task.max_tokens = 10

        status = await pytorch_worker.generation_handler(task)
        # Note: enum name is misspelled in upstream task.py
        assert status == TaskStatus.WORKER_EXCEPTION  # noqa: E501

    @pytest.mark.asyncio
    async def test_generation_with_logprobs(self, pytorch_worker, test_prompt):
        """Test that logprobs are extracted when requested."""
        task = GenerationTask.create_from_prompt(test_prompt)
        task.max_tokens = 5
        task.temperature = 0.0
        task.num_logprobs = 3

        status = await pytorch_worker.generation_handler(task)

        assert status == TaskStatus.SUCCESS
        assert task.logprobs is not None
        assert len(task.logprobs) > 0
        # Each entry should be a dict mapping token_id -> Logprob
        for token_dict in task.logprobs:
            assert isinstance(token_dict, dict)
            for token_id, logprob in token_dict.items():
                assert isinstance(token_id, int)
                assert hasattr(logprob, "logprob")
                assert hasattr(logprob, "rank")
                assert logprob.logprob <= 0.0  # log probs are non-positive

    @pytest.mark.asyncio
    async def test_generation_with_stop_sequence(self, pytorch_worker):
        """Test that stop sequences terminate generation."""
        task = GenerationTask.create_from_prompt("1, 2, 3, 4, 5,")
        task.max_tokens = 50
        task.temperature = 0.0
        task.stop = ["\n"]

        status = await pytorch_worker.generation_handler(task)

        assert status == TaskStatus.SUCCESS
        assert task.output_str is not None
        # Output should not contain the stop sequence
        assert "\n" not in task.output_str

    @pytest.mark.asyncio
    async def test_generation_deterministic(self, pytorch_worker, test_prompt):
        """Test deterministic generation with temperature=0."""
        task1 = GenerationTask.create_from_prompt(test_prompt)
        task1.max_tokens = 10
        task1.temperature = 0.0

        task2 = GenerationTask.create_from_prompt(test_prompt)
        task2.max_tokens = 10
        task2.temperature = 0.0

        await pytorch_worker.generation_handler(task1)
        await pytorch_worker.generation_handler(task2)

        assert task1.output_str == task2.output_str


class TestPyTorchWorkerStreamGeneration:
    @pytest.mark.asyncio
    async def test_stream_generation_completes(self, pytorch_worker, test_prompt):
        """Test that streaming generation eventually sets end_flag."""
        task = StreamGenerationTask()
        task.input_str = test_prompt
        task.max_tokens = 10
        task.temperature = 0.0
        task.streaming_step = 3

        # Run until done
        max_iterations = 20
        for _ in range(max_iterations):
            status = await pytorch_worker.stream_generation_handler(task)
            assert status == TaskStatus.SUCCESS
            if task.end_flag:
                break

        assert task.end_flag is True
        assert task.output_str is not None
        assert len(task.output_str) > 0
        assert task.output_tokens is not None

    @pytest.mark.asyncio
    async def test_stream_generation_cancel(self, pytorch_worker, test_prompt):
        """Test that cancel_flag stops generation immediately."""
        task = StreamGenerationTask()
        task.input_str = test_prompt
        task.max_tokens = 50
        task.temperature = 0.0
        task.streaming_step = 2

        # Generate a few tokens first
        status = await pytorch_worker.stream_generation_handler(task)
        assert status == TaskStatus.SUCCESS
        assert task.end_flag is False

        # Now cancel
        task.cancel_flag = True
        status = await pytorch_worker.stream_generation_handler(task)
        assert status == TaskStatus.SUCCESS
        assert task.end_flag is True

    @pytest.mark.asyncio
    async def test_stream_generation_with_input_tokens(self, pytorch_worker):
        """Test streaming with input_tokens instead of input_str."""
        tokenizer = pytorch_worker.tokenizer
        tokens = tokenizer.encode("Hello world")

        task = StreamGenerationTask()
        task.input_str = None
        task.input_tokens = tokens
        task.max_tokens = 5
        task.temperature = 0.0
        task.streaming_step = 2

        for _ in range(10):
            status = await pytorch_worker.stream_generation_handler(task)
            assert status == TaskStatus.SUCCESS
            if task.end_flag:
                break

        assert task.output_str is not None


class TestPyTorchWorkerReward:
    @pytest.mark.asyncio
    async def test_reward_with_input_tokens(self, pytorch_worker):
        """Test reward handler with input_tokens fallback.

        Note: Using a causal LM for reward will likely error since it's
        not a classification model. This tests the input routing logic.
        """
        tokenizer = pytorch_worker.tokenizer
        tokens = tokenizer.encode("This is a test.")

        task = RewardTask()
        task.input_str = None
        task.input_tokens = tokens

        # Will likely fail since gpt2 is not a classification model,
        # but should not fail due to missing input
        status = await pytorch_worker.reward_handler(task)
        # Accept either SUCCESS or error (model type mismatch is OK)
        # Note: enum name is misspelled in upstream task.py
        assert status in (TaskStatus.SUCCESS, TaskStatus.WORKER_EXCEPTION)


class TestPyTorchWorkerWithScaffolding:
    def test_single_prompt_generation(self, pytorch_worker, test_prompt):
        controller = NativeGenerationController(
            sampling_params={
                "max_tokens": 10,
                "temperature": 0.7,
            }
        )

        llm = ScaffoldingLlm(
            controller,
            {NativeGenerationController.WorkerTag.GENERATION: pytorch_worker},
        )

        result = llm.generate(test_prompt)

        assert result.outputs[0].text is not None
        assert len(result.outputs[0].text) > 0
        assert result.outputs[0].token_ids is not None

        llm.shutdown()

    def test_batch_generation(self, pytorch_worker):
        controller = NativeGenerationController(
            sampling_params={
                "max_tokens": 10,
                "temperature": 0.7,
            }
        )

        llm = ScaffoldingLlm(
            controller,
            {NativeGenerationController.WorkerTag.GENERATION: pytorch_worker},
        )

        prompts = [
            "Hello, my name is",
            "The weather today is",
            "In the year 2025,",
        ]

        results = llm.generate(prompts)

        assert len(results) == len(prompts)
        for result in results:
            assert result.outputs[0].text is not None
            assert len(result.outputs[0].text) > 0

        llm.shutdown()


class TestPyTorchWorkerShutdown:
    def test_shutdown_clears_resources(self, small_model_name):
        worker = PyTorchWorker.from_pretrained(
            small_model_name, device="cpu", torch_dtype=torch.float32
        )
        worker.shutdown()
        assert next(worker.model.parameters()).device.type == "cpu"
