# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from utils.llm_data import llm_models_root

from tensorrt_llm.scaffolding import (
    BestOfNController,
    GenerationTask,
    NativeGenerationController,
    NativeRewardController,
    PRMController,
    PyTorchWorker,
    RewardTask,
    ScaffoldingLlm,
    StreamGenerationTask,
    TaskStatus,
)


@pytest.fixture(scope="module")
def small_model_path():
    """Path to a small local checkpoint; tests must not download from HuggingFace.

    Skips unless the directory really holds a loadable causal LM. Some entries in the
    model cache carry only tokenizer files, which would make ``from_pretrained`` raise
    instead of letting the suite skip cleanly.
    """
    models_root = llm_models_root()
    if models_root is None:
        pytest.skip("LLM_MODELS_ROOT is not set and the default model cache is not mounted")

    model_dir = models_root / "gpt2"
    if not (model_dir / "config.json").is_file():
        pytest.skip(f"{model_dir} has no config.json; not a loadable model directory")

    weight_globs = ("*.safetensors", "pytorch_model*.bin", "model*.bin")
    if not any(next(model_dir.glob(pattern), None) for pattern in weight_globs):
        pytest.skip(
            f"{model_dir} has no model weights; tokenizer-only directories cannot be loaded"
        )

    return str(model_dir)


@pytest.fixture(scope="module")
def pytorch_worker(small_model_path):
    """Create a PyTorchWorker with a small model (shared across module)."""
    worker = PyTorchWorker.from_pretrained(
        small_model_path,
        device="cpu",
        torch_dtype=torch.float32,
    )
    yield worker
    worker.shutdown()


@pytest.fixture
def test_prompt():
    return "The capital of France is"


class TestPyTorchWorkerInitialization:
    def test_from_pretrained(self, small_model_path):
        worker = PyTorchWorker.from_pretrained(
            small_model_path, device="cpu", torch_dtype=torch.float32
        )
        assert worker.model is not None
        assert worker.tokenizer is not None
        worker.shutdown()

    def test_device_selection_explicit(self, small_model_path):
        worker = PyTorchWorker.from_pretrained(
            small_model_path, device="cpu", torch_dtype=torch.float32
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
        # Note: the enum member name is misspelled in task.py; kept as-is to avoid an API break.
        assert status == TaskStatus.WORKER_EXECEPTION

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

    def test_strip_stop_sequence_cuts_at_earliest_match(self, pytorch_worker):
        """Order in task.stop must not decide where the text is cut."""
        task = GenerationTask.create_from_prompt("irrelevant")
        task.stop = ["B", "A"]

        assert pytorch_worker._strip_stop_sequence("xxAyyBzz", task) == "xx"

        task.stop = ["A", "B"]
        assert pytorch_worker._strip_stop_sequence("xxAyyBzz", task) == "xx"

        task.stop = ["missing"]
        assert pytorch_worker._strip_stop_sequence("xxAyyBzz", task) == "xxAyyBzz"

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


@pytest.fixture(scope="module")
def reward_worker(small_model_path):
    """A sequence-classification worker, as reward scoring actually requires."""
    worker = PyTorchWorker.from_pretrained_reward_model(
        small_model_path, device="cpu", torch_dtype=torch.float32, num_labels=2
    )
    yield worker
    worker.shutdown()


class TestPyTorchWorkerReward:
    @pytest.mark.asyncio
    async def test_reward_scores_from_input_str(self, reward_worker):
        task = RewardTask()
        task.input_str = "This is a test."

        assert await reward_worker.reward_handler(task) == TaskStatus.SUCCESS
        assert task.custom_output_params is not None
        score = task.custom_output_params["score"]
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    @pytest.mark.asyncio
    async def test_reward_with_input_tokens(self, reward_worker):
        """input_tokens is accepted in place of input_str."""
        task = RewardTask()
        task.input_str = None
        task.input_tokens = reward_worker.tokenizer.encode("This is a test.")

        assert await reward_worker.reward_handler(task) == TaskStatus.SUCCESS
        assert isinstance(task.custom_output_params["score"], float)

    @pytest.mark.asyncio
    async def test_reward_without_any_input_fails(self, reward_worker):
        """Missing input is an error, not a score computed from empty text."""
        task = RewardTask()
        task.input_str = None
        task.input_tokens = None

        assert await reward_worker.reward_handler(task) == TaskStatus.WORKER_EXECEPTION


class TestPyTorchWorkerContextLogits:
    @pytest.mark.asyncio
    async def test_context_logits_returned(self, pytorch_worker, test_prompt):
        """return_context_logits fills prompt-position logits, as TRTLLMWorker does."""
        task = GenerationTask.create_from_prompt(test_prompt)
        task.max_tokens = 1
        task.temperature = 0.0
        task.return_context_logits = True

        status = await pytorch_worker.generation_handler(task)

        assert status == TaskStatus.SUCCESS
        assert task.context_logits is not None
        prompt_len = len(pytorch_worker.tokenizer.encode(test_prompt))
        assert task.context_logits.shape[0] == prompt_len
        assert task.context_logits.shape[1] == pytorch_worker.model.config.vocab_size

    @pytest.mark.asyncio
    async def test_not_returned_unless_requested(self, pytorch_worker, test_prompt):
        task = GenerationTask.create_from_prompt(test_prompt)
        task.max_tokens = 1
        task.temperature = 0.0

        assert await pytorch_worker.generation_handler(task) == TaskStatus.SUCCESS
        assert task.context_logits is None

    @pytest.mark.asyncio
    async def test_drives_prm_controller(self, pytorch_worker):
        """Score through the real PRMController, which reads task.context_logits.

        ``split_steps=False`` takes PRMController's last-token path, which works with a
        causal LM's vocab-width logits; the step-splitting path expects a two-label
        process reward model.
        """
        tokenizer = pytorch_worker.tokenizer
        if not getattr(tokenizer, "chat_template", None):
            tokenizer.chat_template = (
                "{% for message in messages %}{{ message['content'] }}\n{% endfor %}"
            )

        controller = PRMController(tokenizer, split_steps=False)

        task = GenerationTask.create_from_prompt("What is 2 + 2?")
        task.output_str = "The answer is 4."

        # Drive the controller's generator by hand so the worker plays the reward role.
        process = controller.process([task])
        reward_tasks = next(process)
        assert reward_tasks, "PRMController should emit at least one reward task"
        for reward_task in reward_tasks:
            assert reward_task.return_context_logits is True
            assert await pytorch_worker.generation_handler(reward_task) == TaskStatus.SUCCESS
        with pytest.raises(StopIteration):
            next(process)

        assert controller.scores is not None
        assert len(controller.scores) == len(reward_tasks)
        for score in controller.scores:
            assert 0.0 <= score <= 1.0


class _RewardTaskController(NativeRewardController):
    """Emits RewardTask so the worker's reward_handler is actually reached.

    The built-in reward controllers forward GenerationTask objects, and Worker.run_task
    dispatches on the exact task type, so reward_handler is unreachable through them
    today. Changing that contract affects every existing worker and is tracked
    separately; this keeps the end-to-end path testable in the meantime.
    """

    # ScaffoldingLlm runs prototype_controller.clone(), i.e. a deepcopy, so scores set on
    # the running instance never reach the prototype the test holds. A class attribute is
    # shared across those copies.
    observed_scores = []

    def process(self, tasks, **kwargs):
        reward_tasks = []
        for task in tasks:
            reward_task = RewardTask()
            reward_task.input_str = task.output_str or ""
            reward_task.worker_tag = self.WorkerTag.REWARD
            reward_tasks.append(reward_task)

        yield reward_tasks

        self.scores = [
            (reward_task.custom_output_params or {}).get("score", 0.0)
            for reward_task in reward_tasks
        ]
        _RewardTaskController.observed_scores.extend(self.scores)


class TestPyTorchWorkerBestOfN:
    def test_best_of_n_with_sequence_classification_reward(self, small_model_path, test_prompt):
        """End-to-end Best-of-N: causal LM generates, classification model scores."""
        _RewardTaskController.observed_scores.clear()

        generation_worker = PyTorchWorker.from_pretrained(
            small_model_path, device="cpu", torch_dtype=torch.float32
        )
        reward_worker = PyTorchWorker.from_pretrained_reward_model(
            small_model_path, device="cpu", torch_dtype=torch.float32, num_labels=2
        )

        controller = BestOfNController(
            NativeGenerationController(sampling_params={"max_tokens": 8, "temperature": 0.9}),
            _RewardTaskController(),
            # BestOfNController.process takes sample_num=4 by default and uses
            # max(sample_num, default_sample_num), so match it to keep the count exact.
            default_sample_num=4,
        )

        llm = ScaffoldingLlm(
            controller,
            {
                NativeGenerationController.WorkerTag.GENERATION: generation_worker,
                NativeRewardController.WorkerTag.REWARD: reward_worker,
            },
        )
        try:
            result = llm.generate(test_prompt)

            # Best-of-N returns the selected candidate's text.
            assert result.outputs[0].text is not None
            # reward_handler ran on the classification model for every candidate.
            assert len(_RewardTaskController.observed_scores) == 4
            for score in _RewardTaskController.observed_scores:
                assert isinstance(score, float)
                assert 0.0 <= score <= 1.0
        finally:
            llm.shutdown(shutdown_workers=True)


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
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU to observe the move")
    def test_shutdown_moves_model_off_gpu(self, small_model_path):
        """On CPU the assertion would hold even if shutdown() did nothing."""
        worker = PyTorchWorker.from_pretrained(
            small_model_path, device="cuda", torch_dtype=torch.float32
        )
        assert next(worker.model.parameters()).device.type == "cuda"

        worker.shutdown()

        assert next(worker.model.parameters()).device.type == "cpu"
