import os
import shutil
import tempfile
import unittest

import torch

from tensorrt_llm.executor.rpc_torch_dist_executor import RpcTorchDistExecutor
from tensorrt_llm.llmapi.llm import LLM
from tensorrt_llm.sampling_params import SamplingParams


class TestRpcTorchDistExecutor(unittest.TestCase):
    def setUp(self):
        self.model_dir = tempfile.mkdtemp()
        # We use a tiny fake model or just skip if not available?
        # Creating a real model is hard. We can mock or use a dummy path if the test environment allows.
        # Since we are testing the orchestration, we ideally want a real tiny model.
        # However, getting a model might be slow.

        # Let's assume we can use "gpt2" from HF if internet is allowed, or skip.
        self.skip_if_no_gpu = unittest.skipIf(not torch.cuda.is_available(), "Skip if no GPU")

    def tearDown(self):
        shutil.rmtree(self.model_dir)

    def _test_executor_impl(self, tp_size):
        model_path = "/workspace/project/TensorRT-LLM/TinyLlama-1.1B-Chat-v1.0"
        if not os.path.exists(model_path):
            print(f"Model not found at {model_path}, skipping test.")
            return

        print(f"Initializing LLM with RpcTorchDistExecutor (TP={tp_size})...")
        llm = None
        try:
            llm = LLM(
                model=model_path,
                tensor_parallel_size=tp_size,
                executor_cls=RpcTorchDistExecutor,
                orchestrator_type="rpc_torch_dist",
                backend="pytorch",
            )

            sampling_params = SamplingParams(max_tokens=10)
            prompts = ["Hello, my name is", "The future of AI is"]

            print("Generating...")
            outputs = llm.generate(prompts, sampling_params=sampling_params)

            for output in outputs:
                print(f"Prompt: {output.prompt}")
                print(f"Generated: {output.outputs[0].text}")
                self.assertTrue(len(output.outputs[0].text) > 0)
        finally:
            print("Shutting down...")
            if llm is not None:
                llm.shutdown()

    @unittest.skipIf(not torch.cuda.is_available(), "Need at least 1 GPU")
    def test_rpc_torch_dist_executor_tp1(self):
        self._test_executor_impl(1)

    @unittest.skipIf(
        not torch.cuda.is_available() or torch.cuda.device_count() < 2, "Need at least 2 GPUs"
    )
    def test_rpc_torch_dist_executor_tp2(self):
        self._test_executor_impl(2)


if __name__ == "__main__":
    unittest.main()
