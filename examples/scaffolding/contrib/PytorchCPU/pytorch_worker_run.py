# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import asyncio
import logging
import sys
import time

from tensorrt_llm import scaffolding
from tensorrt_llm.scaffolding.contrib.PytorchCPU import PytorchWorker

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=
        "Run CPU inference using PyTorch worker for TensorRT-LLM scaffolding")
    parser.add_argument(
        '--model_dir',
        type=str,
        required=True,
        help=
        "Path to the directory containing the generation model or Hugging Face model name"
    )
    parser.add_argument('--run_async',
                        action='store_true',
                        help="Run in async mode (default: sync mode)")
    parser.add_argument('--max_batch_size',
                        type=int,
                        default=32,
                        help="Maximum batch size for inference (default: 32)")
    parser.add_argument(
        '--max_num_tokens',
        type=int,
        default=4096,
        help="Maximum number of tokens to process (default: 4096)")
    parser.add_argument('--temperature',
                        type=float,
                        default=0.9,
                        help="Temperature for sampling (default: 0.9)")
    parser.add_argument('--trust_remote_code',
                        action='store_true',
                        help="Trust remote code when loading models")
    parser.add_argument('--verbose',
                        action='store_true',
                        help="Enable verbose logging")
    args = parser.parse_args()
    return args


def test_sync(prompts, proposer_worker, temperature=0.9):
    """
    Test synchronous generation with the PyTorch CPU worker.

    Args:
        prompts: List of prompts to generate responses for
        proposer_worker: The PyTorch worker instance
        temperature: Temperature for sampling
    """
    logger.info("Starting synchronous generation test")
    start_time = time.time()
    llm = None

    try:
        prototype_controller = scaffolding.NativeGenerationController(
            sampling_params={"temperature": temperature})

        llm = scaffolding.ScaffoldingLlm(
            prototype_controller,
            {
                scaffolding.NativeGenerationController.WorkerTag.GENERATION:
                proposer_worker
            },
        )

        logger.info(f"Processing {len(prompts)} prompts...")
        results = llm.generate(prompts)

        for i, result in enumerate(results):
            logger.info(f"Result {i+1}:")
            print(f"Prompt: {prompts[i][:100]}...")
            print(f"Response: {result.output.output_str}")
            print("-" * 80)

        elapsed_time = time.time() - start_time
        logger.info(f"Generation completed in {elapsed_time:.2f} seconds")

    except Exception:
        logger.exception("Error during synchronous generation")
        raise
    finally:
        logger.info('Shutting down...')
        try:
            if llm is not None:
                llm.shutdown()
            proposer_worker.shutdown()
            logger.info('Shutdown completed successfully')
        except Exception:
            logger.exception("Error during shutdown")


def test_async(prompt, proposer_worker, temperature=0.9):
    """
    Test asynchronous generation with the PyTorch CPU worker.

    Args:
        prompt: Single prompt to generate response for
        proposer_worker: The PyTorch worker instance
        temperature: Temperature for sampling
    """

    async def test_async_func(prompt, proposer_worker, temperature):
        logger.info("Starting asynchronous generation test")
        start_time = time.time()

        llm = None
        try:
            prototype_controller = scaffolding.NativeGenerationController(
                sampling_params={"temperature": temperature})
            llm = scaffolding.ScaffoldingLlm(
                prototype_controller,
                {
                    scaffolding.NativeGenerationController.WorkerTag.GENERATION:
                    proposer_worker
                },
            )

            logger.info("Generating response asynchronously...")
            future = llm.generate_async(prompt)
            result = await future.aresult()

            elapsed_time = time.time() - start_time
            logger.info(
                f"Async generation completed in {elapsed_time:.2f} seconds")

            print(f"Prompt: {prompt[:100]}...")
            print(f"Response: {result.output.output_str}")
            print("-" * 80)

        except Exception:
            logger.exception("Error during asynchronous generation")
            raise
        finally:
            logger.info('Shutting down...')
            try:
                if llm is not None:
                    llm.shutdown()
                proposer_worker.shutdown()
                logger.info('Shutdown completed successfully')
            except Exception:
                logger.exception("Error during shutdown")

    asyncio.run(test_async_func(prompt, proposer_worker, temperature))


def main():
    """
    Main function to run the PyTorch CPU worker example.
    """
    args = parse_arguments()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.info("Verbose logging enabled")

    logger.info(f"Initializing PyTorch CPU worker with model: {args.model_dir}")

    # Sample prompts for testing
    prompts = [
        "Anton sold GPUs to 48 of his friends in April, and then he sold half as many GPUs in May. How many GPUs did Anton sell altogether in April and May?",
        "There exist real numbers x and y, both greater than 1, such that log_x(y^x) = log_y(x^(4y)) = 10. Find xy.",
        "Find the largest possible real part of (75+117i)z + (96+144i)/z where z is a complex number with |z|=4.",
        "What is the capital of France?",
        "Explain the concept of machine learning in simple terms.",
    ]

    try:
        # Initialize the PyTorch worker
        llm_worker = PytorchWorker(
            model_path=args.model_dir,
            max_batch_size=args.max_batch_size,
            max_num_tokens=args.max_num_tokens,
            trust_remote_code=args.trust_remote_code,
        )

        logger.info("PyTorch worker initialized successfully")

        if args.run_async:
            logger.info("Running in asynchronous mode")
            test_async(prompts[0], llm_worker, args.temperature)
        else:
            logger.info("Running in synchronous mode")
            test_sync(prompts, llm_worker, args.temperature)

    except Exception as e:
        logger.error(f"Failed to run example: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
