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
"""Run a scaffolding controller described by a config file.

The controller topology comes entirely from the config, so switching between
plain generation, majority vote and best-of-N is an edit to a JSON or YAML file
rather than a code change.

Example:
    ```bash
    python run_from_config.py \
        --config configs/majority_vote.json \
        --model_dir <path to the generation model>
    ```
"""

import argparse

from tensorrt_llm.scaffolding import (
    NativeGenerationController,
    NativeRewardController,
    ScaffoldingLlm,
    TRTLLMWorker,
    load_controller_config,
)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to a JSON or YAML file describing the controller",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Path to the directory containing the generation model",
    )
    parser.add_argument(
        "--reward_model_dir",
        type=str,
        default=None,
        help="Path to a reward model, required by controllers that score candidates",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()

    controller = load_controller_config(args.config)
    print(f"Built {type(controller).__name__} from {args.config}")

    prompts = [
        "Natalia sold clips to 48 of her friends in April, and then she sold half as many "
        "clips in May. How many clips did Natalia sell altogether in April and May?\r\n\r\n",
    ]

    workers = {}
    llm = None
    try:
        workers[NativeGenerationController.WorkerTag.GENERATION] = TRTLLMWorker.init_with_new_llm(
            args.model_dir
        )
        if args.reward_model_dir is not None:
            workers[NativeRewardController.WorkerTag.REWARD] = TRTLLMWorker.init_with_new_llm(
                args.reward_model_dir
            )

        llm = ScaffoldingLlm(controller, workers)

        for result in llm.generate(prompts):
            print(result.outputs[0].text)
    finally:
        # ScaffoldingLlm owns the workers once it exists; before that we own them, and a
        # failure part way through worker creation must still release the earlier ones.
        if llm is not None:
            llm.shutdown(shutdown_workers=True)
        else:
            for worker in workers.values():
                worker.shutdown()

    print("main shut down done")


if __name__ == "__main__":
    main()
