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
"""The public GSM8K evaluator (``examples/llm-api/gsm8k_eval``) that this CI job runs.

The CI job selects, prompts, asks and scores through exactly the code manual users run with
``examples/llm-api/eval_gsm8k.py``; it only adds the staged inputs, the recipe, the hardware
check and the regression criterion.
"""

import sys
from pathlib import Path

EXAMPLES_DIR = Path(__file__).resolve().parents[5] / "examples" / "llm-api"
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

from gsm8k_eval import protocol, records, runner, server  # noqa: E402

__all__ = ["EXAMPLES_DIR", "protocol", "records", "runner", "server"]
