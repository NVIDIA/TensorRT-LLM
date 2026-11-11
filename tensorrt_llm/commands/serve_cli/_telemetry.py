# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

from tensorrt_llm.usage import (TerminalOutcome, get_observed_signal,
                                record_termination_observation)


def _report_observed_child_failure(return_code: int, component: str,
                                   lifecycle_phase: str) -> None:
    """Preserve a child failure for the parent's authoritative exit boundary."""
    # A child terminated as part of an already-observed process signal is a
    # shutdown consequence, not evidence that the child caused the shutdown.
    if get_observed_signal():
        return
    if return_code < 0:
        signal_number = -return_code
        exit_code = 128 + signal_number
    else:
        signal_number = 0
        exit_code = return_code
    record_termination_observation(
        TerminalOutcome(
            termination_kind="worker_failure",
            component=component,
            reporting_source="supervisor",
            exit_code_known=True,
            exit_code=exit_code,
            signal_number=signal_number,
        ),
        lifecycle_phase=lifecycle_phase,
    )
