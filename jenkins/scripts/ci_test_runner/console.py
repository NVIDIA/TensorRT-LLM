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
"""Banners and phase summaries printed to the stage log."""

BANNER_WIDTH = 80


def print_banner(title, char="="):
    """Print a prominent phase banner."""
    border = char * BANNER_WIDTH
    print(f"\n{border}")
    print(f"  {title}")
    print(f"{border}\n")


def print_phase_summary(phase, passed, failed, duration_s, phase_failed=None):
    """Print a summary line after a phase completes.

    phase_failed overrides the status when pytest failed without recording a
    failed case (e.g. it was killed by a timeout before writing one).
    """
    if phase_failed is None:
        phase_failed = failed > 0
    status = "FAILED" if phase_failed else "PASSED"
    print(f"\n>> [{phase}] {status}: {passed} passed, {failed} failed ({duration_s:.1f}s)")


def format_duration(seconds):
    """Format seconds into a human-readable string."""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    if m > 0:
        return f"{m}m {s}s"
    return f"{s}s"
