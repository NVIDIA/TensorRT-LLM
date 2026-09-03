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

import os


def get_flashinfer_environment() -> tuple[str | None, str | None]:
    """Return every worker's FlashInfer paths exactly once."""
    # Keep importing this pickling helper from initializing MPI in the parent.
    from mpi4py import MPI

    MPI.COMM_WORLD.barrier()
    return (
        os.environ.get("FLASHINFER_WORKSPACE_BASE"),
        os.environ.get("FLASHINFER_CUBIN_DIR"),
    )
