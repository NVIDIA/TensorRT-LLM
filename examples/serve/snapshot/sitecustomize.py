# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Install test observers in spawned native workers; never ignore installation failure."""

import os
import traceback

if os.environ.get("TRTLLM_SNAPSHOT_EXPERIMENT") == "1":
    try:
        import snapshot_observer  # noqa: F401
    except BaseException:
        traceback.print_exc()
        os._exit(78)
