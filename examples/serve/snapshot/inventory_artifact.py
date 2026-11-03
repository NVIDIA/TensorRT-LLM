# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read a root-owned checkpoint through a read-only, unprivileged container mount."""

import json
from pathlib import Path

from qualify import inventory

if __name__ == "__main__":
    print(json.dumps(inventory(Path("/artifact")), sort_keys=True))
