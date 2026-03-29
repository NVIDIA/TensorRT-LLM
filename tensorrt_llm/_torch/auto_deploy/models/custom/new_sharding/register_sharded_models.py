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

"""Toggle hint-driven custom model registration by editing __init__.py.

Each new_sharding model file contains a module-level
``AutoModelForCausalLMFactory.register_custom_model_cls(...)`` call that fires
at import time.  This script adds or removes import lines in the parent
``__init__.py`` so that pytest (or any other process) picks up the
registration state from source.

Usage:
    python register_sharded_models.py 1       # register (add imports)
    python register_sharded_models.py true     # register
    python register_sharded_models.py on       # register
    python register_sharded_models.py 0        # deregister (remove imports)
    python register_sharded_models.py false    # deregister
    python register_sharded_models.py off      # deregister
"""

import sys
from pathlib import Path

_INIT_FILE = Path(__file__).resolve().parent.parent / "__init__.py"

_MARKER = "# >> new_sharding registration"

_IMPORT_LINES = [
    f"import tensorrt_llm._torch.auto_deploy.models.custom.new_sharding.modeling_deepseek  {_MARKER}",
    f"import tensorrt_llm._torch.auto_deploy.models.custom.new_sharding.modeling_nemotron_h  {_MARKER}",
    f"import tensorrt_llm._torch.auto_deploy.models.custom.new_sharding.modeling_qwen3_5_moe  {_MARKER}",
]


def register_all() -> None:
    """Add new_sharding import lines to __init__.py (idempotent)."""
    text = _INIT_FILE.read_text()
    if _MARKER in text:
        print("Already registered (imports present).")
        return
    block = "\n".join(_IMPORT_LINES) + "\n"
    _INIT_FILE.write_text(text.rstrip("\n") + "\n\n" + block)
    print(f"Registered: added {len(_IMPORT_LINES)} import lines to {_INIT_FILE}")


def deregister_all() -> None:
    """Remove new_sharding import lines from __init__.py (idempotent)."""
    text = _INIT_FILE.read_text()
    lines = [line for line in text.splitlines(keepends=True) if _MARKER not in line]
    cleaned = "".join(lines)
    if cleaned == text:
        print("Already deregistered (no imports to remove).")
        return
    _INIT_FILE.write_text(cleaned)
    print(f"Deregistered: removed new_sharding import lines from {_INIT_FILE}")


def _parse_bool(value: str) -> bool:
    if value.lower() in ("1", "true", "on"):
        return True
    if value.lower() in ("0", "false", "off"):
        return False
    raise ValueError(f"Cannot interpret '{value}' as boolean. Use 1/0, true/false, or on/off.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <1|0|true|false|on|off>")
        sys.exit(1)

    if _parse_bool(sys.argv[1]):
        register_all()
    else:
        deregister_all()
