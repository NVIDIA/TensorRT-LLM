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
"""Read the chat template that ships with a local Hugging Face checkpoint."""

import json
from pathlib import Path

# A checkpoint stores its chat template in exactly one of these, depending on
# which `transformers` version wrote it and whether the repository is a
# tokenizer or a processor repository. They are searched in the order
# `transformers` itself prefers them.
_CHAT_TEMPLATE_FILES = (
    "chat_template.jinja",
    "tokenizer_config.json",
    "chat_template.json",
)


def read_chat_template(model: str) -> str:
    """Return the chat template of a local model directory, or "" if there is none.

    Parser auto-detection reads the template to tell thinking from
    non-thinking checkpoints and JSON from XML tool calls. A template that is
    present but stored in a file the reader does not know about therefore has
    to be found: reading it as "no template" silently selects the wrong parser.
    """
    model_dir = Path(model)
    for name in _CHAT_TEMPLATE_FILES:
        path = model_dir / name
        if not path.is_file():
            continue
        template = (
            path.read_text(encoding="utf-8")
            if path.suffix == ".jinja"
            else _read_json_chat_template(path)
        )
        if template:
            return template
    return ""


def _read_json_chat_template(path: Path) -> str:
    """Return the `chat_template` string of a JSON file, or "" if absent."""
    try:
        with open(path, encoding="utf-8") as f:
            config = json.load(f)
    except (OSError, ValueError):
        return ""
    template = config.get("chat_template")
    return template if isinstance(template, str) else ""
