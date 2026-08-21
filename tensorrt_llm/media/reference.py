# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""The wire types for a media reference, shared by serve and VisualGen.

These are declaration-only: the schema a caller fills in, with no resolver,
decoder or engine behind them. They live here rather than under
``visual_gen`` so the common serving protocol can name them without the
request schema of every LLM deployment pulling a vertical in behind it.
Depend on nothing but ``pydantic`` and keep it that way.
"""

from typing import Any, Optional, Union

from pydantic import Field, model_validator
from typing_extensions import Literal

from tensorrt_llm.llmapi.utils import StrictBaseModel, set_api_status

Role = Literal["reference", "first_frame", "last_frame"]

# Wire form of a reference's ``content``. Declared explicitly rather than
# sniffed: a bare string is otherwise ambiguous between a local path and
# base64, and guessing lets a mistyped path silently become base64 (or a
# malformed base64 silently become a filesystem read).
ContentFormat = Literal["path", "url", "base64", "bytes"]


@set_api_status("prototype")
class MediaRef(StrictBaseModel):
    """A single media reference (image / video / audio).

    Carried by ``image_reference`` / ``video_reference`` / ``audio_reference``;
    the field it sits in fixes the modality. ``role`` is required only when the
    target model accepts that modality in more than one role (e.g. image first +
    last frame); otherwise the pipeline knows the reference's meaning and
    ``role`` may be omitted (video/audio are always the single ``reference``).
    """

    content: Union[str, bytes] = Field(
        description="The reference payload, in the form declared by ``format``."
    )
    format: ContentFormat = Field(
        description=(
            "Wire form of ``content``: ``path`` (local file; a ``file://`` URI is "
            "also accepted), ``url`` (``http(s)``, fetched through the SSRF-guarded "
            "loader), ``base64`` (a ``data:`` URI is also accepted), or ``bytes``."
        )
    )
    role: Optional[Role] = Field(
        default=None, description="``reference`` | ``first_frame`` | ``last_frame``."
    )

    @model_validator(mode="after")
    def _check_content_matches_format(self):
        """Reject a ``content`` whose Python type contradicts ``format``.

        ``bytes`` is the only format carrying a binary payload; the other three
        name a location or an encoding and are therefore strings. Checking the
        pairing here fails at construction — an HTTP 422 or an immediate
        ``ValueError`` — instead of deep in the engine's resolve step.
        """
        if self.format == "bytes":
            if not isinstance(self.content, bytes):
                raise ValueError(
                    f"format='bytes' requires bytes content, got {type(self.content).__name__}."
                )
        elif not isinstance(self.content, str):
            raise ValueError(
                f"format={self.format!r} requires string content, got "
                f"{type(self.content).__name__}."
            )
        return self


def reject_bare_refs(value: Any) -> Any:
    """Reject the bare path/bytes shorthand with an actionable message.

    Runs before coercion, so the caller sees what to do instead of a union
    mismatch reported against an inner model. A bare string has nowhere to
    declare its wire form, and guessing is what ``format`` exists to prevent.
    """
    for x in value if isinstance(value, list) else [value]:
        if isinstance(x, (str, bytes)):
            raise ValueError(
                "a reference must declare its wire form; a bare "
                f"{type(x).__name__} is no longer accepted. Pass "
                'MediaRef(content=..., format="path"|"url"|"base64"|"bytes").'
            )
    return value
