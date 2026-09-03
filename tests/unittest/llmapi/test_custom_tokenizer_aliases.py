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
"""Custom-tokenizer alias resolution through ``LlmArgs``.

``tensorrt_llm.tokenizer.TOKENIZER_ALIASES`` is the one place where built-in
custom tokenizers register a short alias. ``llm_args`` used to carry its own
copy of that table, and the copy drifted: an alias present only in the
canonical table loaded fine through ``load_custom_tokenizer`` but made
``LlmArgs(custom_tokenizer=<alias>)`` fail with "not enough values to
unpack", because the unresolved alias was split as if it were a dotted import
path. These tests pin the two tables to one object and drive every registered
alias through the ``LlmArgs`` validator.
"""

import importlib
from unittest import mock

import pytest

import tensorrt_llm.llmapi.llm_args as llm_args_mod
from tensorrt_llm.llmapi.llm_args import TorchLlmArgs
from tensorrt_llm.llmapi.tokenizer import TokenizerBase
from tensorrt_llm.tokenizer import TOKENIZER_ALIASES

pytestmark = pytest.mark.cpu_only

DUMMY_MODEL = "/tmp/dummy_model"


def _resolve(alias: str) -> type[TokenizerBase]:
    module_path, class_name = TOKENIZER_ALIASES[alias].rsplit(".", 1)
    return getattr(importlib.import_module(module_path), class_name)


def test_llm_args_uses_the_canonical_alias_table() -> None:
    """One table, not a copy that can drift."""
    assert llm_args_mod.TOKENIZER_ALIASES is TOKENIZER_ALIASES


@pytest.mark.parametrize("alias", sorted(TOKENIZER_ALIASES))
def test_every_alias_names_an_importable_tokenizer_class(alias: str) -> None:
    tokenizer_class = _resolve(alias)
    assert issubclass(tokenizer_class, TokenizerBase)
    assert callable(getattr(tokenizer_class, "from_pretrained", None))


@pytest.mark.parametrize("alias", sorted(TOKENIZER_ALIASES))
def test_llm_args_resolves_every_registered_alias(alias: str) -> None:
    """``custom_tokenizer=<alias>`` reaches the aliased class's loader.

    ``from_pretrained`` is stubbed so no checkpoint is read; the point is that
    the alias is resolved to the class rather than split as an import path.
    """
    tokenizer_class = _resolve(alias)
    loaded = mock.Mock(spec=TokenizerBase)
    with mock.patch.object(
        tokenizer_class, "from_pretrained", return_value=loaded
    ) as from_pretrained:
        args = TorchLlmArgs(model=DUMMY_MODEL, custom_tokenizer=alias)

    from_pretrained.assert_called_once()
    assert from_pretrained.call_args.args[0] == DUMMY_MODEL
    assert args.tokenizer is loaded


def test_unknown_custom_tokenizer_is_still_rejected() -> None:
    with pytest.raises(ValueError, match="Failed to load custom tokenizer"):
        TorchLlmArgs(model=DUMMY_MODEL, custom_tokenizer="not_a_registered_alias")
