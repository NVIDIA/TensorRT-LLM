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
"""Custom-tokenizer alias resolution through `LlmArgs`.

`tensorrt_llm.tokenizer.TOKENIZER_ALIASES` is the one place where built-in
custom tokenizers register a short alias. `llm_args` used to carry its own
copy of that table, and the copy drifted: an alias present only in the
canonical table loaded fine through `load_custom_tokenizer` but made
`LlmArgs(custom_tokenizer=<alias>)` fail with "not enough values to
unpack", because the unresolved alias was split as if it were a dotted import
path. These tests pin `LlmArgs` to the shared `load_custom_tokenizer` (the
canonical table is re-exported, not copied) and drive every registered alias
through the `LlmArgs` validator.
"""

from unittest import mock

import pytest

import tensorrt_llm.llmapi.llm_args as llm_args_mod
from tensorrt_llm.llmapi.llm_args import TorchLlmArgs
from tensorrt_llm.llmapi.tokenizer import TokenizerBase
from tensorrt_llm.tokenizer import TOKENIZER_ALIASES, load_custom_tokenizer

pytestmark = pytest.mark.cpu_only

DUMMY_MODEL = "/tmp/dummy_model"


def _resolve(alias: str) -> type[TokenizerBase]:
    """Import the tokenizer class an alias maps to.

    The tests below parametrize over the live alias table, so this CPU-only
    module imports every alias target. An alias whose target needs an
    optional dependency (`mistral_common` happens to be pinned in
    requirements.txt today) must skip here rather than fail the CPU test
    list; `importorskip` turns a missing module into a skip.
    """
    module_path, class_name = TOKENIZER_ALIASES[alias].rsplit(".", 1)
    return getattr(pytest.importorskip(module_path), class_name)


def test_llm_args_delegates_to_the_shared_loader() -> None:
    """`LlmArgs` keeps no alias table of its own; it calls the one loader."""
    assert llm_args_mod.TOKENIZER_ALIASES is TOKENIZER_ALIASES
    loaded = mock.Mock(spec=TokenizerBase)
    with mock.patch.object(llm_args_mod, "load_custom_tokenizer", return_value=loaded) as loader:
        args = TorchLlmArgs(
            model=DUMMY_MODEL, custom_tokenizer="deepseek_v32", tokenizer_mode="slow"
        )

    loader.assert_called_once_with(
        "deepseek_v32",
        DUMMY_MODEL,
        trust_remote_code=args.trust_remote_code,
        use_fast=False,
    )
    assert args.tokenizer is loaded


@pytest.mark.parametrize("alias", sorted(TOKENIZER_ALIASES))
def test_every_alias_names_an_importable_tokenizer_class(alias: str) -> None:
    """Each alias target is an importable `TokenizerBase` with a loader."""
    tokenizer_class = _resolve(alias)
    assert issubclass(tokenizer_class, TokenizerBase)
    assert callable(getattr(tokenizer_class, "from_pretrained", None))


@pytest.mark.parametrize("alias", sorted(TOKENIZER_ALIASES))
def test_llm_args_resolves_every_registered_alias(alias: str) -> None:
    """`custom_tokenizer=<alias>` reaches the aliased class's loader.

    `from_pretrained` is stubbed so no checkpoint is read; the point is that
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
    """An identifier that is neither an alias nor an import path errors out."""
    with pytest.raises(ValueError, match="Failed to load custom tokenizer"):
        TorchLlmArgs(model=DUMMY_MODEL, custom_tokenizer="not_a_registered_alias")


def test_unknown_alias_error_names_the_known_aliases() -> None:
    """A dotless non-alias is reported as an unknown alias, not as a bad split."""
    with pytest.raises(ValueError, match="unknown alias") as excinfo:
        load_custom_tokenizer("not_a_registered_alias", DUMMY_MODEL)
    for alias in TOKENIZER_ALIASES:
        assert alias in str(excinfo.value)
    assert "not enough values to unpack" not in str(excinfo.value)


def test_missing_import_path_keeps_the_import_error() -> None:
    """A dotted path that does not import still carries the import error."""
    with pytest.raises(ValueError, match="Failed to load custom tokenizer") as excinfo:
        load_custom_tokenizer("no.such.module.Tokenizer", DUMMY_MODEL)
    assert "No module named" in str(excinfo.value)
