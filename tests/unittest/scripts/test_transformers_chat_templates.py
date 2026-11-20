# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Verify the upstream contract for saving named chat templates."""

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast
from transformers.processing_utils import ProcessorMixin

pytestmark = pytest.mark.cpu_only


class _MinimalProcessor(ProcessorMixin):
    attributes = ["tokenizer"]
    tokenizer_class = "PreTrainedTokenizerFast"

    def __init__(self, tokenizer, chat_template=None):
        super().__init__(tokenizer=tokenizer, chat_template=chat_template)


@pytest.fixture(params=["tokenizer", "processor"])
def template_owner(request):
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({"[UNK]": 0, "hello": 1}, unk_token="[UNK]")),
        unk_token="[UNK]",
    )
    if request.param == "processor":
        return _MinimalProcessor(tokenizer=tokenizer)
    return tokenizer


@pytest.mark.parametrize("name", ["../escaped", "../../escaped", "absolute"])
def test_chat_template_save_rejects_paths_outside_template_directory(
    template_owner, tmp_path, name
):
    if name == "absolute":
        name = str(tmp_path / "escaped")
    save_dir = tmp_path / "saved"
    escaped_file = (save_dir / "additional_chat_templates" / f"{name}.jinja").resolve()
    template_owner.chat_template = {"default": "{{ messages }}", name: "untrusted template"}

    with pytest.raises(ValueError):
        template_owner.save_pretrained(save_dir)

    assert not escaped_file.exists()


def test_named_chat_templates_save_and_load(template_owner, tmp_path):
    templates = {"default": "{{ messages }}", "tool_use": "{{ tools }}"}
    template_owner.chat_template = templates
    template_owner.save_pretrained(tmp_path)

    reloaded = type(template_owner).from_pretrained(tmp_path, local_files_only=True)
    assert reloaded.chat_template == templates
