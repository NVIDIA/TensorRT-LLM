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
"""Regression test for issue #13320.

When FlashMLA is enabled for an MLA model, ``create_py_executor`` overrides the
local ``tokens_per_block`` variable to 64 before it constructs the
``KVCacheManager``. The same override must also be written back onto
``kv_cache_config.tokens_per_block`` so that ``KvCacheConnectorScheduler``
subclasses (LMCache, Dynamo KVBM, ...) — which are instantiated further down
via ``scheduler_cls(llm_args)`` and read ``llm_args.kv_cache_config.tokens_per_block``
— see the effective block size rather than the stale user/default value.

Without the propagation, the connector sized its block pools at 32 tokens
while the engine's ``KVCacheManager`` ran at 64, producing a frozen
``cache_block_ids`` view and silently-corrupted decode KV.

This test pins the propagation in source so a future refactor cannot quietly
drop it again. The end-to-end behavior is covered separately by C++ tests
under cpp/tests/unit_tests/batch_manager/kvCacheManagerTest.cpp.
"""

import inspect
import re

from tensorrt_llm._torch.pyexecutor import py_executor_creator


def _get_create_py_executor_source() -> str:
    return inspect.getsource(py_executor_creator.create_py_executor)


def test_flash_mla_branch_overrides_local_tokens_per_block():
    """Sanity check: the FlashMLA branch still sets ``tokens_per_block = 64``.

    If this assertion ever fails the propagation test below is meaningless,
    so we keep the two checks split for clearer failure messages.
    """
    source = _get_create_py_executor_source()
    assert re.search(
        r"if\s+model_engine\.model\.model_config\.enable_flash_mla\s*:\s*\n"
        r"\s*tokens_per_block\s*=\s*64",
        source,
    ), (
        "FlashMLA branch must still force tokens_per_block to 64 inside "
        "create_py_executor; the regression guard for #13320 depends on it."
    )


def test_flash_mla_tokens_per_block_propagates_to_kv_cache_config():
    """Regression guard for #13320.

    Inside the ``if model_engine.model.model_config.enable_flash_mla`` block,
    after ``tokens_per_block = 64``, ``kv_cache_config.tokens_per_block`` must
    be assigned the same value so connector schedulers see the effective
    block size.
    """
    source = _get_create_py_executor_source()

    flash_mla_block_match = re.search(
        r"if\s+model_engine\.model\.model_config\.enable_flash_mla\s*:\s*\n"
        r"((?:[ \t]+.*\n)+)",
        source,
    )
    assert flash_mla_block_match, (
        "Could not locate the FlashMLA branch in create_py_executor. "
        "The regression test for #13320 needs updating if the surrounding "
        "structure changed."
    )

    flash_mla_body = flash_mla_block_match.group(1)
    assert "tokens_per_block = 64" in flash_mla_body, (
        "FlashMLA branch no longer overrides tokens_per_block to 64."
    )
    assert "kv_cache_config.tokens_per_block = tokens_per_block" in flash_mla_body, (
        "FlashMLA branch must propagate the tokens_per_block override back "
        "onto kv_cache_config so KvCacheConnectorScheduler subclasses "
        "(LMCache, Dynamo KVBM, ...) read the effective value instead of the "
        "stale user/default. See https://github.com/NVIDIA/TensorRT-LLM/issues/13320."
    )
