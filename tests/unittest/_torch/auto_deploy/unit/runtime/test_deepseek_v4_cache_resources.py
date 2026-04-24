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

import torch

from tensorrt_llm._torch.auto_deploy._compat import KvCacheConfig
from tensorrt_llm._torch.auto_deploy.custom_ops.attention_interface import (
    DeepSeekV4CacheResourceDescriptor,
    PagedResourceMetadataNames,
    PagedResourceSequenceMetadata,
    SequenceInfo,
    UnpagedResourceHandler,
)
from tensorrt_llm._torch.auto_deploy.shim.interface import CachedSequenceInterface


def _descriptor_set() -> dict[str, DeepSeekV4CacheResourceDescriptor]:
    return {
        "swa": DeepSeekV4CacheResourceDescriptor(
            resource_name="swa",
            cache_suffix="swa_kv_cache",
            token_shape=(2, 1, 8),
            dtype=torch.float16,
            logical_length_divisor=1,
            tokens_per_block=4,
            max_logical_entries_per_seq=16,
        ),
        "mhc": DeepSeekV4CacheResourceDescriptor(
            resource_name="mhc",
            cache_suffix="mhc_cache",
            token_shape=(1, 8),
            dtype=torch.bfloat16,
            logical_length_divisor=4,
            tokens_per_block=2,
            max_logical_entries_per_seq=4,
        ),
        "indexer": DeepSeekV4CacheResourceDescriptor(
            resource_name="indexer",
            cache_suffix="indexer_cache",
            token_shape=(1, 4),
            dtype=torch.float16,
            logical_length_divisor=4,
            tokens_per_block=2,
            max_logical_entries_per_seq=4,
        ),
        "compressor_state": DeepSeekV4CacheResourceDescriptor(
            resource_name="compressor_state",
            cache_suffix="compressor_state",
            token_shape=(4, 8),
            dtype=torch.bfloat16,
            logical_length_divisor=1,
            tokens_per_block=1,
            max_logical_entries_per_seq=1,
        ),
    }


def _sequence_info(max_num_tokens: int = 16) -> SequenceInfo:
    seq_info = SequenceInfo(
        max_seq_len=256,
        max_batch_size=2,
        max_num_tokens=max_num_tokens,
        tokens_per_block=4,
    )
    seq_info.to("cpu")
    seq_info.update_cache_information(num_blocks=32)
    return seq_info


def _register_all_metadata(
    seq_info: SequenceInfo,
) -> tuple[
    dict[str, DeepSeekV4CacheResourceDescriptor],
    dict[str, PagedResourceMetadataNames],
]:
    descriptors = _descriptor_set()
    metadata_names = {}
    for descriptor in descriptors.values():
        metadata_names[descriptor.resource_name] = descriptor.create_handler().register_metadata(
            seq_info
        )
    return descriptors, metadata_names


def _activate_named_metadata_host_args(
    seq_info: SequenceInfo, metadata_names: dict[str, PagedResourceMetadataNames]
) -> None:
    for names in metadata_names.values():
        for arg_name in names.all_arg_names():
            seq_info.activate_arg(f"{arg_name}_host")


def test_deepseek_v4_resource_handlers_allocate_paged_pools() -> None:
    seq_info = _sequence_info()

    for descriptor in _descriptor_set().values():
        handler = descriptor.create_handler()
        metadata_names = handler.register_metadata(seq_info)
        cache = handler.allocate(seq_info)

        assert handler.is_paged
        assert not isinstance(handler, UnpagedResourceHandler)
        assert metadata_names.cache_loc in seq_info.available_args
        assert metadata_names.cu_num_pages in seq_info.available_args

        expected_pages = seq_info.max_batch_size * handler.max_pages_per_seq(seq_info) + 1
        expected_shape = (
            expected_pages,
            handler.tokens_per_block or seq_info.tokens_per_block,
            *handler.token_shape,
        )
        assert cache.shape == expected_shape
        assert cache.dtype == descriptor.dtype
        assert cache.device.type == "cpu"


def test_sequence_info_indexes_named_v4_page_tables_with_different_lengths() -> None:
    seq_info = _sequence_info()
    descriptors = _descriptor_set()
    for descriptor in descriptors.values():
        descriptor.create_handler().register_metadata(seq_info)

    token_lengths = [9, 13]
    paged_metadata = {
        "swa": PagedResourceSequenceMetadata(
            cache_loc=[7, 8, 9, 7, 8, 10, 11],
            cu_num_pages=[0, 3, 7],
            seq_len_with_cache=descriptors["swa"].logical_lengths(token_lengths),
        ),
        "mhc": descriptors["mhc"].build_metadata(
            page_assignments=[[21, 22], [21, 23]],
            token_lengths=token_lengths,
        ),
        "indexer": descriptors["indexer"].build_metadata(
            page_assignments=[[31, 32], [31, 33]],
            token_lengths=token_lengths,
        ),
        "compressor_state": descriptors["compressor_state"].build_metadata(
            page_assignments=[[41], [42]],
            token_lengths=[1, 1],
        ),
    }

    seq_info.nest_sequences(
        input_ids=torch.arange(4, dtype=torch.int),
        cu_seqlen=[0, 2, 4],
        input_pos=[0, 0],
        paged_resource_metadata=paged_metadata,
        slot_idx=[0, 1],
    )

    swa_args = seq_info.get_paged_resource_args("swa", host=True)
    mhc_args = seq_info.get_paged_resource_args("mhc", host=True)

    assert swa_args["cache_loc"].tolist() == [7, 8, 9, 7, 8, 10, 11]
    assert swa_args["cu_num_pages"].tolist() == [0, 3, 7]
    assert mhc_args["cache_loc"].tolist() == [21, 22, 21, 23]
    assert mhc_args["cu_num_pages"].tolist() == [0, 2, 4]

    assert swa_args["cache_loc"][:2].tolist() == swa_args["cache_loc"][3:5].tolist()
    assert mhc_args["cache_loc"][0].item() == mhc_args["cache_loc"][2].item()
    assert swa_args["seq_len_with_cache"].tolist() == token_lengths
    assert mhc_args["seq_len_with_cache"].tolist() == [3, 4]
    assert swa_args["last_page_len"].tolist() == [1, 1]
    assert mhc_args["last_page_len"].tolist() == [1, 2]


def test_sequence_info_advances_named_v4_page_metadata_after_switch_to_generate() -> None:
    seq_info = _sequence_info(max_num_tokens=32)
    descriptors, metadata_names = _register_all_metadata(seq_info)
    _activate_named_metadata_host_args(seq_info, metadata_names)

    token_lengths = [8, 9]
    paged_metadata = {
        "swa": descriptors["swa"].build_metadata(
            page_assignments=[[10, 11], [20, 21, 22]],
            token_lengths=token_lengths,
        ),
        "mhc": descriptors["mhc"].build_metadata(
            page_assignments=[[30], [40, 41]],
            token_lengths=token_lengths,
        ),
        "indexer": descriptors["indexer"].build_metadata(
            page_assignments=[[50], [60, 61]],
            token_lengths=token_lengths,
        ),
        "compressor_state": descriptors["compressor_state"].build_metadata(
            page_assignments=[[70], [80]],
            token_lengths=[1, 1],
        ),
    }

    seq_info.nest_sequences(
        input_ids=torch.arange(sum(token_lengths), dtype=torch.int),
        cu_seqlen=[0, token_lengths[0], sum(token_lengths)],
        input_pos=[0, 0],
        paged_resource_metadata=paged_metadata,
        extra_page_per_seq=[90, 91],
        slot_idx=[0, 1],
    )

    seq_info.switch_to_generate_()

    assert seq_info.get_arg("input_pos_host", truncate=True).tolist() == [7, 8]
    assert seq_info.get_arg("seq_len_host", truncate=True).tolist() == [1, 1]
    assert seq_info.get_paged_resource_args("swa", host=True)["seq_len_with_cache"].tolist() == [
        8,
        9,
    ]

    seq_info.offset_pos_and_cache_(torch.ones(2, dtype=torch.int32))

    swa_args = seq_info.get_paged_resource_args("swa", host=True)
    mhc_args = seq_info.get_paged_resource_args("mhc", host=True)
    indexer_args = seq_info.get_paged_resource_args("indexer", host=True)
    compressor_args = seq_info.get_paged_resource_args("compressor_state", host=True)

    assert swa_args["seq_len_with_cache"].tolist() == [9, 10]
    assert swa_args["last_page_len"].tolist() == [1, 2]
    assert swa_args["cu_num_pages"].tolist() == [0, 3, 6]
    assert swa_args["cache_loc"].tolist() == [10, 11, 90, 20, 21, 22]

    assert mhc_args["seq_len_with_cache"].tolist() == [3, 3]
    assert mhc_args["last_page_len"].tolist() == [1, 1]
    assert mhc_args["cu_num_pages"].tolist() == [0, 2, 4]
    assert mhc_args["cache_loc"].tolist() == [30, 90, 40, 41]

    assert indexer_args["seq_len_with_cache"].tolist() == [3, 3]
    assert indexer_args["last_page_len"].tolist() == [1, 1]
    assert indexer_args["cu_num_pages"].tolist() == [0, 2, 4]
    assert indexer_args["cache_loc"].tolist() == [50, 90, 60, 61]

    assert compressor_args["seq_len_with_cache"].tolist() == [1, 1]
    assert compressor_args["last_page_len"].tolist() == [1, 1]
    assert compressor_args["cu_num_pages"].tolist() == [0, 1, 2]
    assert compressor_args["cache_loc"].tolist() == [70, 80]


def test_ratio_128_compressed_metadata_uses_fewer_entries_than_token_cache() -> None:
    token_cache = DeepSeekV4CacheResourceDescriptor(
        resource_name="swa",
        cache_suffix="swa_kv_cache",
        token_shape=(1, 8),
        dtype=torch.float16,
        logical_length_divisor=1,
        tokens_per_block=8,
    )
    ratio_128_cache = DeepSeekV4CacheResourceDescriptor(
        resource_name="mhc",
        cache_suffix="mhc_cache",
        token_shape=(1, 8),
        dtype=torch.bfloat16,
        logical_length_divisor=128,
        tokens_per_block=2,
    )

    token_lengths = [128, 129, 255]

    assert token_cache.logical_lengths(token_lengths).tolist() == token_lengths
    assert ratio_128_cache.logical_lengths(token_lengths).tolist() == [1, 2, 2]
    assert ratio_128_cache.page_counts(token_lengths).tolist() == [1, 1, 1]
    assert torch.all(
        ratio_128_cache.page_counts(token_lengths) < token_cache.page_counts(token_lengths)
    )


def test_v4_paged_resources_keep_prefix_reuse_enabled_without_unpaged_fallback() -> None:
    interface = CachedSequenceInterface(
        max_seq_len=256,
        max_batch_size=2,
        max_num_tokens=16,
        device="cpu",
        kv_cache_config=KvCacheConfig(
            tokens_per_block=4,
            max_tokens=64,
            free_gpu_memory_fraction=0.0,
            enable_block_reuse=True,
            copy_on_partial_reuse=True,
        ),
    )

    resource_names = interface.add_resource_group(
        {
            descriptor.cache_suffix: descriptor.create_handler()
            for descriptor in _descriptor_set().values()
        }
    )
    interface._caches = {name: None for name in interface._resource_lookup}

    tuned_config = interface._prepare_kv_cache_config(max_tokens=32, kv_managed={})

    assert list(resource_names) == [
        "swa_kv_cache",
        "mhc_cache",
        "indexer_cache",
        "compressor_state",
    ]
    assert all(handler.is_paged for handler in interface._resource_lookup.values())
    assert tuned_config.enable_block_reuse is True
    assert tuned_config.copy_on_partial_reuse is False
