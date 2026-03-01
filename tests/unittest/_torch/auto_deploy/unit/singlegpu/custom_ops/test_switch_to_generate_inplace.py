# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Tests for SequenceInfo.switch_to_generate_inplace."""

import torch

from tensorrt_llm._torch.auto_deploy.custom_ops.attention_interface import SequenceInfo

TOKENS_PER_BLOCK = 4
MAX_SEQ_LEN = 128
MAX_BATCH_SIZE = 8
MAX_NUM_TOKENS = 64

_CACHED_ARGS = (
    "cu_seqlen",
    "batch_info_host",
    "tokens_gather_info_host",
    "seq_len_with_cache",
    "last_page_len",
    "pages_per_seq",
    "cu_num_pages",
    "cache_loc",
    "extra_page_per_seq",
    "input_pos",
    "seq_len",
)


def _make_seq_info(extra_activate=()) -> SequenceInfo:
    """Build a SequenceInfo with typical cached-attention args activated."""
    si = SequenceInfo(
        max_seq_len=MAX_SEQ_LEN,
        max_batch_size=MAX_BATCH_SIZE,
        tokens_per_block=TOKENS_PER_BLOCK,
        max_num_tokens=MAX_NUM_TOKENS,
    )
    for arg in _CACHED_ARGS:
        si.activate_arg(arg)
    for arg in extra_activate:
        si.activate_arg(arg)
    return si


def _nest_prefill(si: SequenceInfo, input_ids, pages_per_seq, cache_loc, **kw):
    """Convenience wrapper: nest prefill sequences and return the SequenceInfo."""
    si.nest_sequences(
        input_ids, input_pos=0, pages_per_seq=pages_per_seq, cache_loc=cache_loc, **kw
    )
    return si


class TestSwitchToGeneratePackedToDecode:
    """Packed (prefill/extend) layout -> decode layout."""

    def test_basic_two_prefill_sequences(self):
        """Two prefill sequences, increment by 1 each."""
        si = _make_seq_info()
        _nest_prefill(
            si,
            input_ids=[[1, 2, 3], [4, 5, 6, 7]],
            pages_per_seq=[1, 1],
            cache_loc=[10, 20],
        )

        increment = torch.tensor([1, 1], dtype=torch.int32)
        si.switch_to_generate_(increment)

        # batch_info_host: all decode
        bi = si._input_buffer.get_host_view("batch_info")
        assert bi[:6].tolist() == [0, 0, 0, 0, 2, 2]

        # tokens_gather_info: gathering disabled
        lgi = si._input_buffer.get_host_view("tokens_gather_info")
        assert lgi[0].item() == 2
        assert lgi[1].item() == 0

        # position_ids: last positions were 2 and 3, +1 = 3 and 4
        pos = si._input_buffer.get_view("position_ids")
        assert pos[0].item() == 3
        assert pos[1].item() == 4

        # cu_seqlen: decode layout
        cu = si._input_buffer.get_view("cu_seqlen")
        assert cu[:3].tolist() == [0, 1, 2]

        # seq_len_with_cache: was [3, 4], +1 = [4, 5]
        swc = si._input_buffer.get_view("seq_len_with_cache")
        assert swc[0].item() == 4
        assert swc[1].item() == 5

        # last_page_len: (4-1)%4+1=4, (5-1)%4+1=1
        lpl = si._input_buffer.get_view("last_page_len")
        assert lpl[0].item() == 4
        assert lpl[1].item() == 1

        # seq_len: all 1 (decode)
        sl = si._input_buffer.get_view("seq_len")
        assert sl[0].item() == 1
        assert sl[1].item() == 1

        # input_pos: was [0, 0], +1 = [1, 1]
        ip = si._input_buffer.get_view("input_pos")
        assert ip[0].item() == 1
        assert ip[1].item() == 1

    def test_single_prefill_with_nonzero_input_pos(self):
        """Single prefill with input_pos > 0 (resumption scenario)."""
        si = _make_seq_info()
        si.nest_sequences(
            input_ids=[[1, 2, 3, 4, 5]],
            input_pos=[10],
            pages_per_seq=[4],
            cache_loc=[0, 1, 2, 3],
        )

        increment = torch.tensor([1], dtype=torch.int32)
        si.switch_to_generate_(increment)

        # position_ids: last pos was 14 (10+5-1), +1 = 15
        pos = si._input_buffer.get_view("position_ids")
        assert pos[0].item() == 15

        # seq_len_with_cache: was 15 (10+5), +1 = 16
        swc = si._input_buffer.get_view("seq_len_with_cache")
        assert swc[0].item() == 16

        # input_pos: was 10, +1 = 11
        ip = si._input_buffer.get_view("input_pos")
        assert ip[0].item() == 11


class TestSwitchToGenerateDecodeToDecode:
    """Already in decode layout -> advance by 1."""

    def _setup_decode_batch(self, si, positions, swc_vals, pages_per_seq, cache_loc):
        """Set up a generate-only batch then manually verify decode layout."""
        batch_size = len(positions)
        si.nest_sequences(
            input_ids=[[tok] for tok in range(batch_size)],
            position_ids=[[p] for p in positions],
            input_pos=positions,
            seq_len=[1] * batch_size,
            batch_info=[0, 0, 0, 0, batch_size, batch_size],
            pages_per_seq=pages_per_seq,
            cache_loc=cache_loc,
            seq_len_with_cache=swc_vals,
        )

    def test_decode_increment_by_one(self):
        """All-decode batch, increment each sequence by 1."""
        si = _make_seq_info()
        self._setup_decode_batch(
            si,
            positions=[5, 10],
            swc_vals=[6, 11],
            pages_per_seq=[2, 3],
            cache_loc=[10, 11, 20, 21, 22],
        )

        increment = torch.tensor([1, 1], dtype=torch.int32)
        si.switch_to_generate_(increment)

        pos = si._input_buffer.get_view("position_ids")
        assert pos[0].item() == 6
        assert pos[1].item() == 11

        swc = si._input_buffer.get_view("seq_len_with_cache")
        assert swc[0].item() == 7
        assert swc[1].item() == 12

        bi = si._input_buffer.get_host_view("batch_info")
        assert bi[:6].tolist() == [0, 0, 0, 0, 2, 2]

    def test_decode_with_varying_increment(self):
        """Non-uniform increments (e.g. after speculative decoding acceptance)."""
        si = _make_seq_info()
        self._setup_decode_batch(
            si,
            positions=[5, 10],
            swc_vals=[6, 11],
            pages_per_seq=[2, 3],
            cache_loc=[10, 11, 20, 21, 22],
        )

        increment = torch.tensor([3, 1], dtype=torch.int32)
        si.switch_to_generate_(increment)

        pos = si._input_buffer.get_view("position_ids")
        assert pos[0].item() == 8
        assert pos[1].item() == 11

        swc = si._input_buffer.get_view("seq_len_with_cache")
        assert swc[0].item() == 9
        assert swc[1].item() == 12


class TestSwitchToGeneratePageBoundary:
    """Page boundary crossing scenarios."""

    def test_page_crossing_updates_metadata(self):
        """Increment that crosses a page boundary updates pages_per_seq and cu_num_pages."""
        si = _make_seq_info()
        # seq_len_with_cache = 8, tokens_per_block = 4 -> pages = 2, last_page_len = 4 (full)
        # After +1: swc = 9 -> pages = 3, last_page_len = 1
        self._setup_decode_at_page_boundary(si)

        increment = torch.tensor([1], dtype=torch.int32)
        si.switch_to_generate_(increment)

        swc = si._input_buffer.get_view("seq_len_with_cache")
        assert swc[0].item() == 9

        pps = si._input_buffer.get_view("pages_per_seq")
        assert pps[0].item() == 3

        lpl = si._input_buffer.get_view("last_page_len")
        assert lpl[0].item() == 1

        cnp = si._input_buffer.get_view("cu_num_pages")
        assert cnp[0].item() == 0
        assert cnp[1].item() == 3

    def test_page_crossing_with_extra_page_inserts_into_cache_loc(self):
        """Page boundary crossing with active extra_page_per_seq inserts the page."""
        si = _make_seq_info()
        si.nest_sequences(
            input_ids=[[1]],
            position_ids=[[7]],
            input_pos=[7],
            seq_len=[1],
            batch_info=[0, 0, 0, 0, 1, 1],
            pages_per_seq=[2],
            cache_loc=[10, 11],
            seq_len_with_cache=[8],
            extra_page_per_seq=[99],
        )

        increment = torch.tensor([1], dtype=torch.int32)
        si.switch_to_generate_(increment)

        cl = si._input_buffer.get_view("cache_loc")
        assert cl[0].item() == 10
        assert cl[1].item() == 11
        assert cl[2].item() == 99

    @staticmethod
    def _setup_decode_at_page_boundary(si):
        """Set up a single decode sequence at swc=8 (full page boundary)."""
        si.nest_sequences(
            input_ids=[[1]],
            position_ids=[[7]],
            input_pos=[7],
            seq_len=[1],
            batch_info=[0, 0, 0, 0, 1, 1],
            pages_per_seq=[2],
            cache_loc=[10, 11],
            seq_len_with_cache=[8],
        )


class TestSwitchToGenerateHostArgHandling:
    """Validate host arg warning and d2h sync behavior."""

    def test_non_native_host_arg_syncs_device_to_host(self):
        """Activating a non-native host arg should sync device -> host instead of raising."""
        si = _make_seq_info()
        si.activate_arg("position_ids_host")
        _nest_prefill(
            si,
            input_ids=[[1, 2, 3]],
            pages_per_seq=[1],
            cache_loc=[0],
        )

        increment = torch.tensor([1], dtype=torch.int32)
        si.switch_to_generate_(increment)

        # Device was updated: last pos was 2, +1 = 3
        device_pos = si._input_buffer.get_view("position_ids")
        assert device_pos[0].item() == 3

        # Host should now be synced (d2h sync triggered by position_ids_host being active)
        host_pos = si._input_buffer.get_host_view("position_ids")
        assert host_pos[0].item() == 3

    def test_non_native_host_seq_len_with_cache_syncs(self):
        """seq_len_with_cache_host should get synced from device."""
        si = _make_seq_info()
        si.activate_arg("seq_len_with_cache_host")
        _nest_prefill(
            si,
            input_ids=[[1, 2, 3], [4, 5, 6, 7]],
            pages_per_seq=[1, 1],
            cache_loc=[10, 20],
        )

        increment = torch.tensor([1, 1], dtype=torch.int32)
        si.switch_to_generate_(increment)

        # Device: swc was [3, 4], +1 = [4, 5]
        device_swc = si._input_buffer.get_view("seq_len_with_cache")
        assert device_swc[0].item() == 4
        assert device_swc[1].item() == 5

        # Host should be synced
        host_swc = si._input_buffer.get_host_view("seq_len_with_cache")
        assert host_swc[0].item() == 4
        assert host_swc[1].item() == 5

    def test_native_host_args_do_not_raise(self):
        """batch_info_host, tokens_gather_info_host, cu_seqlen_host, seq_len_host are native."""
        si = _make_seq_info()
        _nest_prefill(
            si,
            input_ids=[[1, 2]],
            pages_per_seq=[1],
            cache_loc=[0],
        )

        increment = torch.tensor([1], dtype=torch.int32)
        si.switch_to_generate_(increment)


class TestSwitchToGenerateOnlyUpdatesActiveArgs:
    """Verify that inactive args are not touched."""

    def test_inactive_input_pos_not_modified(self):
        """If input_pos is not activated, its buffer should not change."""
        si = SequenceInfo(
            max_seq_len=MAX_SEQ_LEN,
            max_batch_size=MAX_BATCH_SIZE,
            tokens_per_block=TOKENS_PER_BLOCK,
            max_num_tokens=MAX_NUM_TOKENS,
        )
        for arg in _CACHED_ARGS:
            if arg != "input_pos":
                si.activate_arg(arg)

        _nest_prefill(si, input_ids=[[1, 2, 3]], pages_per_seq=[1], cache_loc=[0])

        original_ip = si._input_buffer.get_view("input_pos")[:1].clone()

        increment = torch.tensor([1], dtype=torch.int32)
        si.switch_to_generate_(increment)

        assert si._input_buffer.get_view("input_pos")[0].item() == original_ip[0].item()

    def test_no_device_to_host_sync_for_position_ids(self):
        """Device position_ids should be updated but host buffer should NOT be synced."""
        si = _make_seq_info()
        _nest_prefill(
            si,
            input_ids=[[1, 2, 3]],
            pages_per_seq=[1],
            cache_loc=[0],
        )
        host_pos_before = si._input_buffer.get_host_view("position_ids")[:3].clone()

        increment = torch.tensor([1], dtype=torch.int32)
        si.switch_to_generate_(increment)

        # Device should be updated: last pos was 2, +1 = 3
        device_pos = si._input_buffer.get_view("position_ids")
        assert device_pos[0].item() == 3

        # Host should still have the old packed values from nest_sequences
        host_pos_after = si._input_buffer.get_host_view("position_ids")[:3]
        assert host_pos_after.tolist() == host_pos_before.tolist()
