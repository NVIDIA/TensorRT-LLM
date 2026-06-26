# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from unittest.mock import patch

import pytest
import torch
from test_triton_mamba_cached_op import _random_params

import tensorrt_llm._torch.auto_deploy  # noqa: F401
from tensorrt_llm._torch.auto_deploy._compat import KvCacheConfig
from tensorrt_llm._torch.auto_deploy.custom_ops.attention_interface import (
    BatchInfo,
    CausalConvResourceHandler,
    IntermediateConvStateHandler,
    ReplayCacheBufIdxHandler,
    ReplayNWritesHandler,
    ReplayOldBHandler,
    ReplayOldDAcumsumHandler,
    ReplayOldDtHandler,
    ReplayOldXHandler,
    ReplayPrevNumAcceptedHandler,
    ReplayWorkItemsHandler,
    SSMResourceHandler,
)
from tensorrt_llm._torch.auto_deploy.shim.interface import CachedSequenceInterface
from tensorrt_llm._torch.modules.mamba.mamba2_metadata import (
    REPLAY_WORK_CACHE_BUF_IDX,
    REPLAY_WORK_CACHE_SLOT,
    REPLAY_WORK_ITEM_WIDTH,
    REPLAY_WORK_PNAT,
    REPLAY_WORK_POSITION_IN_DECODE_BATCH,
)
from tensorrt_llm._torch.modules.mamba.replay_selective_state_update import (
    replay_selective_state_update as _real_replay_selective_state_update,
)


@pytest.fixture
def mamba_env():
    device = "cuda"
    dtype = torch.bfloat16
    atol = 1e-3
    rtol = 1e-3
    torch.manual_seed(42)
    torch.cuda.empty_cache()
    return {"device": device, "dtype": dtype, "atol": atol, "rtol": rtol}


def test_flashinfer_decode_matches_triton(mamba_env):
    device = mamba_env["device"]
    dtype = mamba_env["dtype"]
    atol = mamba_env["atol"]
    rtol = mamba_env["rtol"]

    batch, seq = 2, 1
    num_heads, head_dim = 2, 64
    n_groups, ssm_state_size = 2, 64
    (hidden_states, A, B, C, D, dt, dt_bias, time_step_limit, chunk_size) = _random_params(
        device, dtype, batch, seq, num_heads, head_dim, n_groups, ssm_state_size
    )

    max_batch_size = 4
    slot_idx = torch.tensor([0, 2], device=device, dtype=torch.int32)
    ssm_state_cache_triton = torch.randn(
        max_batch_size, num_heads, head_dim, ssm_state_size, device=device, dtype=dtype
    )
    ssm_state_cache_flashinfer = ssm_state_cache_triton.clone()

    _bi = BatchInfo()
    _bi.update([0, 0, 0, 0, batch, batch])
    batch_info_host = _bi.serialize()
    cu_seqlen = torch.zeros(batch + 1, device=device, dtype=torch.int32)
    use_initial_states = torch.zeros(batch, device=device, dtype=torch.bool)

    any_prefill_use_initial_states_host = torch.tensor([False], device=device, dtype=torch.bool)
    y_triton = torch.ops.auto_deploy.triton_cached_ssm(
        hidden_states,
        A,
        B,
        C,
        D,
        dt,
        dt_bias,
        # STANDARD METADATA
        batch_info_host,
        cu_seqlen,
        slot_idx,
        use_initial_states,
        any_prefill_use_initial_states_host,
        # EXTRA METADATA
        None,  # chunk indices
        None,  # chunk offsets
        None,  # seq_idx_prefill
        # CACHES
        ssm_state_cache_triton,
        None,
        # CONSTANTS
        time_step_limit,
        chunk_size,
    )

    y_flashinfer = torch.ops.auto_deploy.flashinfer_cached_ssm(
        hidden_states,
        A,
        B,
        C,
        D,
        dt,
        dt_bias,
        # STANDARD METADATA
        batch_info_host,
        cu_seqlen,
        slot_idx,
        use_initial_states,
        any_prefill_use_initial_states_host,
        # EXTRA METADATA
        None,  # chunk indices
        None,  # chunk offsets
        None,  # seq_idx_prefill
        # CACHES
        ssm_state_cache_flashinfer,
        None,  # intermediate_ssm_state_cache
        None,  # replay_old_x
        None,  # replay_old_b
        None,  # replay_old_dt
        None,  # replay_old_da_cumsum
        None,  # replay_cache_buf_idx
        None,  # replay_prev_num_accepted
        None,  # replay_work_items
        None,  # replay_n_writes
        # CONSTANTS
        time_step_limit,
        chunk_size,
    )

    assert y_triton.shape == hidden_states.shape
    assert y_flashinfer.shape == hidden_states.shape
    assert torch.isfinite(y_flashinfer).all()
    assert torch.allclose(y_flashinfer, y_triton.to(y_flashinfer.dtype), atol=atol, rtol=rtol)

    after_triton = ssm_state_cache_triton.index_select(0, slot_idx)
    after_flashinfer = ssm_state_cache_flashinfer.index_select(0, slot_idx)
    assert torch.allclose(
        after_flashinfer.to(after_triton.dtype), after_triton, atol=atol, rtol=rtol
    )


@pytest.mark.parametrize(
    "head_dim",
    [
        pytest.param(32, id="unsupported_flashinfer_headdim"),  # not in {64, 128}
        pytest.param(64, id="supported_flashinfer_headdim"),  # in {64, 128}
    ],
)
def test_flashinfer_extend_replay_calls_replay_kernel(mamba_env, head_dim):
    """Verify replay_selective_state_update is invoked on the extend+replay path.

    Parametrized over head_dim to cover both a dim that FlashInfer does not support
    (proving replay is independent of the FlashInfer head-dim constraint) and one
    that it does support (ensuring the path works for production-like configs too).
    """
    device = mamba_env["device"]
    dtype = mamba_env["dtype"]

    num_extend = 1
    tokens_per_extend = 2  # draft tokens per MTP verification step
    num_heads = 4
    n_groups, ssm_state_size = 2, 16  # ssm_state_size >= 16 for replay tl.dot
    max_batch_size = 2  # cache slots

    (hidden_states, A, B, C, D, dt, dt_bias, time_step_limit, chunk_size) = _random_params(
        device, dtype, num_extend, tokens_per_extend, num_heads, head_dim, n_groups, ssm_state_size
    )

    ssm_state_cache = torch.zeros(
        max_batch_size, num_heads, head_dim, ssm_state_size, device=device, dtype=dtype
    )
    slot_idx = torch.tensor([0], device=device, dtype=torch.int32)

    # Replay buffers: all zeros (first step, nothing cached yet; kernel still runs).
    replay_history_size = 16
    replay_old_x = torch.zeros(
        max_batch_size,
        2,
        replay_history_size,
        num_heads,
        head_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    replay_old_b = torch.zeros(
        max_batch_size,
        2,
        replay_history_size,
        n_groups,
        ssm_state_size,
        device=device,
        dtype=torch.bfloat16,
    )
    replay_old_dt = torch.zeros(
        max_batch_size, 2, num_heads, replay_history_size, device=device, dtype=torch.float32
    )
    replay_old_da_cumsum = torch.zeros(
        max_batch_size, 2, num_heads, replay_history_size, device=device, dtype=torch.float32
    )
    replay_cache_buf_idx = torch.zeros(max_batch_size, device=device, dtype=torch.int32)
    replay_prev_num_accepted = torch.zeros(max_batch_size, device=device, dtype=torch.int32)
    replay_work_items = torch.zeros(
        max_batch_size, REPLAY_WORK_ITEM_WIDTH, device=device, dtype=torch.int32
    )
    replay_work_items[0, REPLAY_WORK_POSITION_IN_DECODE_BATCH] = 0
    replay_work_items[0, REPLAY_WORK_CACHE_SLOT] = slot_idx[0]
    replay_work_items[0, REPLAY_WORK_PNAT] = 0
    replay_work_items[0, REPLAY_WORK_CACHE_BUF_IDX] = 0
    replay_n_writes = torch.zeros(1, device=device, dtype=torch.int32)

    # Extend-only batch with replay mode enabled.
    _bi = BatchInfo()
    _bi.update([0, 0, num_extend, num_extend * tokens_per_extend, 0, 0])
    _bi.update_use_replay(True)
    batch_info_host = _bi.serialize()

    cu_seqlen = torch.tensor([0, tokens_per_extend], device=device, dtype=torch.int32)
    use_initial_states = torch.zeros(num_extend, device=device, dtype=torch.bool)
    any_prefill_use_initial_states_host = torch.tensor([False], device=device, dtype=torch.bool)

    _TARGET = (
        "tensorrt_llm._torch.auto_deploy.custom_ops.mamba"
        ".flashinfer_backend_mamba.replay_selective_state_update"
    )
    with patch(_TARGET, wraps=_real_replay_selective_state_update) as mock_replay:
        out = torch.ops.auto_deploy.flashinfer_cached_ssm(
            hidden_states,
            A,
            B,
            C,
            D,
            dt,
            dt_bias,
            # STANDARD METADATA
            batch_info_host,
            cu_seqlen,
            slot_idx,
            use_initial_states,
            any_prefill_use_initial_states_host,
            # EXTRA METADATA
            None,
            None,
            None,  # chunk_indices, chunk_offsets, seq_idx_prefill
            # CACHES
            ssm_state_cache,
            None,  # intermediate_ssm_state_cache (None in replay mode)
            replay_old_x,
            replay_old_b,
            replay_old_dt,
            replay_old_da_cumsum,
            replay_cache_buf_idx,
            replay_prev_num_accepted,
            replay_work_items,
            replay_n_writes,
            # CONSTANTS
            time_step_limit,
            chunk_size,
        )

    assert mock_replay.call_count == num_extend, (
        f"Expected {num_extend} replay call(s), got {mock_replay.call_count}"
    )
    assert out.shape == hidden_states.shape
    assert torch.isfinite(out).all()


class _SpecDecModeForReplayTest:
    def use_one_engine(self):
        return True


class _SpecConfigForReplayTest:
    def __init__(self, max_draft_len: int):
        self.max_draft_len = max_draft_len
        self.tokens_per_gen_step = max_draft_len + 1
        self.spec_dec_mode = _SpecDecModeForReplayTest()


def _build_interface_with_replay_buffers(num_heads, head_dim, d_state, n_groups, max_batch_size):
    """Allocate replay buffers through the real production path (CachedSequenceInterface).

    Registers the Mamba + replay-buffer resource bundle for one layer and runs
    initialize_resources(), which is where the cache-manager-bound replay
    work-items buffer (interface._replay_work_items -- the tensor the replay SSM
    kernel actually reads) is allocated.
    """
    conv_dim = head_dim * num_heads + 2 * n_groups * d_state
    interface = CachedSequenceInterface(
        max_seq_len=128,
        max_batch_size=max_batch_size,
        max_num_tokens=(128 + 1) * max_batch_size,
        device="cuda",
        kv_cache_config=KvCacheConfig(
            tokens_per_block=32, max_tokens=1024, free_gpu_memory_fraction=0.0
        ),
        spec_config=_SpecConfigForReplayTest(max_draft_len=2),
    )
    interface.add_resource(
        "ssm_state_0",
        SSMResourceHandler(
            num_heads=num_heads, head_dim=head_dim, d_state=d_state, dtype=torch.bfloat16
        ),
    )
    interface.add_resource(
        "conv_state_0", CausalConvResourceHandler(conv_dim=conv_dim, d_conv=4, dtype=torch.float32)
    )
    interface.add_resource(
        "intermediate_conv_state_0",
        IntermediateConvStateHandler(conv_dim=conv_dim, d_conv=4, dtype=torch.float32),
    )
    interface.add_resource(
        "replay_old_x_0",
        ReplayOldXHandler(num_heads=num_heads, head_dim=head_dim, dtype=torch.bfloat16),
    )
    interface.add_resource(
        "replay_old_B_0",
        ReplayOldBHandler(n_groups=n_groups, d_state=d_state, dtype=torch.bfloat16),
    )
    interface.add_resource("replay_old_dt_0", ReplayOldDtHandler(num_heads=num_heads))
    interface.add_resource("replay_old_dA_cumsum_0", ReplayOldDAcumsumHandler(num_heads=num_heads))
    interface.add_resource("replay_cache_buf_idx_0", ReplayCacheBufIdxHandler())
    interface.add_resource("replay_prev_num_accepted_0", ReplayPrevNumAcceptedHandler())
    interface.add_resource("replay_work_items_0", ReplayWorkItemsHandler())
    interface.add_resource("replay_n_writes_0", ReplayNWritesHandler())
    return interface


def test_extend_replay_init_buffers(mamba_env):
    """The replay path must not cause an out-of-bounds access on the replay buffers.

    Behavioral guard for the replay path: every buffer the prepare hook populates (the
    work-items buffer and the n-writes count) is filled with garbage (out-of-bounds
    values, simulating uninitialized memory), then the production metadata-prep path runs
    and the real replay op executes; the test asserts no CUDA fault. With the fix, prep
    populates the buffers before the kernel reads them, so the garbage never reaches the
    kernel; without it the out-of-bounds values survive and fault.

    Filling the buffers directly keeps the poison confined to them and makes the failure
    deterministic: fresh CUDA memory is often benign, so a poison-free run cannot
    reliably reproduce the bug.
    """
    device = mamba_env["device"]
    dtype = mamba_env["dtype"]

    # Production SuperV3 Mamba2 shape (AutoDeploy replicates mamba -> full heads/groups),
    # large enough that the replay kernel runs its persistent_main path, which reads the
    # cache slot from the replay work-items buffer.
    num_extend = 8
    tokens_per_extend = 7  # num_nextn_predict_layers (6) + 1
    num_heads = 128
    head_dim = 64
    n_groups, ssm_state_size = 8, 128

    interface = _build_interface_with_replay_buffers(
        num_heads, head_dim, ssm_state_size, n_groups, max_batch_size=num_extend
    )
    interface.initialize_resources()

    # Poison every buffer the prepare hook populates -- the work-items buffer and the
    # n-writes count -- with out-of-bounds values, simulating garbage / uninitialized
    # memory. The production metadata-prep below must overwrite them before the kernel
    # reads them; if prep is missing (the bug) the poison survives and faults.
    interface._replay_work_items.fill_(0x7FFFFFFF)  # int32-max: out-of-bounds cache slot
    interface._replay_n_writes.fill_(0x7FFFFFFF)  # int32-max: out-of-bounds write count

    # Drive the production metadata-prep path -- the same one cudagraph capture uses --
    # so the replay work-items / n-writes buffers are populated exactly as in real runs
    # (set_capture_batch -> nest_sequences -> prepare_replay_metadata host-prepare hook).
    interface.info.set_capture_batch(max_draft_len=tokens_per_extend - 1, batch_size=num_extend)
    replay_work_items = interface._replay_work_items
    replay_n_writes = interface._replay_n_writes

    # Per-token inputs and the remaining replay caches for the same extend batch.
    (hidden_states, A, B, C, D, dt, dt_bias, time_step_limit, chunk_size) = _random_params(
        device, dtype, num_extend, tokens_per_extend, num_heads, head_dim, n_groups, ssm_state_size
    )
    ssm_state_cache = torch.zeros(
        num_extend, num_heads, head_dim, ssm_state_size, device=device, dtype=dtype
    )
    slot_idx = torch.arange(num_extend, device=device, dtype=torch.int32)

    replay_history_size = 16
    replay_old_x = torch.zeros(
        num_extend, 2, replay_history_size, num_heads, head_dim, device=device, dtype=torch.bfloat16
    )
    replay_old_b = torch.zeros(
        num_extend,
        2,
        replay_history_size,
        n_groups,
        ssm_state_size,
        device=device,
        dtype=torch.bfloat16,
    )
    replay_old_dt = torch.zeros(
        num_extend, 2, num_heads, replay_history_size, device=device, dtype=torch.float32
    )
    replay_old_da_cumsum = torch.zeros(
        num_extend, 2, num_heads, replay_history_size, device=device, dtype=torch.float32
    )
    replay_cache_buf_idx = torch.zeros(num_extend, device=device, dtype=torch.int32)
    replay_prev_num_accepted = torch.zeros(num_extend, device=device, dtype=torch.int32)

    _bi = BatchInfo()
    _bi.update([0, 0, num_extend, num_extend * tokens_per_extend, 0, 0])
    _bi.update_use_replay(True)
    batch_info_host = _bi.serialize()
    cu_seqlen = torch.arange(
        0, (num_extend + 1) * tokens_per_extend, tokens_per_extend, device=device, dtype=torch.int32
    )
    use_initial_states = torch.zeros(num_extend, device=device, dtype=torch.bool)
    any_prefill_use_initial_states_host = torch.tensor([False], device=device, dtype=torch.bool)

    out = torch.ops.auto_deploy.flashinfer_cached_ssm(
        hidden_states,
        A,
        B,
        C,
        D,
        dt,
        dt_bias,
        # STANDARD METADATA
        batch_info_host,
        cu_seqlen,
        slot_idx,
        use_initial_states,
        any_prefill_use_initial_states_host,
        # EXTRA METADATA
        None,
        None,
        None,  # chunk_indices, chunk_offsets, seq_idx_prefill
        # CACHES
        ssm_state_cache,
        None,  # intermediate_ssm_state_cache (None in replay mode)
        replay_old_x,
        replay_old_b,
        replay_old_dt,
        replay_old_da_cumsum,
        replay_cache_buf_idx,
        replay_prev_num_accepted,
        replay_work_items,
        replay_n_writes,
        # CONSTANTS
        time_step_limit,
        chunk_size,
    )

    # Synchronize so any out-of-bounds access on the replay buffers surfaces here as a
    # CUDA error rather than asynchronously later.
    torch.cuda.synchronize()
    assert out.shape == hidden_states.shape
    assert torch.isfinite(out).all()
