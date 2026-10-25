# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Peer memory of the step program's tensor-parallel collectives, IPC-mapped across the ranks'
processes through the runtime's IpcMemory (one allocation per rank, every rank's pointer opened
in every process; zero-filled):

- the Lamport all-reduce (one-shot / two-shot kernels over NVLink: data arrival is the
  synchronization, receive buffers pre-filled with a sentinel), which the mHC boundaries, the
  MoE combine, the MTP chain's norms and the vocabulary-parallel head's key max fuse into
  themselves; messages above its capacity go to NCCL;
- the sequence-parallel boundaries of large steps: every rank's layer-input rows x and
  sublayer-output rows out ([rows, hidden] bf16, exposed as this rank's tensors), the step
  program's IO, with the arrival flags of the reduce-scatter / all-gather;
- the token-split prefill attention's flags (it exchanges rows through the same x / out).

All ranks must issue the same sequence of steps: the buffers rotate by per-call epochs kept on
the device, which a CUDA graph replays."""

import torch

from ._lib import ops, stream

AR_MAX_BLOCKS = 128


class Collectives:
    def __init__(self, mapping, rows: int, hidden: int, ar_max_tokens: int = 1024):
        from tensorrt_llm._ipc_utils import IpcMemory
        from tensorrt_llm._utils import TensorWrapper, convert_to_torch_tensor

        self.tp = mapping.tp_size
        self.rank = mapping.tp_rank
        assert 2 <= self.tp <= ops.max_tp
        # all-reduce: receive buffers, flags, the head's key slots; this rank's epochs (+ the
        # Lamport state at [AR_MAX_BLOCKS], + the key max-reduce state)
        self.ar_max_elems = ar_max_tokens * hidden
        self._ar_buf = IpcMemory(mapping, ops.allreduce_buf_elems(self.tp, self.ar_max_elems) * 2)
        self._ar_flags = IpcMemory(mapping, 4 * self.tp * AR_MAX_BLOCKS * 4)
        self._ar_keys = IpcMemory(mapping, 3 * self.tp * ops.max_key_rows * 8)
        self.ar_peers = ops.ArPeers(
            list(self._ar_buf.peer_ptrs),
            list(self._ar_flags.peer_ptrs),
            list(self._ar_keys.peer_ptrs),
        )
        self.ar_epochs = torch.zeros(AR_MAX_BLOCKS + 8, dtype=torch.int32, device="cuda")
        # (the Lamport receive buffers start sentinel-filled, on every rank before any rank pushes)
        ops.allreduce_lamport_init(self._ar_buf.local_ptr, self.tp, self.ar_max_elems, stream())
        # sequence-parallel boundaries and the token-split attention
        nbytes = rows * hidden * 2
        self._out = IpcMemory(mapping, nbytes)
        self._x = IpcMemory(mapping, nbytes)
        self._sp_flags = IpcMemory(mapping, ops.sp_flag_words * 4)
        self._ts_flags = IpcMemory(mapping, ops.ts_flag_words * 4)
        self.sp_peers = ops.SpPeers(
            list(self._out.peer_ptrs), list(self._x.peer_ptrs), list(self._sp_flags.peer_ptrs)
        )
        self.ts_peers = ops.TsPeers(
            list(self._out.peer_ptrs), list(self._x.peer_ptrs), list(self._ts_flags.peer_ptrs)
        )
        self.sp_state = torch.zeros(ops.sp_state_words, dtype=torch.int32, device="cuda")
        self.ts_state = torch.zeros(ops.ts_state_words, dtype=torch.int32, device="cuda")
        self.x = convert_to_torch_tensor(
            TensorWrapper(self._x.local_ptr, torch.bfloat16, [rows, hidden])
        )
        self.out = convert_to_torch_tensor(
            TensorWrapper(self._out.local_ptr, torch.bfloat16, [rows, hidden])
        )
        torch.cuda.synchronize()
        _barrier(mapping)


def _barrier(mapping) -> None:
    from tensorrt_llm._torch.distributed.communicator import Distributed

    Distributed.get(mapping).tp_barrier()
