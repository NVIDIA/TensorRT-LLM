# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The DeepSeek-V4.1 targets' Engram table: the engine's `ShardedFp8MultiHeadEmbedding`
(`_torch/modules/engram/engram.py`), FP8 or MXFP4 rows with UE8M0 scales in host memory, read by the GPU through UVA.

The host allocation (NUMA placement, page-cache eviction and its `TRTLLM_ENGRAM_EVICT_PAGE_CACHE=0` opt-out, the
pinned-memory fallbacks), the node-wide sharing of one copy between the processes of a node (with its 30-minute owner
deadline) and the loader marker are the engine's (`_host_memory.py`, `_node_share.py`, `_evict_page_cache.py`). This
subclass changes two things:
- `forward` gathers through the family op `ops/engram/engram_lookup.py` (the engine's kernel, `compile_engram_lookup`);
- `_load_rows` loads exactly the shard's checkpoint rows and width: the engine accepts a short final row shard (a
  reachable row left as padding) and a narrower scale (which `copy_` broadcasts into every block of its row).
"""

import json
from pathlib import Path
from typing import Dict, Optional

import torch

from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.engram.engram_lookup import (
    engram_lookup,
)
from tensorrt_llm._torch.modules.engram.engram import (
    ShardedFp8MultiHeadEmbedding as _EngineShardedFp8MultiHeadEmbedding,
)
from tensorrt_llm._torch.modules.engram.engram import _multi_processor_count
from tensorrt_llm.logger import logger


def table_share_key(checkpoint_dir: Optional[str], layer_id: int) -> Optional[str]:
    """Identify a layer's Engram table by the checkpoint file that holds it: path, device, inode, size and mtime;
    None (a private table) if the checkpoint index does not name that file. Stock's
    ``modeling_deepseekv41._engram_table_share_key``: processes on a node share one copy only of the same file."""
    if not checkpoint_dir:
        return None
    root = Path(checkpoint_dir)
    name = f"layers.{layer_id}.engram.embed.weight"
    try:
        with open(root / "model.safetensors.index.json") as index:
            shard = json.load(index).get("weight_map", {}).get(name)
        if shard is None:
            return None
        path = (root / shard).resolve(strict=True)
        stat = path.stat()
    except (OSError, ValueError) as error:
        logger.debug(f"Engram layer {layer_id} keeps a private table: {error}")
        return None
    # Device and inode tell apart different files mounted at one path.
    return f"{path}:{stat.st_dev}:{stat.st_ino}:{stat.st_size}:{stat.st_mtime_ns}:layer{layer_id}"


def engram_table_mxfp4(cfg, prefix: str) -> bool:
    """Whether the Engram table under checkpoint prefix `prefix` (`layers.<N>.engram.`) is MXFP4: stock's
    `engram_table_is_mxfp4` on the checkpoint's top-level `quantization_config` (`engram_embed_quant == "mxfp4"`), which
    validates its block size, scale type and layer list."""
    from tensorrt_llm._torch.configs.deepseek_v41 import engram_table_is_mxfp4

    quant = getattr(cfg, "_v41_quantization_config", None)
    if quant is None:
        quant = getattr(cfg, "quantization_config", None)
    if quant is not None and not isinstance(quant, dict):
        quant = quant.to_dict() if hasattr(quant, "to_dict") else dict(quant)
    return engram_table_is_mxfp4(quant, int(prefix.split(".")[1]))


class ShardedFp8MultiHeadEmbedding(_EngineShardedFp8MultiHeadEmbedding):
    """The engine's Engram table (see the module docstring), looked up through the family op and loaded only from
    complete checkpoint rows."""

    def _load_rows(self, weights: Dict) -> None:
        start, end = self.vocab_start_idx, self.vocab_end_idx
        for name, param in (("weight", self.weight), ("scale", self.scale)):
            src = weights[name]
            shard = src[start:end]
            if not isinstance(shard, torch.Tensor):
                shard = torch.as_tensor(shard)
            # Every row of [start, end) with the table's width: a short checkpoint would leave a reachable row as
            # padding, and a narrower scale would broadcast into every block of its row in copy_.
            expected = (end - start, *param.shape[1:])
            if tuple(shard.shape) != expected:
                raise ValueError(
                    f"Engram {name}: rank {self.tp_rank}/{self.tp_size} expected rows [{start}, {end}) of a "
                    f"{self.num_embeddings}-row table, shape {expected}, but the checkpoint tensor "
                    f"yielded {tuple(shard.shape)}."
                )
            rows = shard.shape[0]
            param.data[:rows].copy_(shard.to(param.dtype))
            # Only the final row shard is short (the ceil-divide remainder); valid hashes cannot reach its padding.
            param.data[rows:].zero_()

    def forward(self, input_ids: torch.Tensor, background: bool = False) -> torch.Tensor:
        """``[T, num_heads]`` bucket-local indices -> ``[T, num_local_heads, D]``.

        Head shards return only owned heads; row shards return all heads with
        unowned rows zeroed. The caller orders the matching collective after the
        lookup stream. ``background`` caps side-stream occupancy to leave SMs for the main stream.
        """
        num_rows = input_ids.shape[0] * self.num_local_heads
        out = torch.empty(
            (input_ids.shape[0], self.num_local_heads, self.embedding_dim),
            dtype=self.dtype,
            device=input_ids.device,
        )
        if num_rows == 0:
            return out
        weight, scale = self.storage()
        num_sms = _multi_processor_count(input_ids.device.index)
        engram_lookup(
            weight.view(torch.uint8),
            scale.view(torch.uint8),
            input_ids,
            self.offsets,
            out,
            head_start=self.head_start,
            block_size=self.block_size,
            # Hashes must stay within each head's bucket. With rebased offsets,
            # head shards own every lookup and need no ownership mask.
            masked=not self.shard_heads,
            vocab_start=self.vocab_start_idx,
            vocab_end=self.vocab_end_idx,
            num_ctas=max(1, num_sms // 2) if background else num_sms,
            mxfp4=self.mxfp4,
        )
        return out
