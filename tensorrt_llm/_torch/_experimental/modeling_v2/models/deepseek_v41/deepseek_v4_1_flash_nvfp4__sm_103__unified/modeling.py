# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ModelingV2 target: deepseek-v4.1-flash-nvfp4 / sm_103 / unified {tep4, dep4, tp4, tep8, dep8}.

The core holds what the model statically is: geometry, weights, the pool views
and every table derived after load. A ``PrefillTarget`` and a ``DecodeTarget``
each run one step over it (``_target.py``), and the core routes every step by
``phase_of``: decode when the step has no context request, prefill otherwise
(a mixed batch is prefill's). Every trtllm op a target calls is an entry -- the
shared ``catalog/`` or this target's own ``ops/`` -- instantiated by the target
and bound once to its weights; native torch is called directly.

DeepSeek-V4.1-Flash (``DeepseekV41ForCausalLM``), text path: 40 blocks of
hidden 5120 over a 4-copy hyper-connection stream (``hc_mult`` 4), each block
an attention sublayer and a MoE sublayer (384 routed NVFP4 experts at top-6
plus one FP8 shared expert), with Engram n-gram memory added into the stream
before blocks 1 and 14.

**Images (the checkpoint's ``vision.py`` / ``model.py`` VL path).** A context
chunk of a request that carries images (``multimodal_params``, built by the
``deepseek_v41`` input processor) takes its image-span rows from
``VisionTower``: the 32-block DeepSeek-ViT over each image's 14x14 patches
(bidirectional attention within one image), the 3x3 aligner and the learned
span delimiters. Those rows replace the token embedding at the span positions
(every span position carries id 129264). On the same rows the MoE gate selects
experts with ``gate.bias_vl`` instead of ``gate.bias`` (two routing calls and a
per-token select), and Engram treats them as DEAD: no n-gram hashes across
them and their Engram gate is 0. Generated tokens are text. The tower is
replicated on every rank and runs one image per call, only for the images a
chunk's rows reach; the context step itself is never captured, but each tower
call replays a CUDA graph captured at load for its padded patch count
(``TRTLLM_DSV41_VIT_CUDA_GRAPH``, unset = on; see ``VisionTower``).

**DSpark (``speculative_config`` DSpark, block 5).** The
checkpoint's three ``mtp.*`` stages are a block drafter, not chained MTP heads.
With it on, the trunk hands the runtime's ``DSparkSpecMetadata`` the INPUT
stream of layers 37-39 (the metadata averages the 4 hc copies), generation
rows carry 1 + 5 tokens, and the shell drives ``DSv4DSparkWorker`` with
``DSparkDraftModel`` below: per step it writes the bonus position's latent into
each stage's 128-row ring (the worker owns the rings), runs the three stages
over the 5-row block ``[bonus, noise x 4]`` -- each stage's attention reads its
ring plus the whole block, non-causal -- projects through the shell's
``lm_head`` and applies the sequential Markov correction with greedy feedback.
The drafter is replicated on every rank of every member and has no collective
of its own; its logits leave the vocab-parallel ``lm_head`` gathered, so every
TP rank drafts the same tokens. Its one autotuned op, the routed MoE, runs in
the autotuner's ``fallback_tactic()`` scope, so every rank takes the same
untuned tactic whatever its tuner cache holds. Verification is full-block
(static): the confidence head is not computed.

**Encoder SWA bounded replay (switch ``TRTLLM_DSV41_ENCODER_SWA_BOUNDED_REPLAY``,
resolved by the V4.1 cache manager: unset means on under attention TP --
tep4, tp4 -- aggregated and without a KV connector, off elsewhere).** The
prefix cache keeps main KV and index K only. A hit of ``reused`` tokens (a
multiple of 2, the compress ratio) starts the request's first chunk
``R = min(128, reused)`` positions early: those R replay rows run every layer
and rebuild the sliding windows (a query at ``p`` attends window keys
``[max(reused - R, p - 127), p]`` and nothing below the replay start: the
context attention takes its window counts from the metadata's
``v41_swa_seq_lens_cuda``, kv length minus the SWA floor, since the kernel
reads a -1 inside the count as a zero K/V row, not as a mask), and they write
nothing else. The
compressors of the KV sources run over the other rows only, packed as a step
that starts at the reuse boundary, so no cached main-KV or index-K entry is
recomputed or overwritten; the Engram history of the three positions before
the replay start is stored first, from the prompt ids. A context request's
last row is always a suffix row, so replay rows produce no logits. Under
attention DP (dep4) each rank replays the hits of the requests it holds; its
replay rows are rows of its step like any other. Disaggregated serving never
replays (the manager refuses an explicit 1).

**Decoder SWA bounded replay (switch ``TRTLLM_DSV41_DECODER_SWA_BOUNDED_REPLAY``).**
Layers ``0 .. L`` -- ``L`` the last KV-source layer, 20, whose ratio-1
compressor and index K are the decoder's global KV -- run over every prompt
token. Layers ``L+1 .. 39`` run, in a request's final context chunk only,
over the last ``min(N, 128)`` prompt positions, their sliding window truncated
at ``r = N - min(N, 128)``: a query at ``p`` attends window keys
``[max(r, p - 127), p]`` and nothing below ``r`` (the bounded layers' metadata
view carries the tail's own floor ``r`` and window seq_lens ``N - r``). Main KV
and index K stay exact; the decoder layers'
window KV is the replay's. The layer-``L+1`` input of tail positions that
arrive in earlier chunks (hc stream, the lagged pre mixes, layer ``L``'s
top-k and index lengths, and layer 20's index query) is kept per request
between chunks, and dropped with the request once the cache manager no longer
holds it (a request cancelled or evicted between two of its chunks never
reaches its last one). Past 2048 x 8 visible positions the re-index layers
mask their scores with layer 20's candidate blocks: the decoder pass scores
its tail rows' masks from layer 20's carried query against the step's index
K, which is the mask a full prefill gives those rows; so any prompt length
the engine admits runs bounded. Only tail rows
exist past layer ``L``, so the step returns hidden rows for them and for the
generation rows (every other row is zero) and DSpark captures only tail rows.
Prompt logprobs and context logits are refused, as is a captured context
forward. Under attention DP (dep4) each rank bounds its own requests, and the
bounded layers' MoE is a collective: in every eager step the ranks exchange
their passes over the bounded layers (planned on the host before the layers
run: a bounded rank's slices, or its step rows as one pass) and every rank
runs as many passes as the rank with the most, each padded to that pass's
largest row count. A rank with fewer passes, or no decoder row, joins the
remaining ones with zero rows, and each pass gathers the router's image-span
mask again from its own rows.
Generation rows run every layer. On a prefix-cache hit with encoder replay on
the replay start lies at or below the tail's first position, so every tail row
is forwarded and the decoder layers' window KV is this request's own. With
encoder replay off, a hit inside the tail would keep its cached rows' decoder
window KV, which no forward of the request wrote; so with block reuse on,
decoder replay runs only beside encoder replay. The V4.1 cache manager resolves
that (an unset switch resolves off there with one line naming why; an explicit
1 raises at construction), and the model takes decoder replay as the manager
resolved it.

**Attention.** 64 query heads over ONE shared 512-wide latent (K == V), RoPE
on the last 64 channels (adjacent pairs), inverse RoPE on the output, a
grouped low-rank output projection (8 groups x 1024) and fp32 per-head sinks.
Every layer attends a 128-token sliding window; layers 2-19 add ratio-2
compressed KV (sources 2/8/14), layers 20-39 ratio-1 compressed KV (source
20). Compressed positions are chosen per query by an indexer (32 heads x 128,
MXFP4 q/k, top-512); layer 20 additionally publishes a candidate-block mask
(top-2048 blocks of 8) that the re-index layers 24/28/32/36 apply. The
top-512 is deterministic: the same set on every call and every rank, a tie at
the 512th score going to the lower position, in ascending position order, so
the sparse attention sums a row's columns in a fixed order. The per-layer role
table is asserted against the checkpoint at construction.

**Hyper-connections (Single-Pass mHC).** ``post``/``comb`` are used by the
sublayer that computed them, but ``pre`` is one sublayer late: attention
collapses the stream with the previous FFN's mixes, the FFN with this
attention's, the final norm with layer 39's FFN mixes, and layer 0 with a
one-hot on copy 0 (which is the embedding itself).

**Cache writes.** No V4.1 attention
op appends K/V -- the sparse kernel only reads. So this forward writes its
pools itself through scatter entries, at slots the runtime assigned
(``DeepseekV4TrtllmAttentionMetadata``'s V4.1 address book): the SWA window KV
(FP8-e4m3 x UE8M0/32), the compressed main KV (E2M1 x E4M3/16, no global
scale), the index K (native MXFP4), the ratio-2 compressor's pending pair
(fp32) and the Engram token history. It never allocates, moves or frees a page;
a negative slot means skip.

**KV storage.** The window KV and the main KV are stored NATIVELY
by default: 528-byte rows (512 E4M3 codes + 16 UE8M0 bytes) and 288-byte rows
(256 bytes of packed E2M1 + 32 E4M3 scales), so the persistent KV is 890 bytes
per token (3 x (288 + 68) / 2 + (288 + 68)). The sparse-attention kernel reads
bf16 rows only, so each step dequantizes exactly the rows it reads into two
bf16 buffers and hands the kernel those, with its index table renumbered onto
them: per layer, the 128 window rows of every query (row ``t * 128 + c`` is
column ``c`` of query ``t``; a -1 column becomes a zero row, which the kernel
treats as it treats -1), and the compressed rows either per KV source, every
compressed entry of every request of the step (row ``b * L + e``), or per
top-k set, every query's top-k entries (row ``t * 512 + k``) -- whichever
buffer is smaller for the engine's batch, ``max_seq_len`` and
``max_num_tokens``. Dequantizing a native row gives the
bf16 value its bf16-storage container held, bit for bit, so native and bf16
storage produce the same tokens. ``TRTLLM_DSV41_KV_STORAGE=bf16`` keeps both
pools in bf16 containers of the same value sets (2730 bytes per token) and
the kernel reads the pools directly.

**FP8 attention (opt-in, lossy).** ``TRTLLM_DSV41_FP8_ATTENTION=1`` (native
storage only) runs the sparse attention with FP8 tensor-core math: the read
buffers hold E4M3 rows with one power-of-two scale per layer (per compressed
source group; bounded from the norm weights so nothing saturates), the query is
quantized to E4M3 with a power-of-two scale from the step's max|q|, and the
same trtllm-gen kernel family runs its ``QkvE4m3OBfloat16`` variants, which
also round the probabilities to E4M3 for the PV product. The window's MXFP8
values stay exact in E4M3 in its normal range; the query, the compressed
NVFP4 values (up to 6 significant bits) and the probabilities do not, so the
output departs from the reference's bf16 attention. The DSpark drafter's
attention stays bf16.

**Parallel members, folded from the mapping at construction:**

* ``tep4`` -- attention TP4 (16 heads, groups 2r/2r+1 per rank, fp32
  all-reduce after ``wo_b``), routed experts EP4 (96 per rank), fp32
  all-reduce of the routed partials;
* ``dep4`` -- attention DP4 (all 64 heads, each rank its own requests, no
  attention collective), EP4 with the token all-gather ahead of the router and
  an fp32 reduce-scatter after the experts;
* ``tp4`` -- attention TP4, routed experts MoE-TP4 (all 384, I = 576 per rank),
  fp32 all-reduce of the routed partials;
* ``tep8`` / ``dep8`` -- the same over two NVL72 trays (8 ranks): attention TP8
  (8 heads, output group r per rank) or DP8, routed experts EP8 (48 per rank).
  The IPC all-reduce workspace is single-node, so a TP group spanning the two
  trays sums on NCCL, which runs over the NVLink fabric (MNNVL) between them.

The indexer, compressor, hyper-connections, shared expert, embedding and every
KV pool are replicated.

**Pipeline parallelism (``pp_size`` > 1).** Each stage runs its layers
(``mapping.pp_layers``) with any member above as its TP layout, on its own
pools: a stage holds every layer its layers read from (a compressed layer's KV
source and index source, the candidate source of the re-index layers, the
Engram history owner), which leaves this checkpoint's stage boundaries at
layers 1 and 20 (PP2's even split 0-19 | 20-39; construction raises
otherwise). A later stage receives the hc stream ``[T, 4, 5120]`` and the
previous stage's last FFN mixes (its first attention's lagged pre; that
layer's ``hc_ffn`` triple is loaded there too) through ``pp_recv_tensors``, and
an earlier one sends them with ``pp_send_tensors``. The embedding and Engram
run on the stage that holds them, the final norm, the lm_head, DSpark and the
decoder tail on the last stage.

**Engram residency (host_offload).** By default the two Engram tables (202.76
GB) stay in the node's one shared host copy
(the sibling ``engram_host_table.py``: CUDA VMM host-NUMA allocations
shared by file descriptor, or with ``TRTLLM_DSV41_ENGRAM_HOST_TABLE_BACKEND=shm``
one registered ``/dev/shm`` file): every rank maps the whole table and gathers
its own tokens' rows through the mapping, so no rank holds any of the table in
device memory and no collective follows the gather. ``TRTLLM_DSV41_ENGRAM_HOST_OFFLOAD=0``
row-shards the tables 4 ways on the devices instead: a bf16 all-reduce of the
zero-filled lookups (tep4/tp4), or all-gather of the hash ids + reduce-scatter
(dep4). The two give the same bits -- each looked-up value is a sum with one
non-zero term -- except that the collective turns a -0 row element into +0.
"""

import atexit
import math
import os
import sys
import time
import weakref
from types import SimpleNamespace

import torch
from torch import nn
from transformers import PretrainedConfig

from tensorrt_llm._torch._experimental.modeling_v2._target import Phase, Target, phase_of
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import advance_step_generation
from tensorrt_llm._torch._experimental.modeling_v2.catalog.activation.flashinfer_silu_and_mul import (  # noqa: E501
    FlashinferSiluAndMul,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.thop_attention import (
    ThopAttention,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.comm.allgather import Allgather
from tensorrt_llm._torch._experimental.modeling_v2.catalog.comm.reducescatter import Reducescatter
from tensorrt_llm._torch._experimental.modeling_v2.catalog.gemm.bmm_out import BmmOut
from tensorrt_llm._torch._experimental.modeling_v2.catalog.gemm.cublas_mm import CublasMm
from tensorrt_llm._torch._experimental.modeling_v2.catalog.moe.fp4_block_scale_moe_runner import (  # noqa: E501
    Fp4BlockScaleMoeRunner,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.moe.mxe4m3_mxe2m1_block_scale_moe_runner import (  # noqa: E501
    Mxe4m3Mxe2m1BlockScaleMoeRunner,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.norm.flashinfer_rmsnorm import (
    FlashinferRmsnorm,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.quantization.fp4_quantize import (
    Fp4Quantize,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.quantization.mxfp8_quantize import (
    Mxfp8Quantize,
)
from tensorrt_llm._torch.attention.backends.interface import AttentionMetadata
from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4.metadata import (
    DeepseekV4TrtllmAttentionMetadata,
)
from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4.params import DeepseekV4AttentionType
from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4.v41_layout import (
    DEEPSEEK_V41_DECODER_REPLAY_ENV,
    DEEPSEEK_V41_KV_STORAGE_BF16,
    DEEPSEEK_V41_KV_STORAGE_NATIVE,
    DeepseekV41Layout,
    use_deepseek_v41_layout_mode,
)
from tensorrt_llm._torch.autotuner import fallback_tactic
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_utils import (
    DecoderModel,
    DecoderModelForCausalLM,
    register_auto_model,
)
from tensorrt_llm._torch.speculative import get_spec_worker
from tensorrt_llm._utils import local_mpi_barrier, local_mpi_comm, local_mpi_rank

from . import engram_host_table as _engram_host
from . import weights as _weights
from .ops.activation.moe_gelu import MoeGelu
from .ops.activation.silu_and_mul import SiluAndMul
from .ops.attention.candidate_block_select import CandidateBlockSelect
from .ops.attention.compressor_paged_kv_compress import CompressorPagedKvCompress
from .ops.attention.compressor_prefill_reduction import CompressorPrefillReduction
from .ops.attention.deepgemm_fp8_fp4_mqa_logits import DeepgemmFp8Fp4MqaLogits
from .ops.attention.deepgemm_fp8_fp4_paged_mqa_logits import (
    DeepgemmFp8Fp4PagedMqaLogits,
    DeepgemmPagedMqaLogitsMetadata,
)
from .ops.attention.dsv4_local_to_global_indices import Dsv4LocalToGlobalIndices
from .ops.attention.flashinfer_top_k_ragged_deterministic import FlashinferTopKRaggedDeterministic
from .ops.attention.indexer_k_cache_gather import IndexerKCacheGather
from .ops.attention.indexer_k_cache_scatter import IndexerKCacheScatter
from .ops.attention.mla_rope_inplace import MlaRopeInplace
from .ops.attention.trtllm_sparse_mla_dsv4 import TrtllmSparseMlaDsv4
from .ops.attention.trtllm_sparse_mla_dsv4_fp8 import TrtllmSparseMlaDsv4Fp8
from .ops.comm.allreduce import Allreduce
from .ops.comm.pp_recv_tensors import PpRecvTensors
from .ops.comm.pp_send_tensors import PpSendTensors
from .ops.embedding.engram_gather import EngramGather
from .ops.embedding.engram_hash import EngramHash
from .ops.gemm.mhc_gemm_sqrsum_fma import MhcGemmSqrsumFma
from .ops.gemm.mxfp8_mxfp8_gemm import Mxfp8Mxfp8Gemm
from .ops.moe.gate_forward import GateForward
from .ops.norm.engram_gate import EngramGate
from .ops.norm.mhc_big_fuse import MhcBigFuse
from .ops.norm.mhc_hc_head_apply import MhcHcHeadApply
from .ops.norm.mhc_post_mapping import MhcPostMapping
from .ops.quantization.e4m3_quantize_pow2_dynamic import E4m3QuantizePow2Dynamic
from .ops.quantization.mxfp8_dequantize import Mxfp8Dequantize
from .ops.quantization.mxfp8_fake_quantize import Mxfp8FakeQuantize
from .ops.quantization.mxfp8_to_e4m3 import Mxfp8ToE4m3
from .ops.quantization.nvfp4_fake_quantize import Nvfp4FakeQuantize
from .ops.quantization.nvfp4_kv_to_bf16 import Nvfp4KvToBf16
from .ops.quantization.nvfp4_kv_to_e4m3 import Nvfp4KvToE4m3

_HC = _weights.HC  # hyper-connection copies
_MIX = _weights.MIX  # 24 = 4 pre + 4 post + 16 comb
_WINDOW = 128
_TOPK = 512
_ROPE = 64
_IDX_HEADS = _weights.IDX_HEADS
_IDX_DIM = _weights.IDX_DIM
_PAGE = 128  # tokens_per_block of every V4.1 pool
_CAND_BLOCK = 8
_CAND_TOPK_BLOCKS = 2048
_SWIGLU_LIMIT = 10.0
# NVFP4 block (one e4m3 scale per 16 along K) / MXFP4 index-K block (UE8M0 per 32).
_SF16 = 16
_SF32 = 32
# The MoE runner reads its activation scales LINEAR; the swizzled buffer has
# the same byte count at T % 128 == 0 (every graph batch) and is silently wrong.
_SF_LINEAR = False
# Pre-routed path of the trtllm-gen block-scale runner: routing inert.
_ROUTING_METHOD_INERT = 1
_ACT_SWIGLU = 0
# The runner's and the routing op's token columns stop at 8192.
_MOE_MAX_T = 8192  # capacity: fixed: the MoE runner's token columns; larger steps chunk
# Rows the index top-k's zero offsets cover at load. The first forward grows
# them to the engine's step capacity (max_num_tokens): a step with more rows
# than offsets trips the top-k wrapper's one-entry-per-row assert.
_TOPK_MAX_ROWS = 8192  # capacity: max_num_tokens (grown on the first step)
# fp32 logits elements (1 GiB) one prefill index call may produce: longer
# contexts run the query rows in tiles (``_index_row_tiles``).
_INDEX_TILE_ELEMS = 1 << 28  # capacity: fixed: per-call logits budget; longer contexts tile
# Rows one index-K gather call may write: row 2^25 x 64 bytes is past int32
# (``_gather_index_k``).
_GATHER_MAX_ROWS = 1 << 25  # capacity: fixed: per-call gather rows; longer gathers chunk
# Bytes of candidate-block masks (one bool per context row x gathered key) a
# prefill step keeps for its re-index layers; past this each re-index layer
# re-scores its tiles' masks instead (``_index_prefill``).
_CAND_MASK_BYTES = 1 << 31  # capacity: fixed: kept-mask budget; past it masks are re-scored
# comm/allreduce strategies: fp32 sums on AUTO (workspace), bf16 on NCCL --
# mixing dtypes on one workspace's one-shot kernel is wrong or hangs.
_AR_NCCL = 0
_AR_AUTO = 3
# Tensor-parallel (or attention-DP) sizes this target folds: heads 64/tp and
# output groups 8/tp per rank, experts 384/tp per rank under EP.
_TP_SIZES = (2, 4, 8)
# Sparse-attention workspace (the certified size), zero-initialised once.
_ATTN_WS_BYTES = 128 * 1024 * 1024  # capacity: fixed: the certified workspace size


# --------------------------------------------------------------------------
# Runtime-value dump: metadata of every entry call (shapes, dtypes, the
# constants passed), a set of tuples appended to in Python only. CUDA-graph
# capture executes Python, so every captured batch size lands here; replay
# executes none of it. Off unless TRTLLM_MODELING_V2_VALUE_DUMP=1: the dump is
# for auditing a run against the entries' certified cells, and a served engine
# must not print it at every exit.
_VALUE_DUMP_ENV = "TRTLLM_MODELING_V2_VALUE_DUMP"
_VALUE_DUMP = os.environ.get(_VALUE_DUMP_ENV, "").strip() == "1"
_RUNTIME_VALUES: set = set()


def _rv(entry: str, *meta) -> None:
    if _VALUE_DUMP:
        _RUNTIME_VALUES.add((entry,) + tuple(meta))


def _emit_runtime_values() -> None:
    if not _RUNTIME_VALUES:
        return
    rank = os.environ.get("OMPI_COMM_WORLD_RANK", os.environ.get("RANK", "?"))
    lines = [f"=== MODELINGV2 RUNTIME VALUE DUMP (deepseek_v41 unified, rank {rank}) ==="]
    lines += [repr(v) for v in sorted(_RUNTIME_VALUES, key=repr)]
    lines.append("=== END MODELINGV2 RUNTIME VALUE DUMP ===")
    sys.stdout.write("\n".join(lines) + "\n")
    sys.stdout.flush()


if _VALUE_DUMP:
    atexit.register(_emit_runtime_values)


def _sh(t: torch.Tensor) -> tuple:
    return tuple(t.shape)


# --------------------------------------------------------------------------
# Load-time helpers (not on the forward path).


def _rope_table(base: float, positions: int, yarn: dict | None, device) -> torch.Tensor:
    """``[positions, 2, 32]`` fp32 cos/sin table, the checkpoint reference's
    ``precompute_freqs_cis`` recipe: frequency interpolation only (YaRN ramp
    between ``beta_fast`` and ``beta_slow``), no mscale, built in fp32 on the
    device the reference builds it on."""
    dim = _ROPE
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
    if yarn is not None:
        orig, factor = yarn["original"], yarn["factor"]

        def corrected_dim(rotations):
            return dim * math.log(orig / (rotations * 2 * math.pi)) / (2 * math.log(base))

        low = max(math.floor(corrected_dim(yarn["beta_fast"])), 0)
        high = min(math.ceil(corrected_dim(yarn["beta_slow"])), dim - 1)
        ramp = (
            (torch.arange(dim // 2, dtype=torch.float32, device=device) - low)
            / max(high - low, 1e-3)
        ).clamp(0, 1)
        smooth = 1 - ramp
        freqs = freqs / factor * (1 - smooth) + freqs * smooth
    angles = torch.outer(torch.arange(positions, dtype=torch.float32, device=device), freqs)
    return torch.stack([angles.cos(), angles.sin()], dim=1).contiguous()


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    f = 3
    while f * f <= n:
        if n % f == 0:
            return False
        f += 2
    return True


def _engram_layout(cfg, tokenizer_json: str, device) -> dict:
    """The Engram hash constants of the checkpoint's ``inference/engram.py``:
    the compressed-vocab token map (normalizer chain over each decoded token,
    raw token string for partial UTF-8), per-(layer, shift) odd multipliers
    from numpy PCG64 seeded ``10007 * layer_id``, the 48 distinct primes above
    ``engram_vocab_size - 1`` drawn layer -> n-gram -> head, and the per-layer
    bucket offsets (exclusive cumsum of the layer's primes)."""
    import numpy as np
    from tokenizers import Regex, Tokenizer, normalizers

    layer_ids = tuple(cfg.engram_layer_ids)
    n_max = int(cfg.engram_max_ngram_size)
    heads = int(cfg.engram_n_heads)
    tok = Tokenizer.from_file(tokenizer_json)
    sentinel = "\ue000"
    normalizer = normalizers.Sequence(
        [
            normalizers.NFKC(),
            normalizers.NFD(),
            normalizers.StripAccents(),
            normalizers.Lowercase(),
            normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
            normalizers.Replace(Regex(r"^ $"), sentinel),
            normalizers.Strip(),
            normalizers.Replace(sentinel, " "),
        ]
    )
    vocab = tok.get_vocab_size(with_added_tokens=True)
    assert vocab == cfg.vocab_size, (vocab, cfg.vocab_size)
    key_to_new: dict = {}
    lookup = [0] * vocab
    for token_id in range(vocab):
        text = tok.decode([token_id], skip_special_tokens=False)
        if "\ufffd" in text:
            key = tok.id_to_token(token_id)
        else:
            normalized = normalizer.normalize_str(text)
            key = normalized if normalized else text
        new_id = key_to_new.get(key)
        if new_id is None:
            new_id = len(key_to_new)
            key_to_new[key] = new_id
        lookup[token_id] = new_id
    classes = len(key_to_new)
    assert classes == cfg.engram_compressed_vocab_size, (
        f"Engram compressed vocab {classes} != checkpoint {cfg.engram_compressed_vocab_size}; "
        "every hash multiplier derives from it"
    )
    bound = max(1, (np.iinfo(np.int64).max // classes) // 2)
    mults = []
    for layer_id in layer_ids:
        gen = np.random.default_rng(10007 * layer_id)
        vals = gen.integers(low=0, high=bound, size=(n_max,), dtype=np.int64)
        mults.append([int(v) * 2 + 1 for v in vals])
    primes, seen = [], set()
    engram_vocab = _weights._engram_vocab_size(cfg)
    for _ in layer_ids:
        per_ngram = []
        for _ in range(n_max - 1):
            sizes, cur = [], engram_vocab - 1
            for _ in range(heads):
                cur += 1
                while not _is_prime(cur) or cur in seen:
                    cur += 1
                seen.add(cur)
                sizes.append(cur)
            per_ngram.append(sizes)
        primes.append(per_ngram)
    offsets = []
    for li, layer in enumerate(primes):
        flat = [p for per in layer for p in per]
        assert sum(flat) == int(cfg.engram_num_embeddings[li]), (
            f"Engram layer {layer_ids[li]}: primes sum to {sum(flat)}, table has "
            f"{cfg.engram_num_embeddings[li]} rows"
        )
        offsets.append([int(x) for x in np.cumsum([0] + flat[:-1])])
    return {
        "token_map": torch.tensor(lookup, dtype=torch.int32, device=device),
        "multipliers": torch.tensor(mults, dtype=torch.int64, device=device),
        "primes": torch.tensor(primes, dtype=torch.int64, device=device),
        "offsets": torch.tensor(offsets, dtype=torch.int64, device=device),
        "pad_id": int(lookup[int(cfg.engram_pad_token_id)]),
    }


def _gemm_backend(m: int) -> str:
    """mxfp8_mxfp8_gemm backend by row count: CuTe DSL is fastest at M <= 32,
    cuDNN above (both certified)."""
    return "cute-dsl" if m <= 32 else "cudnn"


def _tile_n(m: int) -> int:
    """mhc_gemm_sqrsum_fma's measured tile: 1 at M <= 128, 6 to 2048, 24 above."""
    if m <= 128:
        return 1
    if m <= 2048:
        return 6
    return 24


def _moe_chunks(total: int) -> list:
    full, rest = divmod(total, _MOE_MAX_T)
    return [_MOE_MAX_T] * full + ([rest] if rest else [])


def _index_row_tiles(rows: int, s_len: int) -> list:
    """Row ranges ``[a, b)`` of one prefill index pass over ``s_len`` gathered
    keys, so that no range's fp32 logits exceed ``_INDEX_TILE_ELEMS`` (the op
    pads each row to ``align(s_len + 256, 256)`` columns). One range when they
    fit; otherwise ranges of a multiple of 64 rows where possible. Every op of
    the pass (logits, candidate select, mask, top-k) is per row, so the tiling
    does not change any row's result."""
    width = -(-(s_len + 256) // 256) * 256
    per = max(1, _INDEX_TILE_ELEMS // width)
    if per >= rows:
        return [(0, rows)]
    if per >= 64:
        per -= per % 64
    return [(a, min(rows, a + per)) for a in range(0, rows, per)]


# --------------------------------------------------------------------------
# DSpark drafter geometry (``mtp.*``; the checkpoint's ``inference/model.py``
# DSparkBlock / DSparkAttention / forward_spec). Its weights are the
# ``DRAFT_WEIGHTS`` table.

_DS_BLOCK = 5
_DS_STAGES = 3
_DS_EXPERTS = 128
_DS_TOPK = 3
# Index row of a block query: 128 ring columns, the 5 block rows, 3 x -1 so
# the width (136) is a multiple of 4; 133 columns are live.
_DS_BLK_COLS = 8
_DS_IDX_W = _WINDOW + _DS_BLK_COLS
_DS_IDX_LEN = _WINDOW + _DS_BLOCK
# The MXFP4 runner reads its MXFP8 activation linear, padded to 512 columns.
_DS_MOE_ALIGN = 512
# trtllm_sparse_mla_dsv4 Precondition 8: a pool the call loads a -1 index
# from must span >= 128 mapped rows from its first row. The block K/V rows sit
# at the head of a persistent buffer this many rows longer than 5 x ring rows.
_DS_BLK_PAD_ROWS = 128


# --------------------------------------------------------------------------
# Vision tower geometry (``vision_config``; the checkpoint's
# ``inference/vision.py`` ViT + Aligner) and the per-call row envelope the
# entries certify for it. Its weights are the ``vis*`` rows of the table.

_IMAGE_TOKEN = 129264  # every image-span position carries this id
_VIT_LAYERS = _weights.VIT_LAYERS
_VIT_DIM = _weights.VIT_DIM
_VIT_HEADS = _weights.VIT_HEADS
_VIT_HEAD_DIM = 64
_VIT_INTER = _weights.VIT_INTER
_VIT_PATCH_IN = _weights.VIT_PATCH_IN  # a patch flattened (channel, kh, kw)
_VIT_RATIO = _weights.VIT_RATIO  # the aligner's 3x3 block
_VIT_EPS = 1e-6
_VIT_ROPE_HALF = _VIT_HEAD_DIM // 2  # 16 row + 16 column angles per patch
# thop_attention no-pool: bidirectional within each sequence (= image).
_VIT_MASK_BIDIRECTIONAL = 0
# thop's max_seq_len / max_context_length / attention_window_size: inert
# without a pool, but must cover the longest image (9189 patches).
_VIT_ATTN_MAX_LEN = 16384  # capacity: fixed: the largest image's patch count fits
# thop_attention's reference-checked per-call envelope (cublas_mm and the
# RoPE are certified beyond it); it sizes the load-time ``arange`` tables. The
# tower runs one image, at most 9189 patches, per call.
_VIT_MAX_PATCHES = 17037  # capacity: fixed: the certified per-call patch envelope
# moe_gelu and the aligner's GEMMs are certified to this many cells per call.
_AL_MAX_ROWS = 2048  # capacity: fixed: certified cells per call; more cells chunk
# The patch embed's cublas_mm (bf16 [M, 588] x [588, 1024]) picks a split-K
# kernel with bf16 partial sums at these row counts only, and the entry's
# is_valid refuses them. The patch rows are zero-padded past the band instead
# (``_patch_embed_rows``) and the padded rows' output is dropped.
_VIT_SPLITK_ROWS = (2305, 2432)
_VIT_SPLITK_PAD = 2433
# The largest image: the span limit (max_image_tokens 1024) allows 9 * 1021
# patches (a 3 x 3063 grid); the tower runs one image per call.
_VIT_MAX_IMAGE_PATCHES = 9216  # capacity: fixed: max_image_tokens 1024 of the checkpoint
# ViT CUDA graphs: one captured tower call per padded patch count, 256-patch
# steps to 4096 and 512-patch steps to 9216. No bucket falls in the split-K
# band above, so a replay never needs the pad.
_VIT_GRAPH_ENV = "TRTLLM_DSV41_VIT_CUDA_GRAPH"
_VIT_GRAPH_BUCKETS = tuple(range(1024, 4096 + 1, 256)) + tuple(
    range(4608, _VIT_MAX_IMAGE_PATCHES + 1, 512)
)
# Sequence slots of a captured call: the image, then the pad sequence over
# rows [P, B) (empty when P == B).
_VIT_GRAPH_SLOTS = 2
assert not any(_VIT_SPLITK_ROWS[0] <= b <= _VIT_SPLITK_ROWS[1] for b in _VIT_GRAPH_BUCKETS)
assert _VIT_GRAPH_BUCKETS[-1] == _VIT_MAX_IMAGE_PATCHES <= _VIT_MAX_PATCHES


def _patch_embed_rows(patches: int) -> int:
    """Rows the patch-embed GEMM runs for ``patches`` patch rows: the rows
    themselves, or ``_VIT_SPLITK_PAD`` when they fall in the split-K band."""
    lo, hi = _VIT_SPLITK_ROWS
    return _VIT_SPLITK_PAD if lo <= patches <= hi else patches


def _vision_geometry(cfg) -> None:
    """Assert the vision geometry this assembly is written for."""
    vc = getattr(cfg, "vision_config", None)
    assert vc is not None, "the checkpoint config has no vision_config"
    if isinstance(vc, dict):
        vc = PretrainedConfig(**vc)
    got = (
        int(vc.num_hidden_layers),
        int(vc.hidden_size),
        int(vc.num_attention_heads),
        int(vc.intermediate_size),
        int(vc.patch_size),
        int(vc.downsample_ratio),
        float(vc.rope_theta),
    )
    assert got == (_VIT_LAYERS, _VIT_DIM, _VIT_HEADS, _VIT_INTER, 14, _VIT_RATIO, 10000.0), got
    assert int(cfg.image_token_id) == _IMAGE_TOKEN, cfg.image_token_id


def _dspark_geometry(tc) -> SimpleNamespace:
    """The drafter geometry this assembly is written for, read off
    ``text_config`` -- the drafter's own ``dspark_*`` keys, not the trunk's
    384 experts at top-6."""
    g = SimpleNamespace(
        hidden=int(tc.hidden_size),
        vocab=int(tc.vocab_size),
        eps=float(tc.rms_norm_eps),
        heads=int(tc.num_attention_heads),
        head_dim=int(tc.head_dim),
        q_lora=int(tc.q_lora_rank),
        o_groups=int(tc.o_groups),
        o_lora=int(tc.o_lora_rank),
        window=int(tc.sliding_window),
        theta=float(tc.rope_theta),
        inter=int(tc.moe_intermediate_size),
        stages=int(tc.num_nextn_predict_layers),
        block=int(tc.dspark_block_size),
        noise=int(tc.dspark_noise_token_id),
        markov_rank=int(tc.dspark_markov_rank),
        experts=int(tc.dspark_n_routed_experts),
        topk=int(tc.dspark_num_experts_per_tok),
        targets=tuple(tc.dspark_target_layer_ids),
        route_scale=float(tc.routed_scaling_factor),
        swiglu=float(tc.swiglu_limit),
    )
    n = int(tc.num_hidden_layers)
    ratios = list(tc.compress_ratios)
    assert ratios[n:] == [0] * g.stages, (
        f"drafter stages are window-only (theta 10000, no YaRN); got {ratios[n:]}"
    )
    assert (g.hidden, g.vocab, g.eps) == (5120, 129280, 1e-20), (g.hidden, g.vocab, g.eps)
    assert (g.heads, g.head_dim, g.q_lora, g.o_groups, g.o_lora) == (64, 512, 1280, 8, 1024)
    assert (g.window, g.theta) == (_WINDOW, 10000.0)
    assert (g.stages, g.block, g.experts, g.topk) == (
        _DS_STAGES,
        _DS_BLOCK,
        _DS_EXPERTS,
        _DS_TOPK,
    ), (
        g.stages,
        g.block,
        g.experts,
        g.topk,
    )
    assert (g.inter, g.markov_rank, g.noise) == (2304, 256, 128799)
    assert g.targets == (37, 38, 39), g.targets
    assert (g.route_scale, g.swiglu) == (1.5, _SWIGLU_LIMIT)
    return g


# --------------------------------------------------------------------------
# Bounded replays (tech report 3.2.2): the decoder
# replay's switch and tail plan, and the witness counters of both replays
# (``TRTLLM_MODELING_V2_WITNESS=1``).

#: The decoder replay's switch, read once at construction: 1/true/on/yes, 0/false/off/no,
#: or unset (on, under attention TP and attention DP alike). The engine's cache manager
#: reads it too and turns an unset switch off where block reuse is on while encoder
#: replay is off (``DeepseekV4CacheManager.v41_decoder_replay``).
_REPLAY_ENV = DEEPSEEK_V41_DECODER_REPLAY_ENV
#: Set to 1 by the replay witness: one counter line per context request, when its
#: context phase completes.
_WITNESS_ENV = "TRTLLM_MODELING_V2_WITNESS"
# FP8 attention: E4M3 query and E4M3 pools (one power-of-two scale each) through
# the FP8-MMA sparse-MLA kernel. Off by default: it is lossy against the
# reference's bf16 attention (see ``_FP8_KV_BOUND``).
_FP8_ATTN_ENV = "TRTLLM_DSV41_FP8_ATTENTION"
# |kv| <= sqrt(512) * max|norm weight| for an RMS-normed 512-vector (RoPE keeps
# each rotated pair's norm), and the stored MXFP8 / NVFP4 rounding raises a
# block's amax by at most 2^-4: the per-tensor KV scale 2^ceil(log2(bound/448))
# never saturates.
_FP8_KV_BOUND = 512**0.5 * (1.0 + 2.0**-4)


def _env_switch(name: str):
    """The explicit value of boolean switch ``name``, or None when unset."""
    raw = os.environ.get(name, "").strip().lower()
    if raw == "":
        return None
    if raw in ("1", "true", "on", "yes"):
        return True
    if raw in ("0", "false", "off", "no"):
        return False
    raise ValueError(f"{name}={os.environ[name]!r}: expected 1/true/on/yes or 0/false/off/no")


def _witness_line(**fields) -> None:
    sys.stdout.write(
        "MODELING_V2_WITNESS " + " ".join(f"{k}={int(v)}" for k, v in fields.items()) + "\n"
    )
    sys.stdout.flush()


def _dp_pass_rows(gathered: list) -> list:
    """dep4: the padded row count of each pass over the bounded layers, from
    every rank's ``[count, rows_0, rows_1, ...]``: as many passes as the rank
    with the most, each the max over the ranks that hold one."""
    passes = max(r[0] for r in gathered)
    return [max(r[1 + k] if k < r[0] else 0 for r in gathered) for k in range(passes)]


class _TailRecord:
    """A context request between two of its chunks: the position its next
    chunk must start at, its counters, and (under replay) the stash -- the
    layer-(L+1) input rows of its last ``window`` forwarded positions
    ``[start, end)``, keyed like the step's per-row tensors. A first chunk
    after a prefix hit under encoder replay starts ``replayed`` positions
    before the reuse boundary: the prefix cache served ``first_position +
    replayed`` tokens."""

    __slots__ = ("end", "reused", "replayed", "fwd", "dec", "start", "stash")

    def __init__(self, first_position: int, replayed: int = 0):
        self.end = first_position
        self.reused = first_position + replayed
        self.replayed = replayed
        self.fwd = 0
        self.dec = 0
        self.start = first_position
        self.stash = None


class _TailMetadata:
    """The step's attention metadata as the bounded layers see it: the per-row
    and per-request fields of the decoder row set; every other attribute is
    the engine's."""

    def __init__(self, md, fields: dict):
        self.__dict__["_md"] = md
        self.__dict__.update(fields)

    def __getattr__(self, name):
        return getattr(self.__dict__["_md"], name)


class _TailSlots:
    """``v41_swa_slot_mapping`` of the decoder row set, per local layer: a tail
    row's slot through the layer's SWA block table at the row's position (a
    stash row's position lies in an earlier chunk, outside this step's address
    book); generation rows as the metadata has them. ``l2g`` is the prefill
    target's ``dsv4_local_to_global_indices`` entry."""

    def __init__(self, md, req, self_local, T, tc, Td, tcd, dev, l2g=None):
        self.md, self.req, self.self_local, self.dev = md, req, self_local, dev
        self.T, self.tc, self.Td, self.tcd = T, tc, Td, tcd
        self.l2g = l2g
        self._slots = {}

    def __getitem__(self, local):
        s = self._slots.get(local)
        if s is not None:
            return s
        md = self.md
        s = torch.empty([self.Td], dtype=torch.int64, device=self.dev)
        if self.tcd:
            # Column 0 of a [row position, -1, ...] table is the row's own slot.
            bt = md.sliding_block_tables[local, DeepseekV4AttentionType.SWA.value]
            _rv("dsv4_local_to_global_indices", self.tcd, _WINDOW)
            g = self.l2g(self.req, bt, self.self_local, _PAGE, 0)
            s[: self.tcd].copy_(torch.reshape(torch.split(g, [1, _WINDOW - 1], 1)[0], [self.tcd]))
        if self.Td > self.tcd:
            s[self.tcd :].copy_(md.v41_swa_slot_mapping[local][self.tc : self.T])
        self._slots[local] = s
        return s


class _TailCapture:
    """``spec_metadata`` for the bounded layers. DSpark reads a captured
    layer's input over the step's rows, so the tail rows are put back at their
    rows of the step first; the tail rows of earlier chunks have no row there
    and are captured on their own for the ring (``_DecoderTail.stage``)."""

    def __init__(self, spec, tail):
        self.__dict__["_spec"] = spec
        self.__dict__["_tail"] = tail

    def is_layer_capture(self, layer_id: int) -> bool:
        return self._spec.is_layer_capture(layer_id)

    def maybe_capture_hidden_states(self, layer_id: int, hidden_states: torch.Tensor) -> None:
        self._spec.maybe_capture_hidden_states(
            layer_id, self._tail.stage(layer_id, hidden_states, self._spec)
        )

    def __getattr__(self, name):
        return getattr(self.__dict__["_spec"], name)


class _DecoderTail:
    """Decoder SWA bounded replay for one step that has context requests.

    Planned before the layers run, from host metadata only (request ids, chunk
    and cached lengths, the engine's last-chunk flags). A request whose chunk
    is not its last contributes no decoder row, and its last ``window``
    forwarded positions become its stash. A request in its last chunk
    contributes its tail ``[s, N)``, ``s = max(N - min(N, window), first
    forwarded position)``: its stash rows below this chunk, then this chunk's.
    The decoder row set is those tails, request after request, then the
    generation rows; it runs the bounded layers in slices of at most the
    engine's step capacity, cut at request boundaries (``enter``). ``bound``
    is False when no context row would change (every request in its last
    chunk, ``N <= window``, nothing stashed): the step then runs every layer
    over every row. With replay off the plan only counts rows for the
    witness.

    Prefill-only: a decode step has no context request to plan. ``l2g`` is the
    prefill target's ``dsv4_local_to_global_indices`` entry, which the tail
    rows' window slots are resolved through."""

    def __init__(self, core, md, st, dev, l2g=None):
        self.core, self.md, self.dev = core, md, dev
        self.l2g = l2g
        T, tc, nc = st["T"], st["tc"], st["nc"]
        self.T, self.tc, self.nc = T, tc, nc
        W = core.replay_window
        last = getattr(md, "context_last_chunk", None)
        assert last is not None and len(last) == nc, (
            "decoder replay and its witness counter need the engine's per-request "
            f"context_last_chunk flags; got {last!r} for {nc} context requests"
        )
        if core.replay:
            flags = getattr(md, "context_return_logits", None)
            assert flags is not None and len(flags) == nc, flags
            if any(flags):
                raise ValueError(
                    "decoder SWA bounded replay runs the layers after the last KV-source layer "
                    "over each prompt's last 128 positions only, so prompt logprobs and context "
                    f"logits do not exist; serve them with {_REPLAY_ENV}=0"
                )
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "decoder SWA bounded replay selects a context step's decoder rows on the "
                    "host; a captured context forward (prefill CUDA graphs) cannot run it"
                )
        records = core._tail_records
        rids = list(md.request_ids)
        for rid in rids[nc:]:
            records.pop(rid, None)  # a generation row: its context phase is over
        seq = md.seq_lens[:nc].tolist()
        cached = md.cached_token_lens_cpu[:nc].tolist()
        # Encoder replay rows at the head of a hit's first chunk: they are
        # forwarded rows like any other (a tail starts at or after them).
        replayed = md.v41_num_replay_tokens_cpu[:nc].tolist()
        reqs, off, bound, need = [], 0, False, W + 1
        for b in range(nc):
            L, c, R, rid = int(seq[b]), int(cached[b]), int(replayed[b]), rids[b]
            rec = records.get(rid)
            if rec is None or rec.end != c or R:
                rec = _TailRecord(c, R)  # the first chunk of this context phase
            r = SimpleNamespace(
                b=b, rid=rid, off=off, L=L, c=c, R=R, N=c + L, last=bool(last[b]), rec=rec
            )
            if core.replay:
                stashed = rec.stash is not None
                if not r.last:
                    bound = True
                    r.keep = max(r.N - W, rec.start if stashed else c)
                else:
                    r.t0 = r.N - min(r.N, W)
                    r.s = max(r.t0, rec.start if stashed else c)
                    bound = bound or r.N > W or stashed
                need = max(need, r.N + 1)
            reqs.append(r)
            off += L
        assert off == tc, (off, tc)
        # Decoder rows the step needs (tails, then generation rows).
        self.rows = sum(r.N - r.s for r in reqs if r.last) + st["gen"] if core.replay else 0
        core._ensure_ar(max(need, nc * W + 1), dev)
        self.reqs = reqs
        self.bound = bound
        self.counted = False
        self.slices, self.cur, self.h_full, self.out = [], None, None, None
        self.h_layer = {}

    def count_forwarded(self, x) -> None:
        """Layer 0: each request's rows of its input ``x``."""
        for r in self.reqs:
            r.rec.fwd += x[r.off : r.off + r.L].shape[0]

    def count_full(self, h) -> None:
        """Unbounded step: layer L+1 runs over each request's rows of ``h``."""
        for r in self.reqs:
            r.rec.dec += h[r.off : r.off + r.L].shape[0]
        self.counted = True

    def enter(self, rows: dict, st: dict, cache: dict, max_rows: int) -> list:
        """At the first bounded layer. Stash the rows of requests whose chunk
        is not their last, and return the decoder passes: the decoder row set
        cut at request boundaries into slices of at most ``max_rows`` rows,
        each as (its rows of every per-row tensor in ``rows`` -- the step's
        layer-(L+1) inputs --, its step dict, its per-step cache). The tail
        requests come in step order; the generation rows go in the last slice.
        No decoder row: an empty list.

        ``max_rows`` is the engine's step capacity (max_num_tokens), which
        every per-row buffer and every certified op shape is sized for. A
        finishing request's tail holds min(N, 128) rows, the rows of its
        earlier chunks included, so the row set outgrows the step itself when
        more than max_rows / 128 requests finish in it. Every bounded op is
        per row or per request, so the slices compute what one pass would."""
        core, md = self.core, self.md
        W = core.replay_window
        T, gen = self.T, st["gen"]
        assert max_rows >= W, (max_rows, W)
        rows = dict(rows)
        # Per-row inputs of the bounded layers besides the stream: the last
        # KV-source layer's top-k (layers 21-23 read it), the window + top-k
        # counts and the visible compressed length (index row ends).
        rows["topk"] = cache["topk"][1]
        rows["lens"] = md.sparse_mla_topk_lens[1][:T]
        rows["vis"] = md.compressed_visible_lens_cuda[1][:T]
        # The candidate source's index query (MXFP4 q, its UE8M0 words, the
        # head weights): a tail row's candidate-block mask is scored from it in
        # the decoder pass, against the index K of the step that runs it.
        rows["cq"], rows["cqs"], rows["cw"] = cache["cand_q"]
        names = list(rows)
        for r in self.reqs:
            if r.last:
                continue
            rec = r.rec
            parts = {n: [] for n in names}
            if rec.stash is not None and r.keep < r.c:
                for n in names:
                    parts[n].append(rec.stash[n][r.keep - rec.start :])
            a = max(r.keep, r.c) - r.c
            for n in names:
                parts[n].append(rows[n][r.off + a : r.off + r.L])
            rec.stash = {n: torch.cat(parts[n], 0) for n in names}
            rec.start = r.keep
        fin = [r for r in self.reqs if r.last]
        for r in fin:
            r.n_old = r.c - r.s if (r.rec.stash is not None and r.s < r.c) else 0
            r.a = max(r.t0, r.c) - r.c
            r.n = r.N - r.s
            r.rec.dec += r.n
        self.h_full = rows["h"]
        self.counted = True
        groups = self._groups(max_rows)
        last = len(groups) - 1
        self.slices = [
            self._slice(g, gen if k == last else 0, rows, names, st, cache)
            for k, g in enumerate(groups)
        ]
        # The passes are the ones pass_rows planned (attention DP pads to them).
        assert [sl.Td for sl in self.slices] == self.pass_rows(max_rows), (
            [sl.Td for sl in self.slices],
            self.pass_rows(max_rows),
        )
        if self.slices:
            self.out = torch.empty([T, core.hidden], dtype=torch.bfloat16, device=self.dev)
            self.out.copy_(core._const["zero_bf16"].expand([T, core.hidden]))
        return [(sl.dec, sl.st, sl.cache) for sl in self.slices]

    def _groups(self, max_rows: int) -> list:
        """The finishing requests cut at request boundaries into slices of at
        most ``max_rows`` rows (a tail is ``N - s`` rows); the generation rows
        go in the last. Host-only, so it is known before the layers run."""
        gen = self.T - self.tc
        groups, cur, n_cur = [], [], 0
        for r in (r for r in self.reqs if r.last):
            n = r.N - r.s
            if cur and n_cur + n > max_rows:
                groups.append(cur)
                cur, n_cur = [], 0
            cur.append(r)
            n_cur += n
        if gen and cur and n_cur + gen > max_rows:
            groups.append(cur)
            cur = []
        if cur or gen:
            groups.append(cur)
        return groups

    def pass_rows(self, max_rows: int) -> list:
        """The row count of each decoder pass ``enter`` will return."""
        groups = self._groups(max_rows)
        gen = self.T - self.tc
        return [
            sum(r.N - r.s for r in g) + (gen if k == len(groups) - 1 else 0)
            for k, g in enumerate(groups)
        ]

    def _slice(self, fin, gen, rows, names, st, cache):
        """One decoder pass: the tails of ``fin`` (in order), then the step's
        generation rows when ``gen``; its rows, step dict and per-step cache."""
        core, md, dev = self.core, self.md, self.dev
        W = core.replay_window
        T, tc, nc = self.T, self.tc, self.nc
        G = st["G"] if gen else 0
        B = st["B"]
        ar = core._ar32
        parts = {n: [] for n in names}
        pos, req, win, kvl, flo, wsl, ks, cu = [], [], [], [], [], [], [], [0]
        q = 0
        for r in fin:
            rec = r.rec
            for n in names:
                if r.n_old:
                    parts[n].append(rec.stash[n][r.s - rec.start :])
                parts[n].append(rows[n][r.off + r.a : r.off + r.L])
            r.q = q
            got = sum(p.shape[0] for p in parts["h"][-(2 if r.n_old else 1) :])
            assert got == r.n, (got, r.N, r.s)
            pos.append(ar[r.s : r.N])
            req.append(ar[r.b : r.b + 1].expand([r.n]))
            j0 = r.s - r.t0
            win.append(
                torch.where(
                    core._const["tri"][j0 : j0 + r.n],
                    torch.reshape(ar[r.t0 : r.t0 + W], [1, W]).expand([r.n, W]),
                    -1,
                )
            )
            kvl.append(md.kv_lens_cuda_runtime[r.b : r.b + 1])
            # The tail's SWA floor is t0, not the engine's floor for request b:
            # its window count comes from kv length N minus t0.
            flo.append(ar[r.t0 : r.t0 + 1])
            wsl.append(ar[r.N - r.t0 : r.N - r.t0 + 1])
            ks.append(md.v41_ctx_index_ks[1][r.off + r.a : r.off + r.a + 1].expand([r.n]))
            q += r.n
            cu.append(q)
        tcd, ncd = q, len(fin)
        Td = tcd + gen
        max_tail = max((r.n for r in fin), default=0)
        _rv("decoder_replay.rows", T, tcd, ncd, gen, max_tail)
        if gen:
            for n in names:
                parts[n].append(rows[n][tc:T])
            pos.append(st["pos"][tc:T])
            req.append(md.req_idx_per_token[tc:T])
            win.append(md.swa_local_indices_cuda[tc:T])
            kvl.append(md.kv_lens_cuda_runtime[nc:B])
        dec = {n: torch.cat(parts[n], 0) for n in names}
        pos_d = torch.cat(pos, 0)
        req_d = torch.cat(req, 0)
        vis = dec.pop("vis")
        fields = {
            "req_idx_per_token": req_d,
            "swa_local_indices_cuda": torch.cat(win, 0),
            "sparse_mla_topk_lens": {1: dec.pop("lens")},
            "compressed_visible_lens_cuda": {1: vis},
            "kv_lens_cuda_runtime": torch.cat(kvl, 0),
            # Generation rows read their tables at [ncd:ncd+G]; the engine's
            # are at [nc:nc+G].
            "indexer_compress_block_tables": {1: md.indexer_compress_block_tables[1][nc - ncd :]},
            # Per tail request: its floor and the context attention's seq_lens
            # (kv length - floor), in the decoder pass's request order.
            "v41_swa_floor_cuda": torch.cat(flo, 0) if ncd else None,
            "v41_swa_seq_lens_cuda": torch.cat(wsl, 0) if ncd else None,
            "num_contexts": ncd,
            "num_ctx_tokens": tcd,
            "num_seqs": ncd + G,
            "v41_max_ctx_q_len": max_tail,
        }
        self_local, ring = None, None
        if ncd:
            fields["cu_seq_lens_cuda"] = torch.cat([ar[x : x + 1] for x in cu], 0)
            ks_d = torch.cat(ks, 0)
            fields["v41_ctx_index_ks"] = {1: ks_d}
            fields["v41_ctx_index_ke"] = {1: torch.add(ks_d, vis[:tcd])}
            self_local = torch.where(
                core._const["col0"].expand([tcd, W]),
                torch.reshape(pos_d[:tcd], [tcd, 1]).expand([tcd, W]),
                -1,
            )
            old = [r for r in fin if r.n_old]
            if old and core.ds is not None:
                ring = (
                    old,
                    torch.empty(
                        [sum(r.n_old for r in old), len(core.ds.targets) * core.hidden],
                        dtype=torch.bfloat16,
                        device=dev,
                    ),
                )
        fields["v41_swa_slot_mapping"] = _TailSlots(
            md, req_d[:tcd], self_local, T, tc, Td, tcd, dev, self.l2g
        )
        # nc / B count the decoder pass's requests (its context tails, then the
        # generation requests); request indices (req_idx_per_token, block-table
        # rows) stay the engine's, whose request count is B_engine.
        std = dict(st)
        std.update(
            md=_TailMetadata(md, fields),
            T=Td,
            tc=tcd,
            nc=ncd,
            G=G,
            gen=gen,
            B=ncd + G,
            B_engine=B,
            pos=pos_d,
            max_ctx_q=max_tail,
        )
        cached = {
            "topk": {1: dec.pop("topk")},
            "ctx_k": cache["ctx_k"],
            "sched": cache["sched"],
            "cand_q": (dec.pop("cq"), dec.pop("cqs"), dec.pop("cw")),
        }
        if "cand_gen" in cache:
            cached["cand_gen"] = cache["cand_gen"]
        return SimpleNamespace(
            fin=fin,
            gen=gen,
            tcd=tcd,
            Td=Td,
            ring=ring,
            ring_done=set(),
            dec=dec,
            st=std,
            cache=cached,
        )

    def activate(self, k: int) -> None:
        """Slice k is the decoder pass that runs next."""
        self.cur = self.slices[k]

    def stage(self, layer_id: int, h_flat, spec):
        """The step-row view of a bounded layer's input for DSpark's capture:
        the post-layer-L stream with the running slice's decoder rows written
        back to their rows of the step. With several slices each captured
        layer keeps its own copy of the step rows, so an earlier slice's rows
        still hold this layer's input when a later slice captures (a capture
        overwrites the layer's rows; the last slice's holds every row). The
        tail rows of earlier chunks go through the metadata's own capture
        first (it averages the hc copies), and their rows are kept for the
        shell to write into the request's ring."""
        core, sl = self.core, self.cur
        hd = torch.reshape(h_flat, [sl.Td, _HC, core.hidden])
        if sl.ring is not None:
            ring, buf = sl.ring
            k = core.ds.targets.index(layer_id)
            src = torch.cat([hd[r.q : r.q + r.n_old] for r in ring], 0)
            n = src.shape[0]
            spec.maybe_capture_hidden_states(layer_id, torch.reshape(src, [n, core.hc_k]))
            cols = [core.hidden * k, core.hidden, core.hidden * (len(core.ds.targets) - k - 1)]
            torch.split(buf, cols, 1)[1].copy_(torch.split(spec.get_hidden_states(n), cols, 1)[1])
            sl.ring_done.add(k)
        hf = self.h_full
        if len(self.slices) > 1:
            hf = self.h_layer.get(layer_id)
            if hf is None:
                hf = self.h_layer[layer_id] = torch.empty(
                    list(self.h_full.shape), dtype=self.h_full.dtype, device=self.dev
                )
                hf.copy_(self.h_full)
        for r in sl.fin:
            hf[r.off + r.a : r.off + r.L].copy_(hd[r.q + r.n_old : r.q + r.n])
        if sl.gen:
            hf[self.tc : self.T].copy_(hd[sl.tcd : sl.Td])
        return torch.reshape(hf, [self.T, core.hc_k])

    def scatter(self, y) -> None:
        """The running slice's output rows into the step's [T, hidden] output
        (zeros on rows that no bounded layer ran)."""
        sl = self.cur
        for r in sl.fin:
            self.out[r.off + r.a : r.off + r.L].copy_(y[r.q + r.n_old : r.q + r.n])
        if sl.gen:
            self.out[self.tc : self.T].copy_(y[sl.tcd : sl.Td])

    def close(self, out, spec):
        """The step's output rows, then ``finish``: ``out`` for an unbounded
        step, the slices' scattered rows for a bounded one. With no decoder
        row at all the captured layers still read the step's rows (their
        post-layer-L stream) and every output row is zero."""
        if self.bound:
            if not self.slices:
                if spec is not None:
                    flat = torch.reshape(self.h_full, [self.T, self.core.hc_k])
                    for i in range(self.core.late, self.core.num_layers):
                        if spec.is_layer_capture(i):
                            spec.maybe_capture_hidden_states(i, flat)
                out = torch.empty([self.T, self.core.hidden], dtype=torch.bfloat16, device=self.dev)
                out.copy_(self.core._const["zero_bf16"].expand([self.T, self.core.hidden]))
            else:
                out = self.out
        self.finish()
        return out

    def finish(self) -> None:
        """Carry each request to its next chunk; print the witness counter of
        each whose context phase completed."""
        core = self.core
        assert self.counted, "the step never reached the first bounded layer"
        for sl in self.slices:
            if sl.ring is not None and len(sl.ring_done) == len(core.ds.targets):
                ring, buf = sl.ring
                ar, off = core._ar32, 0
                for r in ring:
                    frames = torch.empty([r.n_old], dtype=torch.int64, device=self.dev)
                    frames.copy_(ar[r.s + 1 : r.c + 1])  # frame = position + 1
                    core._tail_ring.append((r.rid, buf[off : off + r.n_old], frames))
                    off += r.n_old
        for r in self.reqs:
            rec = r.rec
            rec.end = r.N
            if r.last:
                core._tail_records.pop(r.rid, None)
                if core._witness:
                    _witness_line(
                        prompt_tokens=r.N,
                        reused_tokens=rec.reused,
                        replayed_tokens=rec.replayed,
                        forwarded_tokens=rec.fwd,
                        decoder_tokens=rec.dec,
                    )
            else:
                core._tail_records[r.rid] = rec


class DeepseekV41UnifiedModelingV2(DecoderModel):
    """The core: what DeepSeek-V4.1-Flash statically is on this rank.

    Geometry and the member read off the mapping, the weights (``w``, and the
    drafter's ``w_ds`` under DSpark), the per-layer operand table and the rope,
    Engram and constant tables derived after load, the engine's pool views, and
    the host-side bookkeeping that outlives a step (decoder-replay records). It
    runs no layer itself: ``forward`` routes each step to the prefill or the
    decode target, which hold and bind every entry they call.
    """

    def __init__(self, model_config: ModelConfig):
        super().__init__(model_config)
        cfg = model_config.pretrained_config
        assert cfg is not None
        tc = getattr(cfg, "text_config", None)
        tc = cfg if tc is None else tc
        # The engine picks the KV cache manager and the attention metadata off
        # the built model's sparse config: the layout mode gives it the
        # deepseek_v4 runtime whose V4.1 pools and metadata this core reads.
        use_deepseek_v41_layout_mode(model_config)

        # ---- identity: topology, dtype ----------------------------------------
        mapping = model_config.mapping
        tp = mapping.tp_size
        assert tp in _TP_SIZES and mapping.cp_size == 1, (
            f"unified target: tp in {_TP_SIZES}, no context parallelism; got tp={tp} cp={mapping.cp_size}"
        )
        assert mapping.world_size == tp * mapping.pp_size, (mapping.world_size, tp, mapping.pp_size)
        self.dp = bool(mapping.enable_attention_dp)
        self.ep = mapping.moe_ep_size
        self.moe_tp = mapping.moe_tp_size
        if self.ep == tp:
            assert self.moe_tp == 1, f"EP{tp} members split experts one way only"
            self.member = f"dep{tp}" if self.dp else f"tep{tp}"
        else:
            assert self.ep == 1 and self.moe_tp == tp and not self.dp, (
                f"members are tep, dep, tp; got tp={tp} ep={self.ep} moe_tp={self.moe_tp} dp={self.dp}"
            )
            self.member = f"tp{tp}"
        # Pipeline stages: this rank runs `pp_layer_ids` (the engine's split,
        # `mapping.pp_layers`), and checks below that none of them reads a pool
        # or a per-step product of another stage.
        self.pp = mapping.pp_size
        self.pp_rank = mapping.pp_rank
        self.first_pp = mapping.is_first_pp_rank()
        self.last_pp = mapping.is_last_pp_rank()
        if self.pp > 1:
            self.member = f"pp{self.pp}x{self.member}"
        self.rank = mapping.rank
        self.tp = tp
        self.tp_rank = mapping.tp_rank
        self.world = mapping.world_size
        # The collective group: this stage's TP (or attention-DP) ranks, as
        # global MPI ranks.
        self.group = list(mapping.tp_group)
        self.gpus_per_node = mapping.gpus_per_node
        # The IPC all-reduce workspace needs the whole group on one node; a
        # group spanning nodes (tep8 over two NVL72 trays) sums on NCCL, which
        # rides the NVLink fabric (MNNVL) between the trays.
        self.ar_one_node = len({r // self.gpus_per_node for r in self.group}) == 1
        self.attn_tp = 1 if self.dp else tp
        self.attn_rank = 0 if self.dp else mapping.tp_rank

        dt = model_config.torch_dtype
        assert dt == torch.bfloat16, f"bf16 target, engine resolved {dt}"
        assert not getattr(tc, "tie_word_embeddings", False), "untied head"
        assert not getattr(tc, "attention_bias", False), "no attention bias"
        assert tc.hidden_act == "silu", tc.hidden_act
        # fp8 32x32 dense weights plus NVFP4 routed experts: the engine reads
        # this export's global quant_algo as FP8_BLOCK_SCALES (its DeepSeek-V4.1
        # loader) or MIXED_PRECISION (a modelopt reading); the experts' NVFP4
        # comes from the checkpoint's per-layer quantization either way.
        algo = getattr(model_config.quant_config, "quant_algo", None)
        assert str(getattr(algo, "value", algo)) in ("FP8_BLOCK_SCALES", "MIXED_PRECISION"), (
            f"fp8 32x32 dense + NVFP4 experts checkpoint; engine read quant_algo {algo!r}"
        )
        # The quantization shape is a routing criterion; a KV-cache quantization
        # is an engine argument the routing context does not carry. The pools'
        # formats are this core's own (see "KV storage" above).
        assert getattr(model_config.quant_config, "kv_cache_quant_algo", None) is None, (
            "the core stores the checkpoint's own KV value sets (TRTLLM_DSV41_KV_STORAGE "
            "picks native or bf16 rows); a declared KV-cache quantization is not supported"
        )

        # ---- geometry -------------------------------------------------------
        self.num_layers = tc.num_hidden_layers
        self.hidden = tc.hidden_size
        self.vocab = tc.vocab_size
        self.eps = float(tc.rms_norm_eps)
        assert (self.num_layers, self.hidden, self.vocab) == (40, 5120, 129280), (
            self.num_layers,
            self.hidden,
            self.vocab,
        )
        assert self.eps == 1e-20, f"every RMS statistic uses eps 1e-20, got {self.eps}"
        assert cfg.num_hidden_layers == self.num_layers and cfg.hidden_size == self.hidden, (
            "mirrored top-level config disagrees with text_config"
        )
        self.heads = tc.num_attention_heads
        self.head_dim = tc.head_dim
        self.q_lora = tc.q_lora_rank
        self.o_groups = tc.o_groups
        self.o_lora = tc.o_lora_rank
        assert (self.heads, self.head_dim, self.q_lora) == (64, 512, 1280)
        assert tc.num_key_value_heads == 1, "one shared latent KV head"
        assert tc.qk_rope_head_dim == _ROPE, "RoPE on the last 64 channels"
        assert (self.o_groups, self.o_lora) == (8, 1024), "grouped low-rank output"
        assert tc.sliding_window == _WINDOW
        self.nope = self.head_dim - _ROPE  # 448
        assert self.heads % self.attn_tp == 0 and self.o_groups % self.attn_tp == 0
        # Every per-rank window -- heads, output groups, experts, Engram rows,
        # pipeline layers -- is the weight table's, derived from the mapping
        # once (``weights.dims``).
        d = _weights.MODEL_WEIGHTS.dims(self)
        self.h_loc = d.h_loc
        self.g_loc = d.g_loc
        # TP heads [h r, h r + h) (h = 64 / tp) belong to groups [g r, g r + g)
        # (g = 8 / tp): wq_b rows, the sinks and wo_a's groups split on the same
        # boundary (tp4: 16 heads, groups 2r, 2r+1; tp8: 8 heads, group r).
        assert self.h_loc * self.o_groups == self.g_loc * self.heads
        self.group_in = d.group_in  # 4096
        assert self.group_in == 4096

        # Indexer, compressor, candidates.
        assert (tc.index_n_heads, tc.index_head_dim, tc.index_topk) == (
            _IDX_HEADS,
            _IDX_DIM,
            _TOPK,
        )
        assert (tc.candidate_block_size, tc.candidate_topk_blocks) == (
            _CAND_BLOCK,
            _CAND_TOPK_BLOCKS,
        )
        self.cand_src = int(tc.candidate_source_layer_id)
        ratios = d.ratios
        self.ratios = ratios
        self.kv_sources = d.kv_sources
        self.index_sources = d.index_sources
        # The per-layer role table this assembly is written for: ratio 0
        # = SWA only; ratio 1 = SWA + layer 20's global KV. A checkpoint with
        # another table is another assembly.
        assert ratios == [0, 0] + [2] * 18 + [1] * 20, ratios
        assert self.kv_sources == (2, 8, 14, 20), self.kv_sources
        assert self.index_sources == (2, 8, 14, 20, 24, 28, 32, 36), self.index_sources
        assert self.cand_src == 20
        self.kv_src = [None] * self.num_layers
        self.idx_src = [None] * self.num_layers
        for i in range(self.num_layers):
            if ratios[i]:
                self.kv_src[i] = max(s for s in self.kv_sources if s <= i)
                self.idx_src[i] = max(s for s in self.index_sources if s <= i)
                assert ratios[self.kv_src[i]] == ratios[i] == ratios[self.idx_src[i]]
        # This stage's layers. A stage runs on its own pools only: every
        # compressed layer's KV source and index source, the candidate source of
        # the re-index layers and the Engram history owner must be on it. For
        # this checkpoint that leaves stage boundaries at layers 1 and 20 (so
        # PP2's even split, 0-19 | 20-39, and PP3 as 1 | 19 | 20).
        self.pp_layer_ids = d.pp_layer_ids
        self.local = d.local
        assert self.pp_layer_ids == list(range(self.pp_layer_ids[0], self.pp_layer_ids[-1] + 1))
        layout = DeepseekV41Layout.from_config(cfg)
        err = layout.stage_dependency_error(self.pp_layer_ids)
        if err is not None:
            raise ValueError(
                f"pipeline stage {self.pp_rank} (layers {self.pp_layer_ids[0]}-{self.pp_layer_ids[-1]}): "
                f"{err}; this checkpoint's stage boundaries can only be at layers "
                f"{list(layout.pipeline_boundaries())} (pp_partition)"
            )
        # The first attention of a later stage collapses its input with the
        # previous stage's last FFN mixes: that layer's hc_ffn triple is held too.
        self.pp_prev = d.pp_prev
        rope = getattr(tc, "rope_scaling", None) or getattr(tc, "rope_parameters", None)
        assert isinstance(rope, dict), rope
        assert rope.get("rope_type", rope.get("type")) == "yarn", rope
        assert "mscale" not in rope and "mscale_all_dim" not in rope, (
            "V4.1's YaRN is frequency interpolation only; a magnitude scale is another model"
        )
        self.yarn = {
            "factor": float(rope["factor"]),
            "original": int(rope["original_max_position_embeddings"]),
            "beta_fast": float(rope["beta_fast"]),
            "beta_slow": float(rope["beta_slow"]),
        }
        assert self.yarn == {
            "factor": 16.0,
            "original": 65536,
            "beta_fast": 32.0,
            "beta_slow": 1.0,
        }, self.yarn
        self.theta = float(tc.rope_theta)
        self.comp_theta = float(tc.compress_rope_theta)
        assert (self.theta, self.comp_theta) == (10000.0, 160000.0)

        # Hyper-connections.
        assert (tc.hc_mult, tc.hc_sinkhorn_iters, float(tc.hc_eps)) == (_HC, 20, 1e-6)
        self.hc_eps = float(tc.hc_eps)
        self.hc_k = d.hc_k  # 20480

        # MoE.
        assert tc.scoring_func == "sqrtsoftplus" and tc.topk_method == "noaux_tc"
        assert tc.norm_topk_prob and float(tc.routed_scaling_factor) == 1.5
        assert float(tc.swiglu_limit) == _SWIGLU_LIMIT
        assert tc.n_shared_experts == 1
        self.num_experts = d.num_experts
        self.topk = tc.num_experts_per_tok
        self.moe_inter = d.moe_inter
        assert (self.num_experts, self.topk, self.moe_inter) == (384, 6, 2304)
        self.route_scale = float(tc.routed_scaling_factor)
        if self.ep == tp:
            assert self.num_experts % tp == 0
        self.e_loc, self.e_off = d.e_loc, d.e_off
        self.i_loc, self.i_off = d.i_loc, d.i_off
        # fp4_block_scale_moe_runner's layout rules at the per-rank width.
        assert self.hidden % 256 == 0 and self.i_loc % 64 == 0

        # Engram.
        self.engram_layers = d.engram_layers
        assert self.engram_layers == (1, 14), self.engram_layers
        assert (tc.engram_max_ngram_size, tc.engram_n_heads, tc.engram_head_dim) == (4, 8, 256)
        self.eng_cols = d.eng_cols  # 24
        self.eng_dim = d.eng_dim
        self.eng_rows = d.eng_rows
        # host_offload: the tables are not parameters; every rank reads all
        # rows of the node's shared pinned copy (attach_engram_host_table).
        self.eng_host = d.eng_host
        self.eng_part, self.eng_start, self.eng_end = d.eng_part, d.eng_start, d.eng_end
        self._eng_table = None
        self.eng_local = bool(self.local & set(self.engram_layers))
        # The node's ranks attach the shared table together (local rank 0 fills
        # it, then a node-local barrier), so every rank of a node attaches when
        # a rank of that node holds the Engram layers.
        eng_stages = {mapping.pp_rank_of_layer(i, self.num_layers) for i in self.engram_layers}
        node = self.rank // self.gpus_per_node
        self.eng_attach = self.eng_host and any(
            r // tp in eng_stages for r in range(self.world) if r // self.gpus_per_node == node
        )
        if self.eng_host:
            # The attach's node barrier runs on the node-local communicator, whose
            # (lazy) split is collective over every rank: split it here, where every
            # rank passes, or a node whose ranks do not attach (pp2 x tp4's second
            # tray) would strand the attaching ranks in the split.
            local_mpi_comm()
        self.eng_wkv_n = d.eng_wkv_n  # 25600
        self.eng_in = d.eng_in  # 6144
        self._tc = tc
        self._ckpt_dir = getattr(cfg, "_name_or_path", None) or getattr(cfg, "name_or_path", None)

        # Vision tower + aligner (always built: the engine serves images for
        # every configuration of this checkpoint).
        _vision_geometry(cfg)

        # DSpark: the drafter's parameters exist only when the engine was
        # built with a speculative_config.
        spec = getattr(model_config, "spec_config", None)
        self.ds = None
        if spec is not None:
            assert spec.spec_dec_mode.is_dspark() and spec.draft_is_embedded_in_target, (
                "this target's mtp.* stages are the checkpoint's embedded DSpark drafter; "
                f"the engine resolved spec_dec_mode {spec.spec_dec_mode!r}"
            )
            self.ds = _dspark_geometry(tc)
            assert int(spec.max_draft_len) == self.ds.block, (
                f"DSpark drafts its trained block of {self.ds.block}; max_draft_len={spec.max_draft_len}"
            )

        # ---- parameters: the weight tables, torch.empty only ------------------
        self.w = _weights.MODEL_WEIGHTS.declare(d)
        self.w_ds = (
            _weights.DRAFT_WEIGHTS.declare(_weights.DRAFT_WEIGHTS.dims(self))
            if self.ds is not None
            else None
        )
        self._vision = VisionTower(self)
        # llm_args.disable_mm_encoder: an engine that will never see an image
        # (e.g. a generation-only disaggregated server) captures no ViT graphs.
        self.disable_mm_encoder = bool(getattr(model_config, "disable_mm_encoder", False))

        self._L: list | None = None
        self._rope_tables: dict | None = None
        self._rope_positions = 0
        self._eng: dict | None = None
        self._const: dict | None = None
        self._ar_ws = None
        self._pools: dict | None = None
        self._pool_mgr = None
        self._derive_pending = True
        self._spec_checked = False
        self._cand_binds_decode = False
        self._prefill: Target | None = None
        self._decode: Target | None = None

        # Decoder SWA bounded replay: the bounded layers are the ones
        # after the last KV-source layer, whose ratio-1 compressor and index K
        # every later layer reads; they own no compressed pool, reuse layer
        # 20's KV and top-k chain, and hold no Engram.
        self.late = max(self.kv_sources) + 1
        assert self.late == 21 and set(ratios[self.late :]) == {1}, (self.late, ratios[self.late :])
        assert max(self.engram_layers) < self.late and self.cand_src < self.late
        self.replay_window = int(tc.sliding_window)
        self.replay_requested = _env_switch(_REPLAY_ENV) is not False
        # Until the first forward takes the cache manager's resolution (_resolve_replay).
        self.replay = self.replay_requested
        self._replay_mgr = None
        self._witness = os.environ.get(_WITNESS_ENV, "").strip() == "1"
        # FP8 attention: the sparse-MLA core with E4M3 query and pools (native KV storage only).
        self.fp8_attn = _env_switch(_FP8_ATTN_ENV) is True
        self._fp8_kv_scale = [None] * self.num_layers
        if self.fp8_attn and self.rank == 0:
            sys.stdout.write(
                f"[modeling_v2 deepseek_v41] {_FP8_ATTN_ENV}=1: sparse attention computes with E4M3 "
                f"Q/K/V (per-tensor power-of-two scales) on {self.member}\n"
            )
            sys.stdout.flush()
        self._tail_records: dict = {}
        self._tail_ring: list = []
        self._ar32 = None

    def __pp_init__(self) -> None:
        """Pipeline stages are this core's own (``pp_layer_ids``, the stream
        handed over in ``forward``); the base class's per-module skip does not
        apply to a flat ParameterDict."""

    def _resolve_replay(self, mgr) -> None:
        """Decoder replay as the engine's cache manager resolved it, once per manager.

        With block reuse on it runs only beside encoder replay; the manager resolves
        both. It reads the same switch and mapping as the constructor, so it never turns
        on what the model turned off (an explicit 0)."""
        if self._replay_mgr is not None and self._replay_mgr() is mgr:
            return
        on = bool(mgr.v41_decoder_replay)
        assert self.replay_requested or not on, (
            f"the cache manager resolved decoder replay on where the model resolved {_REPLAY_ENV} off"
        )
        self.replay = on
        self._replay_mgr = weakref.ref(mgr)

    def _release_tail_records(self, mgr) -> None:
        """Drop the between-chunk records, stashes included, of requests the cache manager
        no longer holds. A request freed between two of its context chunks (cancelled, or
        evicted to be recomputed) never reaches the last chunk that drops its record; one
        that comes back starts a new context phase and a new record."""
        held = mgr.kv_cache_map
        for rid in [rid for rid in self._tail_records if rid not in held]:
            del self._tail_records[rid]

    def _ensure_ar(self, n: int, dev) -> None:
        """The int32 0..n-1 table the replay's per-row indices are views of;
        regrown only in context steps, which are never captured."""
        if self._ar32 is None or self._ar32.shape[0] < n:
            self._ar32 = torch.arange(
                max(n, 2 * _WINDOW + 1, 2 * (0 if self._ar32 is None else self._ar32.shape[0])),
                dtype=torch.int32,
                device=dev,
            )

    # ------------------------------------------------------------------
    def post_load_weights(self) -> None:
        """Post-load derivation, then the two targets.

        The per-layer operand table (views of the weights and the folded NVFP4
        call scalars), the rope tables, the Engram hash constants and residency,
        the all-reduce workspace and the load-time constant tensors the forward
        fills from. Meta is over here. The targets are built last, against the
        real weights, and bind every entry they call.

        Named for the shell hook that drives it. Not an override: that hook lives on
        DecoderModelForCausalLM, and a core derives from DecoderModel, which has none.
        """
        w = self.w
        dev = w["final_norm"].device
        L = [None] * self.num_layers
        if self.pp_prev is not None:
            i = self.pp_prev
            L[i] = {
                "hc_ffn": (w[f"l{i}_hc_ffn_fn"], w[f"l{i}_hc_ffn_scale"], w[f"l{i}_hc_ffn_base"])
            }
        for i in self.pp_layer_ids:
            d = {
                "hc_attn": (
                    w[f"l{i}_hc_attn_fn"],
                    w[f"l{i}_hc_attn_scale"],
                    w[f"l{i}_hc_attn_base"],
                ),
                "hc_ffn": (w[f"l{i}_hc_ffn_fn"], w[f"l{i}_hc_ffn_scale"], w[f"l{i}_hc_ffn_base"]),
                "attn_norm": w[f"l{i}_attn_norm"],
                "ffn_norm": w[f"l{i}_ffn_norm"],
                "wqkv_a": (w[f"l{i}_wqkv_a"], w[f"l{i}_wqkv_a_sf"]),
                "q_norm": w[f"l{i}_q_norm"],
                "kv_norm": w[f"l{i}_kv_norm"],
                "wq_b": (w[f"l{i}_wq_b"], w[f"l{i}_wq_b_sf"]),
                "sink": w[f"l{i}_sink"],
                # [G_loc, 4096, 1024] view: group g's wo_a transposed.
                "wo_a_t": torch.transpose(
                    w[f"l{i}_wo_a"].view(self.g_loc, self.o_lora, self.group_in), 1, 2
                ),
                "wo_b": (w[f"l{i}_wo_b"], w[f"l{i}_wo_b_sf"]),
                "gate_w_t": w[f"l{i}_gate_w"].t(),
                "gate_bias": w[f"l{i}_gate_bias"],
                "gate_bias_vl": w[f"l{i}_gate_bias_vl"],
                "sh_w13": (w[f"l{i}_sh_w13"], w[f"l{i}_sh_w13_sf"]),
                "sh_w2": (w[f"l{i}_sh_w2"], w[f"l{i}_sh_w2_sf"]),
                "fc1_w": w[f"l{i}_fc1_w"],
                "fc1_s": w[f"l{i}_fc1_s"].view(torch.float8_e4m3fn),
                "fc2_w": w[f"l{i}_fc2_w"],
                "fc2_s": w[f"l{i}_fc2_s"].view(torch.float8_e4m3fn),
            }
            if i in self.kv_sources:
                d["comp_w_t"] = w[f"l{i}_comp_w"].t()
                d["comp_norm"] = w[f"l{i}_comp_norm"]
                d["idx_wk_t"] = w[f"l{i}_idx_wk"].t()
                d["idx_k_norm"] = w[f"l{i}_idx_k_norm"]
            if i in self.index_sources:
                d["idx_wq_b"] = (w[f"l{i}_idx_wq_b"], w[f"l{i}_idx_wq_b_sf"])
                d["idx_wproj_t"] = w[f"l{i}_idx_wproj"].t()
            if i in self.engram_layers:
                if not self.eng_host:
                    d["eng_table"] = w[f"l{i}_eng_table"]
                    d["eng_scale"] = w[f"l{i}_eng_scale"]
                d["eng_wkv"] = (w[f"l{i}_eng_wkv"], w[f"l{i}_eng_wkv_sf"])
                d["eng_q"] = w[f"l{i}_eng_q"]
                d["eng_k"] = w[f"l{i}_eng_k"]
            # NVFP4 routed-expert scalars (modelopt stores reciprocals:
            # input_scale = amax/(448*6)). FC1's activation global scale is one
            # number for the layer: the max over all 384 w1/w3 input scales
            # (R-M2), identical on every rank. FC2's is LAYER-level too (C4:
            # a per-expert one saturates the NVFP4 block scales).
            isc1, isc3 = w[f"l{i}_e_isc1"], w[f"l{i}_e_isc3"]
            ws1, ws3 = w[f"l{i}_e_ws1"], w[f"l{i}_e_ws3"]
            assert torch.equal(ws1, ws3), (
                f"layer {i}: w1/w3 weight_scale_2 differ; the fused FC1 and its "
                "clamp need one dequant scale per expert"
            )
            in1 = torch.maximum(isc1.max(), isc3.max())
            in2 = w[f"l{i}_e_isc2"].max()
            win = slice(self.e_off, self.e_off + self.e_loc)
            a1 = (in1 * ws1[win]).contiguous()
            d["g1"] = (1.0 / in1).reshape(1).contiguous()
            d["o1g"] = a1
            d["o1"] = (a1 / in2).contiguous()
            d["o2"] = (in2 * w[f"l{i}_e_ws2"][win]).contiguous()
            # swiglu_limit in the FC1 accumulator domain.
            d["clamp"] = (_SWIGLU_LIMIT / a1).contiguous()
            L[i] = d
        self._L = L
        self._rope_positions = 0
        self._ensure_rope(dev, 65536)
        tj = os.path.join(str(self._ckpt_dir), "tokenizer.json")
        assert os.path.isfile(tj), (
            f"Engram's compressed vocab is built from the checkpoint tokenizer; {tj} not found"
        )
        self._eng = _engram_layout(self._tc, tj, dev)
        self._engram_residency()
        self._const = {
            "neg1_i32": torch.full((1, 1), -1, dtype=torch.int32, device=dev),
            # The index top-k's per-row offsets (window-local positions: 0).
            "zero_i32_rows": torch.zeros((_TOPK_MAX_ROWS,), dtype=torch.int32, device=dev),
            "zero_bf16": torch.zeros((1, 1), dtype=torch.bfloat16, device=dev),
            "ape": torch.zeros((2, self.head_dim), dtype=torch.float32, device=dev),
            "dummy_comp": torch.zeros((1, 1, 64, self.head_dim), dtype=torch.bfloat16, device=dev),
            "attn_ws": torch.zeros(_ATTN_WS_BYTES, dtype=torch.uint8, device=dev),
            "false_b": torch.zeros((1,), dtype=torch.bool, device=dev),
            "true_b": torch.ones((1,), dtype=torch.bool, device=dev),
            # The main KV's NVFP4 has no global scale: the dequantizer's is 1.
            "one_f32": torch.ones((1,), dtype=torch.float32, device=dev),
        }
        if self.fp8_attn:
            self._derive_fp8_kv_scales()
            self._const["dummy_comp8"] = torch.zeros(
                (1, 1, 64, self.head_dim), dtype=torch.float8_e4m3fn, device=dev
            )
        self._vision.post_load_weights()
        if not self.dp and self.ar_one_node:
            from tensorrt_llm._torch.distributed.ops import get_allreduce_workspace
            from tensorrt_llm.mapping import Mapping

            self._ar_ws = get_allreduce_workspace(
                Mapping(
                    world_size=self.world,
                    rank=self.rank,
                    gpus_per_node=self.gpus_per_node,
                    tp_size=self.tp,
                    pp_size=self.pp,
                )
            )
        # Decoder replay: row j of a tail keeps window columns 0..j (its keys
        # r .. r + j); column 0 alone names a row's own window slot.
        w128 = torch.arange(_WINDOW, device=dev)
        self._const["tri"] = (w128.view(1, -1) <= w128.view(-1, 1)).contiguous()
        self._const["col0"] = (w128 == 0).view(1, -1).contiguous()
        self._prefill = PrefillTarget(self)
        self._decode = DecodeTarget(self)

    def _derive_fp8_kv_scales(self) -> None:
        """FP8 attention: one power-of-two KV scale per layer, from the norm weights
        (``_FP8_KV_BOUND``). One attention call takes one scale for both of its pools, so
        every layer that reads a compressed source shares that source's scale, taken over
        the source's compressed bound and every reader's window bound; a window-only layer
        has its own. With a power-of-two scale the window's MXFP8 values stay exact in
        E4M3 wherever they land in E4M3's normal range."""
        w = self.w
        # This pipeline stage's layers (a group's source is on the stage with its readers).
        bound = {
            i: _FP8_KV_BOUND * float(w[f"l{i}_kv_norm"].abs().max()) for i in self.pp_layer_ids
        }
        groups: dict = {}
        for i in self.pp_layer_ids:
            groups.setdefault(self.kv_src[i] if self.ratios[i] else -1 - i, []).append(i)
        for key, layers in groups.items():
            b = max(bound[i] for i in layers)
            if key >= 0:
                b = max(b, _FP8_KV_BOUND * float(w[f"l{key}_comp_norm"].abs().max()))
            s = 2.0 ** math.ceil(math.log2(b / 448.0))
            for i in layers:
                self._fp8_kv_scale[i] = s
        if self.rank == 0:
            sys.stdout.write(
                "[modeling_v2 deepseek_v41] FP8 attention KV scales (log2, per layer): "
                + " ".join(str(int(math.log2(s))) for s in self._fp8_kv_scale if s is not None)
                + "\n"
            )
            sys.stdout.flush()

    def _ensure_rope(self, dev, positions: int) -> None:
        """Both rope tables, sized to ``positions`` rows. Every row depends
        only on its own position, so a longer table is the same table with
        more rows; regrown only outside capture (first forward)."""
        if positions <= self._rope_positions:
            return
        self._rope_positions = positions
        self._rope_tables = {
            0: _rope_table(self.theta, positions, None, dev),
            1: _rope_table(self.comp_theta, positions, self.yarn, dev),
        }

    def _table(self, ratio: int) -> torch.Tensor:
        # Ratio-0 layers: theta 10000, no YaRN. Compressed layers: theta
        # 160000 with YaRN (q, window kv, compressed latent, index q/k).
        return self._rope_tables[0 if ratio == 0 else 1]

    # ------------------------------------------------------------------
    def _derive_first_step(self, md, position_ids) -> None:
        """What the first step fixes, run once and always: the engine-side layout this
        assembly is written for (the cache manager's, fixed at engine construction), and
        the tables sized by the engine's own limits.

        Unconditional, unlike the targets' step-contract probe: the rope table and the
        top-k offsets decide what the model computes, and the layout checks are facts the
        routing context cannot see."""
        if not self._derive_pending:
            return
        assert isinstance(md, DeepseekV4TrtllmAttentionMetadata), type(md)
        assert md.is_deepseek_v41, "the cache manager is not in DeepSeek-V4.1 layout mode"
        assert position_ids.dtype == torch.int32, position_ids.dtype
        assert md.tokens_per_block == _PAGE, md.tokens_per_block
        layout = md.kv_cache_manager.v41_layout
        assert list(layout.compress_ratios) == self.ratios, "manager layout != checkpoint roles"
        for i in range(self.num_layers):
            if self.ratios[i]:
                assert layout.kv_source(i) == self.kv_src[i], (i, layout.kv_source(i))
                assert layout.index_source(i) == self.idx_src[i], (i, layout.index_source(i))
        assert layout.engram_token_owner == self.engram_layers[0]
        assert md.effective_beam_width == 1, "beam search is not implemented"
        assert not md.is_cross, "cross attention is not implemented"
        self._ensure_rope(self.w["final_norm"].device, int(md.max_seq_len) + 64)
        # One zero offset per row of the largest step (outside capture).
        max_t = int(md.swa_local_indices_cuda.shape[0])
        if self._const["zero_i32_rows"].shape[0] < max_t:
            self._const["zero_i32_rows"] = torch.zeros(
                (max_t,), dtype=torch.int32, device=self.w["final_norm"].device
            )
        # Candidate masking can bind in a decode step only when a sequence may
        # hold more than 2048 x 8 ratio-1 compressed positions: a per-engine
        # constant (the paged logits' static width), so the decode graph is
        # built with or without it once.
        self._cand_binds_decode = md.max_compressed_kv_lens[1] > _CAND_TOPK_BLOCKS * _CAND_BLOCK
        self._derive_pending = False

    def _get_pools(self, md) -> dict:
        """Pool views of the engine's cache manager, keyed on the manager
        object (a KV-size-estimation manager precedes the final one). Views
        only: the pools are the manager's and never move. Under native KV
        storage this also allocates, once per manager, the two bf16 read
        buffers the attention is handed (see ``_native_window``)."""
        mgr = md.kv_cache_manager
        if self._pools is not None and self._pool_mgr is not None and self._pool_mgr() is mgr:
            return self._pools
        T = DeepseekV4AttentionType
        layout = mgr.v41_layout
        native = layout.kv_storage == DEEPSEEK_V41_KV_STORAGE_NATIVE
        assert native or layout.kv_storage == DEEPSEEK_V41_KV_STORAGE_BF16, layout.kv_storage
        if self.fp8_attn and not native:
            raise ValueError(
                f"{_FP8_ATTN_ENV}=1 reads the native KV pools; it does not run with KV storage "
                f"{layout.kv_storage!r}"
            )
        H = self.head_dim

        def view(v, shape):
            # A pool is only ever re-viewed, never copied: a copy would take
            # the writes away from the pool the attention reads.
            out = torch.reshape(v, shape)
            assert out.data_ptr() == v.data_ptr() and out.numel() == v.numel(), (
                "pool view is not a view of the manager's pool"
            )
            return out

        # bf16 containers: swa3/comp3 are written, swa4/comp4 read by the
        # attention. Native: (codes, scales) views of the 528 / 288 B rows are
        # written, and read back through the dequantizers.
        swa3, swa4, swa_q = {}, {}, {}
        for i in self.pp_layer_ids:
            if native:
                codes, scales = mgr.get_v41_native_kv_views(i, T.SWA)
                assert codes.dtype == torch.uint8 and tuple(codes.shape[1:]) == (_PAGE, H), (
                    codes.shape
                )
                assert tuple(scales.shape[1:]) == (_PAGE, H // _SF32), scales.shape
                swa_q[i] = (codes, scales)
            else:
                v = mgr.get_buffers(i, T.SWA)
                assert v.dtype == torch.bfloat16 and tuple(v.shape[1:]) == (_PAGE, H), v.shape
                swa3[i] = v
                swa4[i] = view(v, [v.shape[0], 1, v.shape[1], v.shape[2]])
        comp3, comp4, comp_q, comp_q4, comp_q1, idxk, ckv, csc = {}, {}, {}, {}, {}, {}, {}, {}
        bytes_per_token2 = 0  # twice the persistent KV bytes per token, as allocated
        for o in self.kv_sources:
            if o not in self.local:
                continue
            bk = _PAGE // self.ratios[o]
            v = mgr.get_buffers(o, T.COMPRESS)
            if native:
                codes, scales = mgr.get_v41_native_kv_views(o, T.COMPRESS)
                assert codes.dtype == torch.uint8 and tuple(codes.shape[1:]) == (bk, H // 2), (
                    codes.shape
                )
                assert tuple(scales.shape[1:]) == (bk, H // _SF16), scales.shape
                comp_q[o] = (codes, scales)
                # The dequantizer's [pages, page_size, 1, W] operands.
                comp_q4[o] = (
                    view(codes, [codes.shape[0], bk, 1, H // 2]),
                    view(scales, [scales.shape[0], bk, 1, H // _SF16]),
                )
                # The same rows as pages of one row: a slot number
                # (page * bk + slot) is then its own page index.
                comp_q1[o] = (
                    view(codes, [codes.shape[0] * bk, 1, 1, H // 2]),
                    view(scales, [scales.shape[0] * bk, 1, 1, H // _SF16]),
                )
            else:
                assert v.dtype == torch.bfloat16 and tuple(v.shape[1:]) == (bk, H), v.shape
                comp3[o] = v
                comp4[o] = view(v, [v.shape[0], 1, v.shape[1], v.shape[2]])
            idxk[o] = mgr.get_indexer_k_cache_buffers(o)
            assert idxk[o].dtype == torch.uint8 and tuple(idxk[o].shape[1:]) == (bk, 1, 68), idxk[
                o
            ].shape
            if self.ratios[o] == 2:
                ckv[o] = mgr.get_buffers(o, T.COMPRESSOR_KV)
                csc[o] = mgr.get_buffers(o, T.COMPRESSOR_SCORE)
                assert ckv[o].dtype == torch.float32 and csc[o].dtype == torch.float32
            # A ratio-r owner stores one main-KV row and one index-K row per r tokens.
            row_bytes = v.shape[-1] * v.element_size() + idxk[o].shape[-1] * idxk[o].element_size()
            bytes_per_token2 += 2 * row_bytes // self.ratios[o]
        tok = None
        if self.eng_local:
            tok = mgr.get_buffers(layout.engram_token_owner, T.ENGRAM_TOKEN)
            assert (
                tok.dtype == torch.int32
                and tok.shape[1] == _PAGE
                and tok.numel() == tok.shape[0] * _PAGE
            )
            tok = view(tok, [tok.shape[0], tok.shape[1]])
        local = {i: mgr.layer_offsets[i] for i in self.pp_layer_ids}
        self._pools = {
            "native": native,
            "swa3": swa3,
            "swa4": swa4,
            "swa_q": swa_q,
            "comp3": comp3,
            "comp4": comp4,
            "comp_q": comp_q,
            "comp_q4": comp_q4,
            "comp_q1": comp_q1,
            "idxk": idxk,
            "ckv": ckv,
            "csc": csc,
            "tok": tok,
            "local": local,
        }
        if native:
            self._pools.update(self._native_read_buffers(md))
        self._pool_mgr = weakref.ref(mgr)

        # Persistent KV bytes per token as allocated (windowed state excluded).
        assert bytes_per_token2 % 2 == 0, bytes_per_token2
        bpt = bytes_per_token2 // 2
        assert bpt == mgr.get_cache_bytes_per_token(), (bpt, mgr.get_cache_bytes_per_token())
        lines = [
            f"MODELING_V2 deepseek_v41 rank {self.rank}: KV storage {layout.kv_storage}, "
            f"persistent KV {bpt} B/token\n"
        ]
        if os.environ.get(_WITNESS_ENV) == "1":
            lines.append(f"MODELING_V2_WITNESS kv_bytes_per_token={bpt}\n")
        sys.stdout.write("".join(lines))
        sys.stdout.flush()
        return self._pools

    def _native_read_buffers(self, md) -> dict:
        """The bf16 rows the attention reads under native KV storage, allocated
        once per manager (outside capture) and zero-filled, so every row the
        kernel may load is finite:

        * ``swa_c`` [max_tokens * 128, 512]: query t's window column c at row
          ``t * 128 + c``; ``win_idx`` [max_tokens, 128] is that numbering, the
          window half of every renumbered index table;
        * ``comp_c``: as many rows as the smaller of its two forms needs at
          the engine's capacity, holding either form (``_comp_form`` picks one
          per step):

          - per request (``comp_out`` / ``comp_c4_req`` / ``comp_tbl``):
            request b's compressed entry e at row ``b * L_r + e``,
            ``L_r = cols * 128 / r``, viewed per ratio as the dequantizer's
            ``[seqs, L_r, 1, 512]`` output and as the attention's
            ``[seqs * cols, 1, 128 / r, 512]`` pool; ``comp_tbl[r]`` [seqs,
            cols] (row b = ``b * cols + j``) is the page table that numbering
            implies, so the index op maps top-k entry e of request b onto row
            b * L_r + e. Each KV source dequantizes every entry of every
            request of the step (``_native_compressed``). It holds the first
            ``comp_req_fit`` request slots;
          - per query (``topk_out`` / ``comp_c4_q``): query t's top-k column k
            at row ``t * 512 + k`` (``topk_ren`` is that numbering),
            dequantized from the pool slot the index op names
            (``_native_topk``) once per top-k set. It holds ``comp_q_fit``
            query rows.

          The attention reads the same bf16 values in the same column order
          from either form.
        """
        assert not torch.cuda.is_current_stream_capturing(), "read buffers are allocated eagerly"
        dev = self.w["final_norm"].device
        H = self.head_dim
        max_t = md.swa_local_indices_cuda.shape[0]
        # FP8 attention reads E4M3 rows (per-tensor scaled); otherwise the exact bf16 values.
        rdt = torch.float8_e4m3fn if self.fp8_attn else torch.bfloat16
        swa_c = torch.zeros((max_t * _WINDOW, H), dtype=rdt, device=dev)
        win_idx = torch.arange(max_t * _WINDOW, dtype=torch.int32, device=dev).view(max_t, _WINDOW)
        # The ratios of this stage's KV sources (a pipeline stage may hold none:
        # its buffers are then sized for the smallest ratio of the layout).
        ratios = sorted({self.ratios[o] for o in self.kv_sources if o in self.local})
        rmin = min(ratios) if ratios else min(md.compress_block_tables)
        seqs, cols = md.compress_block_tables[rmin].shape
        for r in ratios:
            assert tuple(md.compress_block_tables[r].shape) == (seqs, cols), (
                md.compress_block_tables[r].shape
            )
        out = {
            "swa_c": swa_c,
            "swa_c4": torch.reshape(swa_c, [max_t, 1, _WINDOW, H]),
            "win_idx": win_idx,
        }
        rows_request = seqs * cols * (_PAGE // rmin)
        rows_topk = max_t * _TOPK
        rows = min(rows_request, rows_topk)
        comp_c = torch.zeros((rows, H), dtype=rdt, device=dev)
        req_fit = rows // (cols * (_PAGE // rmin))
        q_fit = rows // _TOPK
        comp_out, comp_c4_req, comp_c4_q = {}, {}, {}
        for r in ratios:
            bk = _PAGE // r
            n = req_fit * cols * bk
            comp_out[r] = torch.reshape(comp_c[:n], [req_fit, cols * bk, 1, H])
            comp_c4_req[r] = torch.reshape(comp_c[:n], [req_fit * cols, 1, bk, H])
            comp_c4_q[r] = torch.reshape(comp_c[: q_fit * _TOPK], [q_fit * _TOPK // bk, 1, bk, H])
        tbl = torch.arange(req_fit * cols, dtype=torch.int32, device=dev).view(req_fit, cols)
        out.update(
            comp_cols=cols,
            comp_req_fit=req_fit,
            comp_q_fit=q_fit,
            comp_out=comp_out,
            comp_c4_req=comp_c4_req,
            comp_tbl={r: tbl for r in ratios},
            comp_c4_q=comp_c4_q,
            topk_out=torch.reshape(comp_c[: q_fit * _TOPK], [q_fit, _TOPK, 1, H]),
            topk_ren=torch.arange(q_fit * _TOPK, dtype=torch.int32, device=dev).view(q_fit, _TOPK),
            # The index op's pool slots, contiguous: the dequantizer's page table.
            topk_slots=torch.full((q_fit, _TOPK), -1, dtype=torch.int32, device=dev),
            topk_cols=torch.full((q_fit,), _TOPK, dtype=torch.int32, device=dev),
        )
        sys.stdout.write(
            f"MODELING_V2 deepseek_v41 rank {self.rank}: native compressed read buffer {rows} rows: "
            f"per request {req_fit} of {seqs} request slots (x cols {cols} x {_PAGE // rmin}), "
            f"per query {q_fit} of {max_t} rows (x {_TOPK})\n"
        )
        sys.stdout.flush()
        return out

    def _comp_form(self, pools, md, st, tail) -> str:
        """The native compressed read buffer's form for this step, "request" or
        "topk" (``_native_read_buffers``): one per step, as both forms share
        the buffer. A form is usable when the step fits it (every request slot
        of the step, every query row of each pass); of the usable ones, the one
        that dequantizes fewer rows:

        * per request, every compressed entry of every request, once per KV
          source: sum over sources and requests of kv_len // ratio;
        * per query, 512 rows per query row once per top-k set: the index
          sources over the step's rows, plus, in a bounded step, the layers
          past L over the decoder rows (whose first layer re-reads layer L's
          set).

        A generation-only step is the same at every length in one CUDA graph,
        so it prices each request at half ``max_seq_len``: the per-request form
        at the short envelopes, the per-query form for long ones. ``st["nc"]``
        is the step's context-request count as the calling target states it."""
        B, T, nc = st["B"], st["T"], st["nc"]
        bounded = tail is not None and tail.bound
        rows = max(T, min(tail.rows, int(md.swa_local_indices_cuda.shape[0]))) if bounded else T
        fits_req = B <= pools["comp_req_fit"]
        fits_q = rows <= pools["comp_q_fit"]
        assert fits_req or fits_q, (B, rows, pools["comp_req_fit"], pools["comp_q_fit"])
        if not (fits_req and fits_q):
            form = "request" if fits_req else "topk"
        else:
            if nc:
                kv = (md.cached_token_lens_cpu[:B] + md.seq_lens_kv[:B]).tolist()
            else:
                kv = [int(md.max_seq_len) // 2] * B
            cost_req = sum(
                n // self.ratios[o] for o in self.kv_sources if o in self.local for n in kv
            )
            index_local = [i for i in self.index_sources if i in self.local]
            early = sum(1 for i in index_local if i < self.late)
            if bounded:
                cost_q = _TOPK * (early * T + (len(index_local) - early + 1) * tail.rows)
            else:
                cost_q = _TOPK * len(index_local) * T
            form = "request" if cost_req <= cost_q else "topk"
        _rv("native_read.form", form, B, T, rows)
        return form

    # ------------------------------------------------------------------
    def attach_engram_host_table(self, checkpoint_dir) -> None:
        """host_offload: map the node's shared host copy of both Engram tables.

        Collective over the node's ranks (``open_shared_engram_table``). With
        the default ``vmm`` backend, local rank 0 imports the table from a
        process on this node that already serves it, or creates and fills the
        CUDA VMM host-NUMA allocations from the checkpoint; after a node
        barrier every rank imports and maps it, and the views are CUDA views of
        host memory. With the ``shm`` backend, local rank 0 fills (or reuses)
        the ``/dev/shm`` file and every rank maps and registers it, and the
        views are pinned CPU tensors whose mapped device pointer is resolved
        here, before any CUDA-graph capture. Each table is checked against the
        checkpoint geometry. The views must not move or go away while a graph
        that captured a gather can replay, so the mapping stays until the
        process exits.
        """
        assert self.eng_attach and self._eng_table is None
        dev = torch.device("cuda", torch.cuda.current_device())
        table = _engram_host.open_shared_engram_table(
            str(checkpoint_dir),
            local_rank=local_mpi_rank(),
            barrier=local_mpi_barrier,
            device=dev,
            owned=True,
        )
        assert tuple(table.layers) == self.engram_layers, (table.layers, self.engram_layers)
        for li, i in enumerate(self.engram_layers):
            wt, sc = table.weight(i), table.scale(i)
            assert tuple(wt.shape) == (self.eng_rows[li], self.eng_dim), (i, tuple(wt.shape))
            assert tuple(sc.shape) == (self.eng_rows[li], self.eng_dim // 32), (i, tuple(sc.shape))
            assert wt.dtype == torch.float8_e4m3fn and sc.element_size() == 1, (wt.dtype, sc.dtype)
        table.check_gather_views(dev)
        self._eng_table = table
        atexit.register(table.close, remove_if_owned=True)
        sys.stdout.write(
            f"MODELING_V2 deepseek_v41 rank {self.rank}: Engram host table {table.describe()}\n"
        )
        sys.stdout.flush()

    def _engram_residency(self) -> None:
        """Hand the Engram layers their tables and print this rank's device
        memory after the weight load, the residency witness's reading."""
        eng = [i for i in self.engram_layers if i in self.local]
        if self.eng_attach and self._eng_table is None:
            # A load path that never ran the weight table's load (LoadFormat.DUMMY).
            self.attach_engram_host_table(self._ckpt_dir)
        if self.eng_host:
            for i in eng:
                self._L[i]["eng_table"] = self._eng_table.weight(i)
                self._L[i]["eng_scale"] = self._eng_table.scale(i).view(torch.uint8)
        # Host-resident tables are CUDA views under the vmm backend; none of their bytes are on the device.
        dev_bytes = (
            0
            if self.eng_host
            else sum(
                self._L[i][k].numel() * self._L[i][k].element_size()
                for i in eng
                for k in ("eng_table", "eng_scale")
                if self._L[i][k].is_cuda
            )
        )
        torch.cuda.synchronize()
        sys.stdout.write(
            f"MODELING_V2 deepseek_v41 rank {self.rank}: Engram tables "
            f"{'host (shared pinned)' if self.eng_host else 'device (row shard)'}, "
            f"{dev_bytes / 2**20:.1f} MiB of them on the device; after weight load "
            f"__DEVICE_MEMORY_MIB__={torch.cuda.memory_allocated() / 2**20:.1f}\n"
        )
        sys.stdout.flush()

    def _dp_rows(self, md, num_tokens: int) -> int:
        """Uniform row count of the dep4 MoE / Engram round trips: the max of
        the engine's per-rank token counts (host ints, identical on every rank)."""
        counts = [int(n) for n in md.all_rank_num_tokens]
        assert len(counts) == self.tp, counts
        assert counts[self.tp_rank] == num_tokens, (self.tp_rank, num_tokens, counts)
        return max(counts)

    # ------------------------------------------------------------------
    def _select_target(self, attn_metadata) -> Target:
        """Which target runs this step."""
        return self._decode if phase_of(attn_metadata) is Phase.DECODE else self._prefill

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: torch.IntTensor | None = None,
        position_ids: torch.IntTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        lora_params: dict | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Derive what the first step fixes, keep the per-manager state current, then
        route the step to its target."""
        assert self._prefill is not None, "load_weights must run before forward"
        assert lora_params is None, "LoRA is not implemented by this target"
        assert inputs_embeds is None, (
            "this target embeds token ids itself; images arrive as multimodal_params"
        )
        assert input_ids is not None and position_ids is not None
        md = attn_metadata
        advance_step_generation()
        self._derive_first_step(md, position_ids)
        spec_metadata = kwargs.get("spec_metadata")
        if spec_metadata is not None and not self._spec_checked:
            assert self.ds is not None, (
                "spec_metadata reached a core built without the DSpark drafter"
            )
            assert tuple(spec_metadata.layers_to_capture or ()) == self.ds.targets, (
                f"DSpark reads the input stream of layers {self.ds.targets}; the runtime "
                f"captures {spec_metadata.layers_to_capture}"
            )
            self._spec_checked = True
        self._resolve_replay(md.kv_cache_manager)
        if self._tail_records:
            self._release_tail_records(md.kv_cache_manager)
        return self._select_target(md).forward(
            md, input_ids=input_ids, position_ids=position_ids, **kwargs
        )


# The MXFP8 x MXFP8 linears of a trunk layer, by the operand-table key each binds.
_MX_LINEARS = ("wqkv_a", "wq_b", "wo_b", "idx_wq_b", "sh_w13", "sh_w2", "eng_wkv")


class PrefillTarget(Target):
    """The general case: context rows, and possibly generation rows beside them.

    Owns everything a context step needs beyond decode: the vision splice, the
    encoder replay's compressor rows and Engram history, the context attention,
    the context indexer with its candidate-block masks, and the decoder SWA
    bounded replay (``_DecoderTail``).
    """

    def __init__(self, core: "DeepseekV41UnifiedModelingV2") -> None:
        """Bind every op this target calls, once, against the real weights."""
        super().__init__(core)
        L = core._L
        eps = core.eps
        e = core._eng

        # MXFP8 linears: one activation quantizer, one GEMM per weight role.
        self._mxq = Mxfp8Quantize()
        self._mx = {name: Mxfp8Mxfp8Gemm() for name in _MX_LINEARS}
        self._mx_shape: dict = {}

        # Norms.
        self._attn_norm = FlashinferRmsnorm()
        self._ffn_norm = FlashinferRmsnorm()
        self._q_norm = FlashinferRmsnorm()
        self._kv_norm = FlashinferRmsnorm()
        self._comp_norm = FlashinferRmsnorm()
        self._idx_k_norm = FlashinferRmsnorm()
        for op in (
            self._attn_norm,
            self._ffn_norm,
            self._q_norm,
            self._kv_norm,
            self._comp_norm,
            self._idx_k_norm,
        ):
            op.bind_const(eps=eps)
        self._final_norm = FlashinferRmsnorm()
        self._final_norm.bind_const(weight=core.w["final_norm"], eps=eps)

        # Attention: RoPE, the window and compressed KV writes, their native
        # read-back, the index table, the sparse MLA core, the grouped output.
        self._rope = MlaRopeInplace()
        self._swa_write = Mxfp8FakeQuantize()
        self._comp_write = Nvfp4FakeQuantize()
        self._swa_read = Mxfp8Dequantize()
        self._swa_read8 = Mxfp8ToE4m3()
        self._comp_read = Nvfp4KvToBf16()
        self._comp_read8 = Nvfp4KvToE4m3()
        self._l2g = Dsv4LocalToGlobalIndices()
        self._q_e4m3 = E4m3QuantizePow2Dynamic()
        self._attn = TrtllmSparseMlaDsv4()
        self._attn.bind_const(
            workspace_buffer=core._const["attn_ws"],
            softmax_scale=core.head_dim**-0.5,
            output_scale=1.0,
        )
        self._attn8 = TrtllmSparseMlaDsv4Fp8()
        self._attn8.bind_const(workspace_buffer=core._const["attn_ws"])
        self._wo_a = BmmOut()

        # Compressor and index K (the KV sources).
        self._comp_mm = CublasMm()
        self._comp_prefill = CompressorPrefillReduction()
        self._comp_paged = CompressorPagedKvCompress()
        for op in (self._comp_prefill, self._comp_paged):
            op.bind_const(
                ape=core._const["ape"], page_size=_PAGE, head_dim=core.head_dim, compress_ratio=2
            )
        self._idx_wk = CublasMm()
        # MXFP4 (UE8M0 per 32, linear scales): the index K and the index query.
        self._mxfp4 = Fp4Quantize()
        self._mxfp4.bind_const(
            global_scale=None, sf_vec_size=_SF32, sf_use_ue8m0=True, is_sf_swizzled_layout=False
        )
        self._idx_scatter = IndexerKCacheScatter()

        # Indexer (the index sources) and the candidate-block masks.
        self._idx_wproj = CublasMm()
        self._idx_gather = IndexerKCacheGather()
        self._idx_gather.bind_const(head_dim=_IDX_DIM // 2)
        self._mqa_logits = DeepgemmFp8Fp4MqaLogits()
        self._mqa_logits.bind_const(clean_logits=True, max_seqlen_k=0, logits_dtype=torch.float32)
        self._paged_logits = DeepgemmFp8Fp4PagedMqaLogits()
        self._paged_logits.bind_const(logits_dtype=torch.float32)
        self._paged_meta = DeepgemmPagedMqaLogitsMetadata()
        self._cand = CandidateBlockSelect()
        self._cand.bind_const(topk_blocks=_CAND_TOPK_BLOCKS, block_size=_CAND_BLOCK, next_n=1)
        self._topk = FlashinferTopKRaggedDeterministic()
        self._topk.bind_const(k=_TOPK)

        # MoE: router (fp32 logits), routing with gate.bias and gate.bias_vl,
        # the NVFP4 activation quantizer and runner, the shared expert's SwiGLU.
        self._router = CublasMm()
        self._router.bind_const(bias=None, out_dtype=torch.float32)
        self._gate = GateForward()
        self._gate_vl = GateForward()
        for op in (self._gate, self._gate_vl):
            op.bind_const(topk=core.topk, route_scale=core.route_scale)
        self._moe_quant = Fp4Quantize()
        self._moe_quant.bind_const(
            sf_vec_size=_SF16, sf_use_ue8m0=False, is_sf_swizzled_layout=_SF_LINEAR
        )
        self._moe_runner = Fp4BlockScaleMoeRunner()
        self._moe_runner.bind_const(
            routing_logits=None,
            routing_bias=None,
            gemm1_bias=None,
            gemm1_alpha=None,
            gemm1_beta=None,
            gemm2_bias=None,
            num_experts=core.num_experts,
            top_k=core.topk,
            n_group=None,
            topk_group=None,
            intermediate_size=core.i_loc,
            local_expert_offset=core.e_off,
            local_num_experts=core.e_loc,
            routed_scaling_factor=None,
            routing_method_type=_ROUTING_METHOD_INERT,
            do_finalize=True,
            act_type=_ACT_SWIGLU,
        )
        self._silu = SiluAndMul()
        self._silu.bind_const(swiglu_limit=_SWIGLU_LIMIT)

        # Collectives: the TP (or attention-DP) group of this stage.
        self._allgather = Allgather()
        self._allgather.bind_const(sizes=None, group=core.group)
        self._reducescatter = Reducescatter()
        self._reducescatter.bind_const(sizes=None, group=core.group)
        # fp32 sums on AUTO (the IPC workspace) when the group is on one node,
        # NCCL across nodes; bf16 sums on NCCL.
        self._ar32_strategy = _AR_AUTO if core._ar_ws is not None else _AR_NCCL
        self._allreduce32 = Allreduce()
        self._allreduce16 = Allreduce()
        for op, ws, strategy in (
            (self._allreduce32, core._ar_ws, self._ar32_strategy),
            (self._allreduce16, None, _AR_NCCL),
        ):
            op.bind_const(
                residual=None,
                norm_weight=None,
                scale=None,
                bias=None,
                workspace=ws,
                group=core.group,
                strategy=strategy,
                op=0,
                eps=1e-6,
                trigger_completion_at_end=True,
            )
        self._pp_send = PpSendTensors()
        self._pp_recv = PpRecvTensors()

        # Engram: the hash constants, the table gather, the gate.
        self._eng_hash = EngramHash()
        self._eng_hash.bind_const(
            token_map=e["token_map"],
            multipliers=e["multipliers"],
            primes=e["primes"],
            offsets=e["offsets"],
            pad_id=e["pad_id"],
        )
        self._eng_gather = EngramGather()
        self._eng_gate = EngramGate()
        self._eng_gate.bind_const(eps=eps, clamp_value=1e-6)

        # Hyper-connections: per sublayer the mix GEMM + squared sums, the fused
        # sinkhorn, and the lagged-pre collapse; the post mapping.
        self._hc_attn_gemm = MhcGemmSqrsumFma()
        self._hc_ffn_gemm = MhcGemmSqrsumFma()
        for op in (self._hc_attn_gemm, self._hc_ffn_gemm):
            op.bind_const(N=_MIX, K=core.hc_k)
        self._hc_attn_fuse = MhcBigFuse()
        self._hc_ffn_fuse = MhcBigFuse()
        for op in (self._hc_attn_fuse, self._hc_ffn_fuse):
            op.bind_const(
                K=core.hc_k,
                hidden_size=0,
                rms_eps=eps,
                hc_pre_eps=core.hc_eps,
                hc_sinkhorn_eps=core.hc_eps,
                hc_post_mult_value=2.0,
                sinkhorn_repeat=20,
                num_splits=1,
                block_size=128,
            )
        self._hc_pre_attn = MhcHcHeadApply()
        self._hc_pre_ffn = MhcHcHeadApply()
        self._hc_pre_final = MhcHcHeadApply()
        for op in (self._hc_pre_attn, self._hc_pre_ffn, self._hc_pre_final):
            op.bind_const(
                mult=_HC, hidden_size=core.hidden, K=core.hc_k, norm_eps=eps, eps=core.hc_eps
            )
        self._hc_post = MhcPostMapping()

        for i in core.pp_layer_ids:
            d = L[i]
            for name in _MX_LINEARS:
                if name in d:
                    self._bind_mx(name, i, d[name])
            self._attn_norm.bind_layered(i, weight=d["attn_norm"])
            self._ffn_norm.bind_layered(i, weight=d["ffn_norm"])
            self._q_norm.bind_layered(i, weight=d["q_norm"])
            self._kv_norm.bind_layered(i, weight=d["kv_norm"])
            self._attn.bind_layered(i, sinks=d["sink"])
            if core.fp8_attn:
                s_kv = core._fp8_kv_scale[i]
                self._attn8.bind_layered(i, sinks=d["sink"], bmm2_scale=s_kv)
                # The call's QK scale is softmax scale x query scale x KV scale;
                # the quantizer writes it with its query scale folded in.
                self._q_e4m3.bind_layered(i, scale_mult=core.head_dim**-0.5 * s_kv)
            self._wo_a.bind_layered(i, b=d["wo_a_t"])
            self._router.bind_layered(i, mat_b=d["gate_w_t"])
            self._gate.bind_layered(i, bias=d["gate_bias"])
            self._gate_vl.bind_layered(i, bias=d["gate_bias_vl"])
            self._moe_quant.bind_layered(i, global_scale=d["g1"])
            self._moe_runner.bind_layered(
                i,
                gemm1_weights=d["fc1_w"],
                gemm1_weights_scale=d["fc1_s"],
                gemm1_clamp_limit=d["clamp"],
                gemm2_weights=d["fc2_w"],
                gemm2_weights_scale=d["fc2_s"],
                output1_scale_scalar=d["o1"],
                output1_scale_gate_scalar=d["o1g"],
                output2_scale_scalar=d["o2"],
            )
            fn, sc, bs = d["hc_attn"]
            self._hc_attn_gemm.bind_layered(i, w=fn)
            self._hc_attn_fuse.bind_layered(i, hc_scale=sc, hc_base=bs)
            # The FFN input collapses with this attention's mixes.
            self._hc_pre_ffn.bind_layered(i, scale=sc, base=bs)
            fn, sc, bs = d["hc_ffn"]
            self._hc_ffn_gemm.bind_layered(i, w=fn)
            self._hc_ffn_fuse.bind_layered(i, hc_scale=sc, hc_base=bs)
            if i > 0:
                # Lagged pre: the attention input collapses with the previous
                # layer's FFN mixes (the previous stage's, at a stage's first layer).
                _, psc, pbs = L[i - 1]["hc_ffn"]
                self._hc_pre_attn.bind_layered(i, scale=psc, base=pbs)
            if i in core.kv_sources:
                self._comp_mm.bind_layered(i, mat_b=d["comp_w_t"])
                self._comp_norm.bind_layered(i, weight=d["comp_norm"])
                self._idx_wk.bind_layered(i, mat_b=d["idx_wk_t"])
                self._idx_k_norm.bind_layered(i, weight=d["idx_k_norm"])
            if i in core.index_sources:
                self._idx_wproj.bind_layered(i, mat_b=d["idx_wproj_t"])
            if i in core.engram_layers:
                li = core.engram_layers.index(i)
                self._eng_gather.bind_layered(
                    i,
                    weight=d["eng_table"],
                    scale=d["eng_scale"],
                    row_start=core.eng_start[li],
                    row_end=core.eng_end[li],
                )
                self._eng_gate.bind_layered(i, q_weight=d["eng_q"], k_weight=d["eng_k"])
        if core.last_pp:
            # The final norm's input collapses with the last layer's FFN mixes.
            _, fsc, fbs = L[core.num_layers - 1]["hc_ffn"]
            self._hc_pre_final.bind_const(scale=fsc, base=fbs)

    def _bind_mx(self, name: str, layer: int, wsf) -> None:
        weight, weight_sf = wsf
        self._mx[name].bind_layered(layer, weight=weight, weight_sf=weight_sf)
        self._mx_shape[name, layer] = tuple(weight.shape)

    def step_args(self, md) -> dict:
        """The metadata surface this target reads, projected once per step.

        Also the step-contract probe (``Target._check_step_contract``): reading every
        field here is what checks that they exist. The context-only fields -- the
        context index-K gather, the encoder replay's compute rows and Engram history,
        the context attention's window counts -- are this target's alone."""
        return dict(
            kv_cache_manager=md.kv_cache_manager,
            num_contexts=md.num_contexts,
            num_ctx_tokens=md.num_ctx_tokens,
            num_seqs=md.num_seqs,
            all_rank_num_tokens=md.all_rank_num_tokens,
            req_idx_per_token=md.req_idx_per_token,
            cu_seq_lens_cuda=md.cu_seq_lens_cuda,
            cached_token_lens_cuda=md.cached_token_lens_cuda,
            kv_lens_cuda_runtime=md.kv_lens_cuda_runtime,
            swa_local_indices_cuda=md.swa_local_indices_cuda,
            sparse_mla_topk_lens=md.sparse_mla_topk_lens,
            compressed_visible_lens_cuda=md.compressed_visible_lens_cuda,
            compressed_kv_lens_cuda=md.compressed_kv_lens_cuda,
            max_compressed_kv_lens=md.max_compressed_kv_lens,
            compress_block_tables=md.compress_block_tables,
            indexer_compress_block_tables=md.indexer_compress_block_tables,
            sliding_block_tables=md.sliding_block_tables,
            cu_new_comp_kv_cuda=md.cu_new_comp_kv_cuda,
            compressed_position_ids_cuda=md.compressed_position_ids_cuda,
            num_total_compressed_tokens=md.num_total_compressed_tokens,
            max_ctx_compressed_tokens=md.max_ctx_compressed_tokens,
            engram_token_block_table=md.engram_token_block_table,
            v41_swa_slot_mapping=md.v41_swa_slot_mapping,
            v41_compress_slot_mapping=md.v41_compress_slot_mapping,
            v41_index_k_slot_data=md.v41_index_k_slot_data,
            v41_index_k_slot_scale=md.v41_index_k_slot_scale,
            v41_ctx_index_k_len=md.v41_ctx_index_k_len,
            v41_ctx_index_k_gather_data=md.v41_ctx_index_k_gather_data,
            v41_ctx_index_k_gather_scale=md.v41_ctx_index_k_gather_scale,
            v41_ctx_index_ks=md.v41_ctx_index_ks,
            v41_ctx_index_ke=md.v41_ctx_index_ke,
            v41_max_ctx_q_len=md.v41_max_ctx_q_len,
            v41_swa_seq_lens_cuda=md.v41_swa_seq_lens_cuda,
            # Encoder replay (always valid; the plain values when a step replays nothing).
            v41_num_replay_tokens_total=md.v41_num_replay_tokens_total,
            v41_num_replay_tokens_cpu=md.v41_num_replay_tokens_cpu,
            v41_num_compute_tokens=md.v41_num_compute_tokens,
            v41_compute_token_index_cuda=md.v41_compute_token_index_cuda,
            v41_compute_start_pos_cuda=md.v41_compute_start_pos_cuda,
            v41_compute_cu_seq_lens_cuda=md.v41_compute_cu_seq_lens_cuda,
            v41_replay_history_len=md.v41_replay_history_len,
            v41_replay_history_token_ids_cuda=md.v41_replay_history_token_ids_cuda,
            v41_replay_history_req_idx_cuda=md.v41_replay_history_req_idx_cuda,
            v41_replay_history_cu_seq_lens_cuda=md.v41_replay_history_cu_seq_lens_cuda,
            v41_replay_history_start_pos_cuda=md.v41_replay_history_start_pos_cuda,
            # Host-side per-request state the decoder replay and the read-buffer
            # form plan from.
            request_ids=md.request_ids,
            seq_lens=md.seq_lens,
            seq_lens_kv=md.seq_lens_kv,
            cached_token_lens_cpu=md.cached_token_lens_cpu,
            tokens_per_block=md.tokens_per_block,
            max_seq_len=md.max_seq_len,
        )

    # ------------------------------------------------------------------
    def _mx_linear(self, name: str, x, layer: int, out_dtype=torch.bfloat16):
        """MXFP8 x MXFP8 linear: 1x32 UE8M0 activation quantization, then the
        GEMM against the checkpoint's 32x32 blocks expanded to 1x32."""
        m = x.shape[0]
        xq, xsf = self._mxq(x)
        be = _gemm_backend(m)
        _rv("mxfp8_mxfp8_gemm", m, self._mx_shape[name, layer], be)
        return self._mx[name](xq, xsf, out_dtype=out_dtype, backend=be, layer=layer)

    def _allreduce_fp32(self, x):
        """TP-member sum of bf16 per-rank partials in fp32 (the reference's
        fp32 all-reduce of bf16-rounded partials); returns fp32. AUTO on the IPC
        workspace when the group is on one node, NCCL across nodes."""
        x32 = torch.empty(list(x.shape), dtype=torch.float32, device=x.device)
        x32.copy_(x)
        _rv("allreduce", _sh(x32), "fp32", self._ar32_strategy)
        return self._allreduce32(x32)[0]

    def _engram(self, i, h, hashes, md, num_tokens, dev, keep=None):
        """Engram at layer i, in place on the stream h [T, 4, 5120]. ``keep``
        [T] bool is False on image-span rows (their gate is 0), or None."""
        core = self.core
        li = core.engram_layers.index(i)
        ids = hashes[:, li, :]  # [T, 24] int64, row stride 48
        if core.eng_host:
            # Every row is in the shared pinned copy: this rank's own tokens,
            # no zero fill and no collective, on every member.
            vals = self._eng_gather(ids, layer=i)
            red = torch.reshape(vals, [num_tokens, core.eng_in])
        elif core.dp:
            rows = core._dp_rows(md, num_tokens)
            ids_c = torch.empty([num_tokens, core.eng_cols], dtype=torch.int64, device=dev)
            ids_c.copy_(ids)
            ids32 = ids_c.view(torch.int32)
            if rows > num_tokens:
                ids32 = nn.functional.pad(ids32, [0, 0, 0, rows - num_tokens])
            _rv("allgather", _sh(ids32), "int32")
            ids_all = self._allgather(ids32).view(torch.int64)
            vals = self._eng_gather(ids_all, layer=i)
            flat = torch.reshape(vals, [vals.shape[0], core.eng_in])
            _rv("reducescatter", _sh(flat), "bf16")
            red = self._reducescatter(flat)
            if rows > num_tokens:
                red = torch.split(red, [num_tokens, rows - num_tokens], 0)[0]
        else:
            vals = self._eng_gather(ids, layer=i)
            flat = torch.reshape(vals, [num_tokens, core.eng_in])
            _rv("allreduce", _sh(flat), "bf16", _AR_NCCL)
            red = self._allreduce16(flat)[0]
        _rv("engram_gather", num_tokens, li)
        kv = self._mx_linear("eng_wkv", red, i)  # [T, 25600]
        key, value = torch.split(kv, [_HC * core.hidden, core.hidden], -1)
        key3 = torch.reshape(key, [num_tokens, _HC, core.hidden])
        _rv("engram_gate" if keep is None else "engram_gate.masked", num_tokens)
        return self._eng_gate(h, key3, value, token_mask=keep, out=h, layer=i)

    # ------------------------------------------------------------------
    def _attention(self, i, x, st, pools, cache, dev):
        """One attention sublayer on the normed input x [T, 5120] -> [T, 5120] bf16."""
        core = self.core
        r = core.ratios[i]
        T, tc, gen = st["T"], st["tc"], st["gen"]
        nc, G, B, next_n = st["nc"], st["G"], st["B"], st["next_n"]
        md = st["md"]
        pos = st["pos"]
        table = core._table(r)
        H = core.h_loc
        hd = core.head_dim

        # Low-rank query + the shared window latent.
        qkv = self._mx_linear("wqkv_a", x, i)  # [T, 1792]
        q_lr, kv_raw = torch.split(qkv, [core.q_lora, hd], -1)
        qr = self._q_norm(q_lr, layer=i)
        kv = self._kv_norm(kv_raw, layer=i)
        qrq, qrsf = self._mxq(qr)
        q = self._mx["wq_b"](qrq, qrsf, out_dtype=torch.bfloat16, backend=_gemm_backend(T), layer=i)
        _rv("mla_rope_inplace", T, H, core.nope, r)
        q3 = self._rope(torch.reshape(q, [T, H, hd]), pos, table, H, core.nope, _ROPE, False, False)
        kv = torch.reshape(
            self._rope(
                torch.reshape(kv, [T, 1, hd]), pos, table, 1, core.nope, _ROPE, False, False
            ),
            [T, hd],
        )
        # Window KV: FP8 E4M3 x UE8M0/32 value set into this layer's SWA slots
        # (native: the codes and UE8M0 bytes of the 528 B rows). A pool write.
        native = pools["native"]
        swa_slots = md.v41_swa_slot_mapping[pools["local"][i]][:T]
        if native:
            codes, scales = pools["swa_q"][i]
            _rv("mxfp8_fake_quantize.native", T, _sh(codes))
            self._swa_write(kv, codes, swa_slots, scales)
        else:
            _rv("mxfp8_fake_quantize", T, _sh(pools["swa3"][i]))
            self._swa_write(kv, pools["swa3"][i], swa_slots)

        # Compressed KV + index K (the four KV sources).
        per_query = native and st["comp_form"] == "topk"
        if i in core.kv_sources:
            self._compress(i, x, st, pools, dev)
            if native and not per_query:
                self._native_compressed(i, st, pools)

        # Compressed positions per query (index sources); readers reuse.
        if i in core.index_sources:
            cache["topk"][r] = self._index(i, x, qrq, qrsf, st, pools, cache, dev)
        topk = cache["topk"].get(r) if r else None

        # Pool offsets -> the attention's index table. Native: the compressed
        # columns land on the step's dequantized rows (per request through
        # ``comp_tbl``; per query renumbered by ``_native_topk``), the window
        # columns are renumbered by ``_native_window``.
        req = md.req_idx_per_token[:T]
        swa_bt = md.sliding_block_tables[pools["local"][i], DeepseekV4AttentionType.SWA.value]
        swa_local = md.swa_local_indices_cuda[:T]
        if r:
            _rv("dsv4_local_to_global_indices", T, 128, _TOPK, _PAGE // r)
            gidx = self._l2g(
                req,
                swa_bt,
                swa_local,
                _PAGE,
                0,
                pools["comp_tbl"][r] if native and not per_query else md.compress_block_tables[r],
                topk,
                _PAGE // r,
                0,
            )
            if per_query:
                gidx = self._native_topk(i, gidx, topk, T, pools, cache)
            if native:
                comp_pool = pools["comp_c4_q"][r] if per_query else pools["comp_c4_req"][r]
            else:
                comp_pool = pools["comp4"][core.kv_src[i]]
        else:
            _rv("dsv4_local_to_global_indices", T, 128)
            gidx = self._l2g(req, swa_bt, swa_local, _PAGE, 0)
            comp_pool = core._const["dummy_comp8" if core.fp8_attn else "dummy_comp"]
        if native:
            gidx, swa_pool = self._native_window(i, gidx, T, pools)
        else:
            swa_pool = pools["swa4"][i]
        lens = md.sparse_mla_topk_lens[r][:T]
        kv_lens = md.kv_lens_cuda_runtime
        o = torch.empty([T, H, hd], dtype=torch.bfloat16, device=dev)
        if core.fp8_attn:
            # E4M3 query, one power-of-two scale s_q from this step's max|q| (on the device);
            # the call's QK scale is scale * s_q * s_kv (written by the quantizer) and its
            # PV scale s_kv.
            qa = torch.empty([T, H, hd], dtype=torch.float8_e4m3fn, device=dev)
            bmm1 = torch.empty([1], dtype=torch.float32, device=dev)
            _rv("e4m3_quantize_pow2_dynamic", T, H)
            qa, bmm1 = self._q_e4m3(q3, qa, bmm1, layer=i)
            attn = self._attn8
            fp8 = dict(bmm1_scale=bmm1)
        else:
            qa = q3
            attn = self._attn
            fp8 = {}
        kind = "trtllm_sparse_mla_dsv4_fp8" if core.fp8_attn else "trtllm_sparse_mla_dsv4"
        # Both calls write their rows of `o`, which the inverse RoPE reads.
        if tc:
            _rv(kind + ".varlen", tc, H, gidx.shape[1], nc, st["max_ctx_q"])
            attn(
                torch.split(qa, [tc, gen], 0)[0],
                swa_pool,
                comp_pool,
                sparse_indices=torch.split(gidx, [tc, gen], 0)[0],
                sparse_topk_lens=torch.split(lens, [tc, gen], 0)[0],
                # Window counts from kv length minus the SWA floor: a replay
                # row counts only the window keys at or above its replay start.
                seq_lens=md.v41_swa_seq_lens_cuda[:nc],
                cum_seq_lens_q=md.cu_seq_lens_cuda[: nc + 1],
                max_q_len=st["max_ctx_q"],
                out=torch.split(o, [tc, gen], 0)[0],
                layer=i,
                **fp8,
            )
        if gen:
            _rv(kind + ".decode", G, next_n, H, gidx.shape[1])
            attn(
                torch.reshape(torch.split(qa, [tc, gen], 0)[1], [G, next_n, H, hd]),
                swa_pool,
                comp_pool,
                sparse_indices=torch.split(gidx, [tc, gen], 0)[1],
                sparse_topk_lens=torch.split(lens, [tc, gen], 0)[1],
                seq_lens=kv_lens[nc:B],
                out=torch.reshape(torch.split(o, [tc, gen], 0)[1], [G, next_n, H, hd]),
                layer=i,
                **fp8,
            )
        # Remove the query rotation from the output's rope tail.
        o = self._rope(o, pos, table, H, core.nope, _ROPE, True, False)
        # Grouped low-rank output: per group [T, 4096] x [4096, 1024], into `oa`.
        oa = torch.empty([T, core.g_loc * core.o_lora], dtype=torch.bfloat16, device=dev)
        _rv("bmm_out", core.g_loc, T, core.group_in, core.o_lora)
        self._wo_a(
            torch.transpose(torch.reshape(o, [T, core.g_loc, core.group_in]), 0, 1),
            out=torch.transpose(torch.reshape(oa, [T, core.g_loc, core.o_lora]), 0, 1),
            layer=i,
        )
        y = self._mx_linear("wo_b", oa, i)  # [T, 5120] bf16 (per-rank partial under TP)
        if core.dp:
            return y
        y32 = self._allreduce_fp32(y)
        yb = torch.empty([T, core.hidden], dtype=torch.bfloat16, device=dev)
        yb.copy_(y32)
        return yb

    def _compress(self, i, x, st, pools, dev):
        """KV source i: this step's compressed entries -> index K (native
        MXFP4 pool) and main KV (E2M1 x E4M3/16 value set, bf16 pool).

        Encoder replay: the replay rows at the head of a prefix hit's first
        chunk rebuild the windows only. The compressor runs over the other rows
        (``v41_compute_token_index_cuda``), packed as a step whose requests
        start at their reuse boundaries (``v41_compute_start_pos_cuda`` /
        ``v41_compute_cu_seq_lens_cuda``); the compressed stream, its positions
        and its write slots already start there, so no cached entry is
        recomputed or overwritten."""
        core = self.core
        r = core.ratios[i]
        md = st["md"]
        T, nc, G, B, next_n = st["T"], st["nc"], st["G"], st["B"], st["next_n"]
        hd = core.head_dim
        rows, start, cu = T, md.cached_token_lens_cuda, md.cu_seq_lens_cuda
        if st["replay"]:
            rows = md.v41_num_compute_tokens
            assert rows == T - st["replay"], (rows, T, st["replay"])
            start, cu = md.v41_compute_start_pos_cuda, md.v41_compute_cu_seq_lens_cuda
            st["compress_rows"] = rows
            if rows == 0:
                return  # every row of the step replays: nothing to compress
            _rv("embedding.compute_rows", rows, T)
            x = nn.functional.embedding(md.v41_compute_token_index_cuda[:rows], x)
        if r == 2:
            n = md.num_total_compressed_tokens[2]
            x32 = torch.empty([rows, core.hidden], dtype=torch.float32, device=dev)
            x32.copy_(x)
            kv_score = self._comp_mm(x32, layer=i)  # fp32 [T, 1024] = [kv || score]
            pooled = torch.empty([n, hd], dtype=torch.bfloat16, device=dev)
            # Reserved decode rows the op leaves unwritten stay finite.
            pooled.copy_(core._const["zero_bf16"].expand([n, hd]))
            local = pools["local"][i]
            bt_kv = md.sliding_block_tables[local, DeepseekV4AttentionType.COMPRESSOR_KV.value]
            bt_sc = md.sliding_block_tables[local, DeepseekV4AttentionType.COMPRESSOR_SCORE.value]
            cu_comp = md.cu_new_comp_kv_cuda[2]
            # Both fill `pooled` and update the compressor state pools.
            if nc:
                _rv("compressor_prefill_reduction", T, nc, md.max_ctx_compressed_tokens[2])
                self._comp_prefill(
                    kv_score,
                    paged_kv=pools["ckv"][i],
                    paged_score=pools["csc"][i],
                    block_table_kv=bt_kv[:nc],
                    block_table_score=bt_sc[:nc],
                    output=pooled,
                    kv_lens=md.kv_lens_cuda_runtime[:nc],
                    start_pos=start[:nc],
                    cu_seq_lens=cu[: nc + 1],
                    cu_kv_comp=cu_comp[: nc + 1],
                    batch_size=nc,
                    max_outputs=md.max_ctx_compressed_tokens[2],
                )
            if G:
                _rv("compressor_paged_kv_compress", G, next_n, n)
                self._comp_paged(
                    kv_score,
                    paged_kv=pools["ckv"][i],
                    paged_score=pools["csc"][i],
                    block_table_kv=bt_kv[nc:B],
                    block_table_score=bt_sc[nc:B],
                    output=pooled,
                    kv_lens=md.kv_lens_cuda_runtime[nc:B],
                    cu_seq_lens=cu[nc : B + 1],
                    cu_kv_comp=cu_comp[nc : B + 1],
                    batch_size=G,
                    next_n=next_n,
                )
            # The ops above ran even with no completed group: they store this
            # step's pending odd token in the compressor state.
            if n == 0:
                return
            latent = self._comp_norm(pooled, layer=i)
        else:
            n = rows
            assert md.num_total_compressed_tokens[1] == rows, (
                md.num_total_compressed_tokens[1],
                rows,
                T,
            )
            latent = self._comp_norm(self._comp_mm(x, layer=i), layer=i)
        cpos = md.compressed_position_ids_cuda[r][:n]
        tbl = core._rope_tables[1]
        # Index K from the RoPE-free latent (before the rotation below).
        k = self._idx_k_norm(self._idx_wk(latent, layer=i), layer=i)
        k = torch.reshape(
            self._rope(
                torch.reshape(k, [n, 1, _IDX_DIM]),
                cpos,
                tbl,
                1,
                _IDX_DIM - _ROPE,
                _ROPE,
                False,
                False,
            ),
            [n, _IDX_DIM],
        )
        kd, ksf = self._mxfp4(k)
        _rv("indexer_k_cache_scatter", n, r, _sh(pools["idxk"][i]))
        self._idx_scatter(
            kd,
            torch.reshape(ksf.view(torch.int32), [n, 1]),
            pools["idxk"][i],
            md.v41_index_k_slot_data[r][:n],
            md.v41_index_k_slot_scale[r][:n],
            n,
        )
        # Main KV: rope at the group-first position j*r, then the E2M1 x E4M3/16
        # value set into the owner's COMPRESS slots (native: the packed codes
        # and E4M3 bytes of the 288 B rows). A pool write.
        latent = torch.reshape(
            self._rope(
                torch.reshape(latent, [n, 1, hd]), cpos, tbl, 1, core.nope, _ROPE, False, False
            ),
            [n, hd],
        )
        slots = md.v41_compress_slot_mapping[r][:n]
        if pools["native"]:
            codes, scales = pools["comp_q"][i]
            _rv("nvfp4_fake_quantize.native", n, r, _sh(codes))
            self._comp_write(latent, codes, slots, scales)
        else:
            _rv("nvfp4_fake_quantize", n, r, _sh(pools["comp3"][i]))
            self._comp_write(latent, pools["comp3"][i], slots)

    def _native_compressed(self, i, st, pools):
        """Native storage, KV source i (after its write): every compressed
        entry of every request of the step, dequantized into the compact
        buffer at row ``b * L_r + e`` -- all the rows the layers reading this
        source can name (``compressed_kv_lens`` = kv_len // r per request).
        The buffer is read through ``comp_c4_req``."""
        core = self.core
        r = core.ratios[i]
        md = st["md"]
        # Rows are the engine's request indices (block-table rows); in the
        # decoder replay pass st["B"] counts its compacted request list.
        B = st.get("B_engine", st["B"])
        codes, scales = pools["comp_q4"][i]
        out = pools["comp_out"][r][:B]
        if core.fp8_attn:
            _rv("nvfp4_kv_to_e4m3", B, out.shape[1], r, _sh(codes))
            self._comp_read8(
                codes,
                scales,
                md.compress_block_tables[r][:B],
                md.compressed_kv_lens_cuda[r][:B],
                out,
                1.0 / core._fp8_kv_scale[i],
            )
            return
        _rv("nvfp4_kv_to_bf16", B, out.shape[1], r, _sh(codes))
        self._comp_read(
            codes,
            scales,
            md.compress_block_tables[r][:B],
            md.compressed_kv_lens_cuda[r][:B],
            core._const["one_f32"],
            out,
        )

    def _native_topk(self, i, gidx, topk, T, pools, cache):
        """Native storage, per-query read buffer, layer i: the index table
        with its compressed half renumbered onto ``topk_out`` (query t's
        column k at row ``t * 512 + k``). The rows are dequantized from the
        pool slots the index op named (-1: left unwritten) when the top-k set
        or its KV source changes; the layers that share both reuse them.

        Renumbering every column is exact: the index op packs a row's valid
        entries first and ``sparse_mla_topk_lens`` counts exactly those, so
        every -1 column is masked, and a masked column carries no weight
        whatever finite row it names (every row of the buffer is finite: zero
        at allocation, dequantized values after)."""
        core = self.core
        win, comp = torch.split(gidx, [_WINDOW, _TOPK], 1)
        src = core.kv_src[i]
        last = cache.get("comp_rows")
        if last is None or last[0] is not topk or last[1] != src:
            slots = pools["topk_slots"][:T]
            slots.copy_(comp)
            codes, scales = pools["comp_q1"][src]
            if core.fp8_attn:
                # Every layer reading this source shares its KV scale (``_derive_fp8_kv_scales``).
                _rv("nvfp4_kv_to_e4m3.topk", T, _TOPK, core.ratios[i], _sh(codes))
                self._comp_read8(
                    codes,
                    scales,
                    slots,
                    pools["topk_cols"][:T],
                    pools["topk_out"][:T],
                    1.0 / core._fp8_kv_scale[i],
                )
            else:
                _rv("nvfp4_kv_to_bf16.topk", T, _TOPK, core.ratios[i], _sh(codes))
                self._comp_read(
                    codes,
                    scales,
                    slots,
                    pools["topk_cols"][:T],
                    core._const["one_f32"],
                    pools["topk_out"][:T],
                )
            # Holding the top-k tensor keeps its identity unique for the step.
            cache["comp_rows"] = (topk, src)
        return torch.cat([win, pools["topk_ren"][:T]], 1)

    def _native_window(self, i, gidx, T, pools):
        """Native storage, layer i: dequantize the window rows each query's
        index row names into the compact buffer -- query t's column c to row
        ``t * 128 + c``, a -1 column to a zero row -- and return the index
        table with its window half renumbered onto that buffer (the compressed
        half is unchanged), and the buffer as the attention's window pool.

        A zero row stands in for -1 exactly: the kernel reads an attended -1
        as an all-zero row, and a masked column (past the query's window count)
        carries no weight whatever finite row it names."""
        core = self.core
        codes, scales = pools["swa_q"][i]
        if gidx.shape[1] == _WINDOW:
            win, rest = gidx, None
        else:
            win, rest = torch.split(gidx, [_WINDOW, gidx.shape[1] - _WINDOW], 1)
        if core.fp8_attn:
            _rv("mxfp8_to_e4m3", T * _WINDOW, _sh(pools["swa_c"]), _sh(codes))
            self._swa_read8(
                codes,
                scales,
                pools["swa_c"],
                1.0 / core._fp8_kv_scale[i],
                torch.reshape(win, [T * _WINDOW]),
            )
        else:
            _rv("mxfp8_dequantize", T * _WINDOW, _sh(pools["swa_c"]), _sh(codes))
            self._swa_read(codes, scales, pools["swa_c"], torch.reshape(win, [T * _WINDOW]))
        ren = pools["win_idx"][:T]
        return (ren if rest is None else torch.cat([ren, rest], 1)), pools["swa_c4"]

    def _gather_index_k(self, pool, slots_data, slots_scale, s_len, r):
        """The step's context index-K rows ``[0, s_len)`` out of the paged pool,
        as the logits op reads them: (int8 codes [s_len, 64], int32 UE8M0
        words [s_len]). Gathered in calls of at most ``_GATHER_MAX_ROWS`` rows:
        the op offsets its output by row index x 64 in 32 bits on builds before
        the int64 fix, which wraps from row 2^25 on (a multi-request step at 1M
        context reaches it). One call when the rows fit."""
        k8s, k32s = [], []
        for a in range(0, s_len, _GATHER_MAX_ROWS):
            n = min(_GATHER_MAX_ROWS, s_len - a)
            _rv("indexer_k_cache_gather", n, r)
            k8, k32 = self._idx_gather(pool, slots_data, slots_scale, a, n)
            k8s.append(k8.view(torch.int8))
            k32s.append(torch.reshape(k32.view(torch.int32), [-1]))
        if len(k8s) == 1:
            return k8s[0], k32s[0]
        return torch.cat(k8s, 0), torch.cat(k32s, 0)

    def _cand_masks(self, cand_q, tiles, ks, ke, kvc, s_len, tag) -> list:
        """The candidate-block masks of the context rows of each row tile,
        scored as the candidate source scores them -- its index query rows
        (``cand_q``: MXFP4 q, UE8M0 words, head weights) against this step's
        gathered index K, each row over its own ``[ks, ke)``. A row's logits
        and mask do not depend on the other rows of the call or on where its
        request's keys sit in the gather, so every row gets the mask the source
        layer gives it: a decoder-replay tail row whichever chunk forwarded it
        (``tag`` "tail"), and a row whose mask the step did not keep
        (``tag`` "rescore", see ``_index_prefill``)."""
        q, qs, w = cand_q
        masks = []
        for a, b in tiles:
            _rv("deepgemm_fp8_fp4_mqa_logits." + tag + "_cand", b - a, s_len, 1)
            logits = self._mqa_logits(q[a:b], qs[a:b], kvc[0], kvc[1], w[a:b], ks[a:b], ke[a:b])
            _rv("candidate_block_select." + tag, b - a, s_len)
            masks.append(self._cand(logits, ks[a:b], ke[a:b]))
        return masks

    def _index_prefill(self, i, q, qs, w, kvc, ks, ke, vis, s_len, st, cache, out) -> None:
        """Index source i, the step's context rows: their top-512 into ``out``
        ([tc, 512] int32), scored against the ``s_len`` gathered index keys
        ``kvc``, row t over its own ``[ks[t], ke[t])``.

        The rows run in tiles (``_index_row_tiles``), so a long context never
        holds more than one tile's logits. Candidate blocks bind only above
        2048 x 8 visible positions (below that the mask is all-visible, exact
        to skip). While a step's masks (tc x s_len bytes) fit
        ``_CAND_MASK_BYTES``, they are kept, one per tile, for the re-index
        layers, which tile their rows the same way: the candidate source keeps
        its own; in a bounded step (whose engine rows stop at layer L) the
        decoder pass scores its tail rows' masks at its first re-index layer.
        Past the budget the source keeps its query rows instead and every
        re-index layer re-scores each tile's masks from them (``_cand_masks``),
        so the step holds one tile's masks at a time."""
        core = self.core
        tc = q.shape[0]
        reindex = core.cand_src < i
        tiles = _index_row_tiles(tc, s_len)
        binds = s_len > _CAND_TOPK_BLOCKS * _CAND_BLOCK
        keep = tc * s_len <= _CAND_MASK_BYTES
        src_masks = None
        if i == core.cand_src and binds and not st["tail_bound"]:
            if keep:
                src_masks = cache["cand_ctx"] = []
            else:
                cache["cand_q"] = (q, qs, w)
        masks = None
        if reindex and binds:
            if keep and "cand_ctx" not in cache:
                cache["cand_ctx"] = self._cand_masks(
                    cache["cand_q"], tiles, ks, ke, kvc, s_len, "tail"
                )
            masks = cache.get("cand_ctx")
        rescore = reindex and binds and masks is None
        r = core.ratios[i]
        for n, (a, b) in enumerate(tiles):
            _rv("deepgemm_fp8_fp4_mqa_logits", b - a, s_len, r)
            logits = self._mqa_logits(q[a:b], qs[a:b], kvc[0], kvc[1], w[a:b], ks[a:b], ke[a:b])
            if src_masks is not None:
                _rv("candidate_block_select.prefill", b - a, s_len)
                src_masks.append(self._cand(logits, ks[a:b], ke[a:b]))
            if masks is not None:
                logits = torch.where(masks[n], logits, float("-inf"))
            elif rescore:
                mask = self._cand_masks(cache["cand_q"], [(a, b)], ks, ke, kvc, s_len, "rescore")[0]
                logits = torch.where(mask, logits, float("-inf"))
            _rv("flashinfer_top_k_ragged_deterministic.prefill", b - a, s_len)
            out[a:b].copy_(
                self._topk(logits, core._const["zero_i32_rows"][: b - a], vis[a:b], ks[a:b])
            )

    def _index(self, i, x, qrq, qrsf, st, pools, cache, dev):
        """Index source i: top-512 compressed positions per query, [T, 512]
        int32, valid entries first in ascending order, -1 after."""
        core = self.core
        r = core.ratios[i]
        md = st["md"]
        T, tc, gen, nc, G, B, next_n = (
            st["T"],
            st["tc"],
            st["gen"],
            st["nc"],
            st["G"],
            st["B"],
            st["next_n"],
        )
        src = core.kv_src[i]
        iq = self._mx["idx_wq_b"](
            qrq, qrsf, out_dtype=torch.bfloat16, backend=_gemm_backend(T), layer=i
        )
        iq3 = self._rope(
            torch.reshape(iq, [T, _IDX_HEADS, _IDX_DIM]),
            st["pos"],
            core._rope_tables[1],
            _IDX_HEADS,
            _IDX_DIM - _ROPE,
            _ROPE,
            False,
            False,
        )
        qd, qsf = self._mxfp4(iq3)  # [T,32,64] u8, [T*128] u8
        q8 = qd.view(torch.int8)
        qs32 = torch.reshape(qsf.view(torch.int32), [T, _IDX_HEADS])
        wb = self._idx_wproj(x, layer=i)  # bf16 [T, 32], 2^-6 folded in
        w32 = torch.empty([T, _IDX_HEADS], dtype=torch.float32, device=dev)
        w32.copy_(wb)
        idx = torch.empty([T, _TOPK], dtype=torch.int32, device=dev)
        idx_ctx, idx_gen = torch.split(idx, [tc, gen], 0)
        reindex = core.cand_src < i
        if i == core.cand_src and st["tail_bound"]:
            # Every row of the step: the decoder pass takes its tail rows (and
            # stashes those of a request's earlier chunks).
            cache["cand_q"] = (q8, qs32, w32)
        if tc:
            s_len = md.v41_ctx_index_k_len[r]
            if s_len == 0:
                idx_ctx.copy_(core._const["neg1_i32"].expand([tc, _TOPK]))
            else:
                ks, ke = md.v41_ctx_index_ks[r], md.v41_ctx_index_ke[r]
                kvc = cache["ctx_k"].get(src)
                if kvc is None:
                    kvc = self._gather_index_k(
                        pools["idxk"][src],
                        md.v41_ctx_index_k_gather_data[r],
                        md.v41_ctx_index_k_gather_scale[r],
                        s_len,
                        r,
                    )
                    cache["ctx_k"][src] = kvc
                self._index_prefill(
                    i,
                    torch.split(q8, [tc, gen], 0)[0],
                    torch.split(qs32, [tc, gen], 0)[0],
                    torch.split(w32, [tc, gen], 0)[0],
                    kvc,
                    ks,
                    ke,
                    md.compressed_visible_lens_cuda[r][:tc],
                    s_len,
                    st,
                    cache,
                    idx_ctx,
                )
        if gen:
            vis = md.compressed_visible_lens_cuda[r][tc:T]
            ctx_lens = torch.reshape(vis, [G, next_n])
            sched = cache["sched"].get(r)
            if sched is None:
                sched = self._paged_meta(ctx_lens, _PAGE // r)
                cache["sched"][r] = sched
            max_ctx = md.max_compressed_kv_lens[r]
            _rv("deepgemm_fp8_fp4_paged_mqa_logits", G, next_n, r, max_ctx)
            logits = self._paged_logits(
                torch.reshape(
                    torch.split(q8, [tc, gen], 0)[1], [G, next_n, _IDX_HEADS, _IDX_DIM // 2]
                ),
                torch.reshape(torch.split(qs32, [tc, gen], 0)[1], [G, next_n, _IDX_HEADS]),
                pools["idxk"][src],
                torch.split(w32, [tc, gen], 0)[1],
                ctx_lens,
                md.indexer_compress_block_tables[r][nc:B],
                sched,
                max_ctx,
            )
            if core._cand_binds_decode:
                if i == core.cand_src:
                    _rv("candidate_block_select.decode", gen, max_ctx)
                    cache["cand_gen"] = self._cand(logits, None, vis)
                elif reindex:
                    logits = torch.where(cache["cand_gen"], logits, float("-inf"))
            _rv("flashinfer_top_k_ragged_deterministic.decode", gen, max_ctx)
            idx_gen.copy_(self._topk(logits, core._const["zero_i32_rows"][:gen], vis, None))
        return idx

    def _route(self, i, chunk, img, dev):
        """Router of one token chunk [n, 5120] bf16: the gate GEMM (bf16 x and
        the checkpoint's bf16 weight on tensor cores, fp32 accumulation, fp32
        logits; every product is exact in fp32, the summation order is the
        tensor cores'), then top-6 by ``sqrt(softplus(logit)) + bias``. With
        ``img`` ([n] bool, True on image-span rows) those rows select with
        ``gate.bias_vl`` instead: a second routing call over the same logits
        and a per-row select of both results. Returns (weights [n, 6] fp32,
        ids [n, 6] int32)."""
        core = self.core
        n = chunk.shape[0]
        logits = self._router(chunk, layer=i)  # fp32 [n, 384]
        _rv("gate_forward", n, core.num_experts, core.topk)
        tw, tid = self._gate(logits, layer=i)
        if img is None:
            return tw, tid
        _rv("gate_forward.bias_vl", n, core.num_experts, core.topk)
        tw_vl, tid_vl = self._gate_vl(logits, layer=i)
        sel = torch.reshape(img, [n, 1])
        return torch.where(sel, tw_vl, tw), torch.where(sel, tid_vl, tid)

    def _moe(self, i, x, rows, dev, img=None):
        """Routed NVFP4 experts + FP8 shared expert on the normed input x.
        ``img`` is the image-span mask of the rows the router sees (this rank's
        rows, or the gathered rows under attention DP), or None. ``rows`` is
        the dep4 per-rank row count every rank pads to (None under attention
        TP). ``x`` None is a dep4 rank with no row in this pass: it sends zero
        rows and returns None."""
        core = self.core
        T = 0 if x is None else x.shape[0]
        assert T or core.dp, "only a dep4 rank runs the MoE without rows"
        if T:
            # Shared expert (replicated, this rank's own tokens): [gate ; up].
            h13 = self._mx_linear("sh_w13", x, i)
            _rv("silu_and_mul", T, 2 * core.moe_inter)
            act = self._silu(h13)
            shared = self._mx_linear("sh_w2", act, i)
        # Routed: dep4 gathers the token set ahead of the router, so every rank
        # routes the identical set and the four expert windows tile it once.
        pad_rows = 0
        xr = x
        if core.dp:
            assert rows is not None and T <= rows, (T, rows)
            pad_rows = rows - T
            if not T:
                xp = torch.empty([rows, core.hidden], dtype=torch.bfloat16, device=dev)
                xp.copy_(core._const["zero_bf16"].expand([rows, core.hidden]))
            else:
                xp = x if not pad_rows else nn.functional.pad(x, [0, 0, 0, pad_rows])
            _rv("allgather", _sh(xp), "bf16")
            xr = self._allgather(xp)
        parts = []
        sizes = _moe_chunks(xr.shape[0])
        if img is not None:
            assert img.shape[0] == xr.shape[0], (tuple(img.shape), tuple(xr.shape))
        img_chunks = torch.split(img, sizes, 0) if img is not None else [None] * len(sizes)
        for chunk, img_c in zip(torch.split(xr, sizes, 0), img_chunks):
            n = chunk.shape[0]
            tw, tid = self._route(i, chunk, img_c, dev)
            twb = torch.empty([n, core.topk], dtype=torch.bfloat16, device=dev)
            twb.copy_(tw)
            hs, hsf = self._moe_quant(chunk, layer=i)
            out = torch.empty([n, core.hidden], dtype=torch.bfloat16, device=dev)
            _rv("fp4_block_scale_moe_runner", n, core.e_loc, core.e_off, core.i_loc)
            # Given `output`, the runner finalizes into it and returns an empty tensor.
            self._moe_runner(
                hidden_states=hs,
                hidden_states_scale=hsf.view(torch.float8_e4m3fn),
                topk_weights=twb,
                topk_ids=tid,
                output=out,
                layer=i,
            )
            parts.append(out)
        routed = parts[0] if len(parts) == 1 else torch.cat(parts, 0)
        if core.dp:
            r32 = torch.empty(list(routed.shape), dtype=torch.float32, device=dev)
            r32.copy_(routed)
            _rv("reducescatter", _sh(r32), "fp32")
            r32 = self._reducescatter(r32)
            if not T:
                return None
            if pad_rows:
                r32 = torch.split(r32, [T, pad_rows], 0)[0]
        else:
            r32 = self._allreduce_fp32(routed)
        m32 = torch.add(r32, shared)  # fp32: the shared expert joins after the reduce
        m = torch.empty([T, core.hidden], dtype=torch.bfloat16, device=dev)
        m.copy_(m32)
        return m

    # ------------------------------------------------------------------
    def _mm_step(self, mm_params, T, tc, dev, emb):
        """Splice this step's image-span rows into the token embedding ``emb``
        [T, 5120] in place; return the step's image masks, or None when no
        context chunk of the step carries images. A later pipeline stage passes
        ``emb`` None and gets the masks only (the first stage spliced the rows
        into the stream it receives).

        A request with images arrives as one ``MultimodalParams``: its
        ``multimodal_data`` is the input processor's (``image``: the tower's
        inputs for all its images; ``token_layout``: masks per prompt
        position), ``input_ids_start_offset`` is its first row in this step and
        ``multimodal_runtime`` the chunk ``[begin, end)`` of the prompt those
        rows hold (the whole prompt when absent). Only the images whose spans
        meet the chunk run the tower (``VisionTower.image_rows``), one image at
        a time, and each image's rows are copied into ``emb`` before the next
        image runs: a step's vision memory is one image's working set, however
        many images the prompt or the step holds. Returns ``img`` [T] bool (True on
        image-span rows: the MoE selects with ``bias_vl``) and ``keep`` [T]
        bool (False on image-span rows: Engram DEAD, gate 0)."""
        if not mm_params:
            return None
        core = self.core
        reqs = []
        for p in mm_params:
            data = p.multimodal_data or {}
            image, layout = data.get("image"), data.get("token_layout")
            if image is None and layout is None:
                continue
            assert image is not None and layout is not None, sorted(data)
            L = layout["image_mask"].shape[0]
            rt = p.multimodal_runtime
            begin, end = (
                (0, L) if rt is None else (int(rt.past_seen_token_num), int(rt.chunk_end_pos))
            )
            row0 = int(p.input_ids_start_offset)
            assert 0 <= begin <= end <= L and row0 + (end - begin) <= tc, (begin, end, L, row0, tc)
            reqs.append((image, layout, begin, end, row0))
        if not reqs:
            return None
        img = torch.empty([T], dtype=torch.bool, device=dev)
        img.copy_(core._const["false_b"].expand([T]))
        dead = torch.empty([T], dtype=torch.bool, device=dev)
        dead.copy_(core._const["false_b"].expand([T]))
        spliced = 0
        for image, layout, begin, end, row0 in reqs:
            n = end - begin
            img[row0 : row0 + n].copy_(layout["image_mask"][begin:end])
            dead[row0 : row0 + n].copy_(layout["engram_dead_mask"][begin:end])
            # (image, first, end) prompt positions of every span that meets the chunk.
            hits = []
            for j, (s, ln) in enumerate(zip(image["span_starts"], image["span_lengths"])):
                s, ln = int(s), int(ln)
                a, b = max(s, begin), min(s + ln, end)
                if a < b:
                    hits.append((j, a, b, s))
            if emb is None:
                continue
            for j, a, b, s in hits:
                rows = core._vision.image_rows(image, j)
                emb[row0 + a - begin : row0 + b - begin].copy_(rows[a - s : b - s])
                spliced += b - a
        keep = torch.where(dead, core._const["false_b"], core._const["true_b"])
        _rv("mm.step", T, len(reqs), spliced)
        return {"img": img, "keep": keep}

    def _dp_gather_mask(self, loc, rows, T, dev):
        """dep4: the image-span mask of the all-gathered token set the router
        sees, each rank padded to ``rows`` as the MoE input is. ``loc`` is this
        rank's [T] mask, or None for T text rows (T may be 0: a pass that holds
        none of this rank's rows)."""
        assert T <= rows, (T, rows)
        if loc is None or not T:
            loc = torch.empty([rows if not T else T], dtype=torch.bool, device=dev)
            loc.copy_(self.core._const["false_b"].expand([loc.shape[0]]))
        u8 = loc.view(torch.uint8)
        if rows > loc.shape[0]:
            u8 = nn.functional.pad(u8, [0, rows - loc.shape[0]])
        _rv("allgather", _sh(u8), "uint8")
        return self._allgather(u8).view(torch.bool)

    def _dp_late_rows(self, mine: list, dev) -> list:
        """dep4 decoder replay: the passes every rank runs over the bounded
        layers, as each pass's row count padded to its max over ranks.

        ``mine`` is this rank's pass row counts (a bounded rank's slices, or its
        step rows as one pass). The bounded layers' MoE is a collective, so the
        ranks must run the same passes in lockstep; a rank's decoder rows can
        also outnumber its step rows (a small last chunk still runs its whole
        tail, the rest from the stash). Every rank calls this in every eager
        step under decoder replay, prefill and decode alike: ``[count, rows_0,
        rows_1, ...]`` rides in the int32 [1, 48] form the Engram ids gather
        in, and the result has one entry per pass of the rank with the most
        (0-row entries never occur: some rank holds the rows of every pass)."""
        core = self.core
        width = 2 * core.eng_cols  # 48
        vals = [len(mine), *mine]
        assert len(vals) <= width, (len(mine), width)
        core._ensure_ar(max(vals) + 1, dev)
        cols = [core._ar32[v : v + 1] for v in vals] + [core._ar32[0:1]] * (width - len(vals))
        v = torch.reshape(torch.cat(cols, 0), [1, width])
        _rv("allgather", _sh(v), "int32")
        got = self._allgather(v).tolist()  # [world, 48]
        assert len(got) == core.tp and got[core.tp_rank][: len(vals)] == vals, (
            core.tp_rank,
            vals,
            got,
        )
        return _dp_pass_rows(got)

    def _dp_idle_pass(self, rows: int, dev) -> None:
        """dep4 decoder replay: this rank's place in a pass over the bounded
        layers that holds none of its rows -- the mask gather and every
        bounded layer's routed round trip, with zero rows."""
        img = self._dp_gather_mask(None, rows, 0, dev)
        for i in range(self.core.late, self.core.num_layers):
            self._moe(i, None, rows, dev, img)

    # ------------------------------------------------------------------
    def _scratch(self, s, T: int, dev) -> None:
        """The per-row scratch of one pass over the T rows of stream ``s.h``."""
        core = self.core
        f32 = torch.float32
        s.T = T
        s.h_flat = torch.reshape(s.h, [T, core.hc_k])
        s.tile = _tile_n(T)
        s.yA = torch.empty([T, _MIX], dtype=f32, device=dev)
        s.rA = torch.empty([T], dtype=f32, device=dev)
        s.mix4 = torch.empty([T, _HC], dtype=f32, device=dev)
        s.post = torch.empty([T, _HC], dtype=f32, device=dev)
        s.comb = torch.empty([T, _HC, _HC], dtype=f32, device=dev)
        s.no_res = torch.empty([T, _HC, 0], dtype=torch.bfloat16, device=dev)
        s.no_li = torch.empty([T, 0], dtype=torch.bfloat16, device=dev)
        s.coll = torch.empty([T, core.hidden], dtype=torch.bfloat16, device=dev)
        _rv("mhc", T, s.tile)

    def _layer(self, i: int, s, dev) -> None:
        """Layer i over the pass ``s`` (its stream, scratch, step dict, cache)."""
        core = self.core
        T, h, h_flat = s.T, s.h, s.h_flat
        # DSpark reads the INPUT stream of its target layers (37-39), every
        # token; the metadata averages the 4 hc copies itself.
        if s.spec is not None and s.spec.is_layer_capture(i):
            s.spec.maybe_capture_hidden_states(i, h_flat)
        if i in core.engram_layers:
            h = self._engram(i, h, s.hashes, s.md, T, dev, s.keep)
        # --- attention sublayer ------------------------------------
        s.yA, s.rA = self._hc_attn_gemm(h_flat, y=s.yA, r=s.rA, M=T, tile_n=s.tile, layer=i)
        s.post, s.comb, s.no_li = self._hc_attn_fuse(
            s.yA,
            s.rA,
            s.no_res,
            post_mix=s.post,
            comb_mix=s.comb,
            layer_input=s.no_li,
            M=T,
            layer=i,
        )
        if i == 0:
            a_in = s.emb  # one-hot pre on copy 0: the embedding itself
        else:
            # Lagged pre: the previous FFN's mixes collapse this input.
            s.mix4.copy_(torch.split(s.yF, [_HC, _MIX - _HC], 1)[0])
            s.coll = self._hc_pre_attn(s.mix4, s.rF, h, s.coll, M=T, layer=i)
            a_in = s.coll
        x = self._attn_norm(a_in, layer=i)
        if s.tail is not None and i == core.pp_layer_ids[0]:
            s.tail.count_forwarded(x)
        a = self._attention(i, x, s.st, s.pools, s.cache, dev)
        h = self._hc_post(h, a, s.post, s.comb, h)
        # --- FFN sublayer --------------------------------------------
        s.yF, s.rF = self._hc_ffn_gemm(h_flat, y=s.yF, r=s.rF, M=T, tile_n=s.tile, layer=i)
        s.post, s.comb, s.no_li = self._hc_ffn_fuse(
            s.yF,
            s.rF,
            s.no_res,
            post_mix=s.post,
            comb_mix=s.comb,
            layer_input=s.no_li,
            M=T,
            layer=i,
        )
        s.mix4.copy_(torch.split(s.yA, [_HC, _MIX - _HC], 1)[0])
        s.coll = self._hc_pre_ffn(s.mix4, s.rA, h, s.coll, M=T, layer=i)
        x = self._ffn_norm(s.coll, layer=i)
        m = self._moe(i, x, s.dp_rows, dev, s.img)
        self._hc_post(h, m, s.post, s.comb, h)

    def _final(self, s):
        """The final collapse with layer 39's FFN mixes, then the final norm."""
        s.mix4.copy_(torch.split(s.yF, [_HC, _MIX - _HC], 1)[0])
        s.coll = self._hc_pre_final(s.mix4, s.rF, s.h, s.coll, M=s.T)
        return self._final_norm(s.coll)

    # ------------------------------------------------------------------
    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: torch.IntTensor | None = None,
        position_ids: torch.IntTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        lora_params: dict | None = None,
        **kwargs,
    ) -> torch.Tensor:
        core = self.core
        self._check_step_contract(attn_metadata)
        md = attn_metadata
        step = self.step_args(md)
        spec_metadata = kwargs.get("spec_metadata")
        pools = core._get_pools(md)
        dev = input_ids.device
        T = input_ids.shape[0]
        nc = step["num_contexts"]
        tc = step["num_ctx_tokens"]
        B = step["num_seqs"]
        G = B - nc
        gen = T - tc
        next_n = gen // G if G else 1
        assert next_n * G == gen, (gen, G)
        # A DSpark generation row verifies its 5 drafts: 1 + 5 tokens.
        assert next_n == 1 or (spec_metadata is not None and next_n == 1 + core.ds.block), (
            f"generation rows of {next_n} tokens (spec_metadata {'set' if spec_metadata is not None else 'None'})"
        )
        st = {
            "md": md,
            "T": T,
            "tc": tc,
            "gen": gen,
            "nc": nc,
            "G": G,
            "B": B,
            "next_n": next_n,
            "pos": torch.reshape(position_ids, [-1]),
            "max_ctx_q": int(step["v41_max_ctx_q_len"]) if nc else 0,
            # Encoder replay rows of this step (host int; 0 without a prefix hit).
            "replay": int(step["v41_num_replay_tokens_total"]) if nc else 0,
        }
        cache = {"topk": {}, "ctx_k": {}, "sched": {}}
        if st["replay"]:
            _rv("encoder_replay.rows", T, st["replay"], nc, gen)

        # The token embedding (first pipeline stage); a context chunk that holds
        # image-span rows gets them spliced in over the placeholder rows (mm
        # None: a text step). A later stage starts from the previous stage's
        # stream and needs the image masks only.
        emb = nn.functional.embedding(input_ids, core.w["embed"]) if core.first_pp else None
        mm = self._mm_step(kwargs.get("multimodal_params"), T, tc, dev, emb)
        # This rank's per-row image-span mask; under dep4 the router reads the
        # gathered one (img_moe).
        img_loc = mm["img"] if mm is not None else None
        img_moe = img_loc
        eager = not torch.cuda.is_current_stream_capturing()
        # dep4: every rank pads its MoE rows to dp_rows (past layer L under
        # decoder replay, to its pass's count instead: _dp_late_rows).
        dp_rows = core._dp_rows(md, T) if core.dp else None
        if core.dp and eager:
            # The router runs on the all-gathered token set, so it needs every
            # rank's mask. Every rank gathers in every eager step (a peer may
            # hold image rows); captured steps are generation-only on every
            # rank (the graph decision is all-rank), where no row is an image.
            img_moe = self._dp_gather_mask(img_loc, dp_rows, T, dev)

        # Encoder replay: the history pool holds nothing before a replay start
        # (the manager rebuilds windowed pages on a hit), so the compressed ids
        # of the three prompt positions before it are stored first. Their hash
        # ids are not used.
        hist = step["v41_replay_history_len"] if nc and core.eng_local else 0
        if hist:
            _rv("engram_hash.replay_history", hist, nc)
            self._eng_hash(
                step["v41_replay_history_token_ids_cuda"][:hist],
                step["v41_replay_history_req_idx_cuda"][:hist],
                step["v41_replay_history_cu_seq_lens_cuda"][: nc + 1],
                step["v41_replay_history_start_pos_cuda"][:nc],
                pools["tok"],
                step["engram_token_block_table"],
            )
        # Engram hash ids of every token (int64 [T, 2, 24]); also writes this
        # step's compressed ids into the paged history pool (-1 = DEAD on
        # image-span rows).
        hashes = None
        if core.eng_local:
            # The history is read and written at the rows' own positions: each
            # request's start is the position of its first row in this step. The
            # metadata's cached_token_lens_cuda is not a substitute in generation
            # steps: under the overlap scheduler with speculation it is the host's
            # count before the previous step's acceptance is applied on the device
            # (on_update_kv_lens corrects the kv lengths and positions, not it), so
            # it runs ahead of the rows by the rejected drafts.
            eng_start = torch.reshape(
                nn.functional.embedding(
                    step["cu_seq_lens_cuda"][:B], torch.reshape(st["pos"], [T, 1])
                ),
                [B],
            )
            _rv("engram_hash" if mm is None else "engram_hash.masked", T, B)
            hashes = self._eng_hash(
                input_ids,
                step["req_idx_per_token"][:T],
                step["cu_seq_lens_cuda"][: B + 1],
                eng_start,
                pools["tok"],
                step["engram_token_block_table"],
                token_mask=mm["keep"] if mm is not None else None,
            )

        h = torch.empty([T, _HC, core.hidden], dtype=torch.bfloat16, device=dev)
        yF = torch.empty([T, _MIX], dtype=torch.float32, device=dev)
        rF = torch.empty([T], dtype=torch.float32, device=dev)
        if core.first_pp:
            h.copy_(torch.reshape(emb, [T, 1, core.hidden]).expand([T, _HC, core.hidden]))
        else:
            # A later pipeline stage starts from the previous stage's stream and
            # its last FFN mixes (the lagged pre of this stage's first attention).
            _rv("pp_recv_tensors", T)
            self._pp_recv([h, yF, rF])

        s = SimpleNamespace(
            h=h,
            yF=yF,
            rF=rF,
            st=st,
            cache=cache,
            img=img_moe,
            spec=spec_metadata,
            emb=emb,
            hashes=hashes,
            keep=mm["keep"] if mm is not None else None,
            md=md,
            pools=pools,
            tail=None,
            dp_rows=dp_rows,
        )
        self._scratch(s, T, dev)
        # Decoder SWA bounded replay: this context step's decoder rows (and the
        # witness counters), planned from host metadata. It lives on the stage
        # that holds the bounded layers (the last one; layer 20, which every
        # bounded layer reads, sits on it too).
        tail = (
            _DecoderTail(core, md, st, dev, self._l2g)
            if nc and (core.replay or core._witness) and core.late in core.local
            else None
        )
        s.tail = tail
        bounded = tail is not None and tail.bound
        # A bounded step's context rows stop at layer L: the candidate source
        # then scores no mask over them, and the decoder pass scores its own.
        st["tail_bound"] = bounded
        # Native KV storage: the compressed read buffer's form for the whole step.
        st["comp_form"] = core._comp_form(pools, md, st, tail) if pools["native"] else None
        max_rows = int(md.swa_local_indices_cuda.shape[0])  # the step capacity
        # dep4 decoder replay: the passes every rank runs over the bounded
        # layers (a bounded rank's slices, or the step rows as one pass).
        dp_late = None
        if core.dp and core.replay and eager:
            dp_late = self._dp_late_rows(tail.pass_rows(max_rows) if bounded else [T], dev)

        stop = core.late if bounded or dp_late is not None else core.num_layers
        for i in core.pp_layer_ids:
            if i >= stop:
                break
            if tail is not None and i == core.late:
                tail.count_full(s.h)
            self._layer(i, s, dev)
        if not core.last_pp:
            # Hand the stream and the last FFN mixes to the next stage; this
            # stage's output rows are never read (its lm_head is skipped).
            _rv("pp_send_tensors", T)
            self._pp_send([s.h, s.yF, s.rF])
            return s.coll
        if bounded:
            # Every per-row tensor a bounded layer reads is mapped to the
            # decoder rows. The image-span mask the router reads follows the
            # decoder rows and the stash too; a text step's is all False (the
            # router's select then returns the text routing exactly), so a
            # request whose chunks differ in images keeps one.
            if img_loc is None:
                img_loc = torch.empty([T], dtype=torch.bool, device=dev)
                img_loc.copy_(core._const["false_b"].expand([T]))
            passes = tail.enter(
                {"h": s.h, "yF": s.yF, "rF": s.rF, "img": img_loc}, st, cache, max_rows
            )
            for k, (rows, st_d, cache_d) in enumerate(passes):
                p = SimpleNamespace(
                    h=rows["h"],
                    yF=rows["yF"],
                    rF=rows["rF"],
                    st=st_d,
                    cache=cache_d,
                    img=rows["img"],
                    spec=_TailCapture(spec_metadata, tail) if spec_metadata is not None else None,
                    emb=None,
                    hashes=None,
                    keep=None,
                    md=md,
                    pools=pools,
                    tail=None,
                    dp_rows=None,
                )
                if dp_late is not None:
                    p.dp_rows = dp_late[k]
                    p.img = self._dp_gather_mask(rows["img"], dp_late[k], st_d["T"], dev)
                self._scratch(p, st_d["T"], dev)
                tail.activate(k)
                for i in range(core.late, core.num_layers):
                    self._layer(i, p, dev)
                tail.scatter(self._final(p))
            for k in range(len(passes), len(dp_late or ())):
                self._dp_idle_pass(dp_late[k], dev)
            out = None
        else:
            if dp_late is not None:
                # Nothing is bounded on this rank: its step rows are its one
                # pass, and it joins its peers' further passes with zero rows.
                if tail is not None:
                    tail.count_full(s.h)
                s.dp_rows = dp_late[0]
                s.img = self._dp_gather_mask(img_loc, dp_late[0], T, dev)
                for i in range(core.late, core.num_layers):
                    self._layer(i, s, dev)
                for k in range(1, len(dp_late)):
                    self._dp_idle_pass(dp_late[k], dev)
            out = self._final(s)
        if tail is not None:
            if st["replay"]:
                # The replay rows the counters report are the rows the step's
                # compressors did not run over.
                skipped = st["T"] - st["compress_rows"]
                assert skipped == sum(r.R for r in tail.reqs), (skipped, [r.R for r in tail.reqs])
            out = tail.close(out, spec_metadata)
        return out


class DecodeTarget(Target):
    """The specialization: routed to only when the step has no context request.

    Every row is a generation row: ``G`` requests of ``next_n`` rows each (1, or
    1 + 5 under DSpark). Binds three fewer ops than the prefill target -- the
    context compressor reduction, the context index-K gather and the context
    index logits serve rows this target's routing guarantees are absent -- and
    holds no vision splice, encoder replay or decoder tail. Under attention DP
    an eager step still joins its peers' collectives (the image-mask gather, the
    decoder-replay pass plan and any idle pass), since a peer rank may be running
    a prefill step at the same time.
    """

    def __init__(self, core: "DeepseekV41UnifiedModelingV2") -> None:
        """Bind every op this target calls, once, against the real weights."""
        super().__init__(core)
        L = core._L
        eps = core.eps
        e = core._eng

        # MXFP8 linears: one activation quantizer, one GEMM per weight role.
        self._mxq = Mxfp8Quantize()
        self._mx = {name: Mxfp8Mxfp8Gemm() for name in _MX_LINEARS}
        self._mx_shape: dict = {}

        # Norms.
        self._attn_norm = FlashinferRmsnorm()
        self._ffn_norm = FlashinferRmsnorm()
        self._q_norm = FlashinferRmsnorm()
        self._kv_norm = FlashinferRmsnorm()
        self._comp_norm = FlashinferRmsnorm()
        self._idx_k_norm = FlashinferRmsnorm()
        for op in (
            self._attn_norm,
            self._ffn_norm,
            self._q_norm,
            self._kv_norm,
            self._comp_norm,
            self._idx_k_norm,
        ):
            op.bind_const(eps=eps)
        self._final_norm = FlashinferRmsnorm()
        self._final_norm.bind_const(weight=core.w["final_norm"], eps=eps)

        # Attention: RoPE, the window and compressed KV writes, their native
        # read-back, the index table, the sparse MLA core, the grouped output.
        self._rope = MlaRopeInplace()
        self._swa_write = Mxfp8FakeQuantize()
        self._comp_write = Nvfp4FakeQuantize()
        self._swa_read = Mxfp8Dequantize()
        self._swa_read8 = Mxfp8ToE4m3()
        self._comp_read = Nvfp4KvToBf16()
        self._comp_read8 = Nvfp4KvToE4m3()
        self._l2g = Dsv4LocalToGlobalIndices()
        self._q_e4m3 = E4m3QuantizePow2Dynamic()
        self._attn = TrtllmSparseMlaDsv4()
        self._attn.bind_const(
            workspace_buffer=core._const["attn_ws"],
            softmax_scale=core.head_dim**-0.5,
            output_scale=1.0,
        )
        self._attn8 = TrtllmSparseMlaDsv4Fp8()
        self._attn8.bind_const(workspace_buffer=core._const["attn_ws"])
        self._wo_a = BmmOut()

        # Compressor and index K (the KV sources).
        self._comp_mm = CublasMm()
        self._comp_paged = CompressorPagedKvCompress()
        self._comp_paged.bind_const(
            ape=core._const["ape"], page_size=_PAGE, head_dim=core.head_dim, compress_ratio=2
        )
        self._idx_wk = CublasMm()
        # MXFP4 (UE8M0 per 32, linear scales): the index K and the index query.
        self._mxfp4 = Fp4Quantize()
        self._mxfp4.bind_const(
            global_scale=None, sf_vec_size=_SF32, sf_use_ue8m0=True, is_sf_swizzled_layout=False
        )
        self._idx_scatter = IndexerKCacheScatter()

        # Indexer (the index sources) over the paged index K, and the candidate-block masks.
        self._idx_wproj = CublasMm()
        self._paged_logits = DeepgemmFp8Fp4PagedMqaLogits()
        self._paged_logits.bind_const(logits_dtype=torch.float32)
        self._paged_meta = DeepgemmPagedMqaLogitsMetadata()
        self._cand = CandidateBlockSelect()
        self._cand.bind_const(topk_blocks=_CAND_TOPK_BLOCKS, block_size=_CAND_BLOCK, next_n=1)
        self._topk = FlashinferTopKRaggedDeterministic()
        self._topk.bind_const(k=_TOPK)

        # MoE: router (fp32 logits), routing with gate.bias and gate.bias_vl,
        # the NVFP4 activation quantizer and runner, the shared expert's SwiGLU.
        self._router = CublasMm()
        self._router.bind_const(bias=None, out_dtype=torch.float32)
        self._gate = GateForward()
        self._gate_vl = GateForward()
        for op in (self._gate, self._gate_vl):
            op.bind_const(topk=core.topk, route_scale=core.route_scale)
        self._moe_quant = Fp4Quantize()
        self._moe_quant.bind_const(
            sf_vec_size=_SF16, sf_use_ue8m0=False, is_sf_swizzled_layout=_SF_LINEAR
        )
        self._moe_runner = Fp4BlockScaleMoeRunner()
        self._moe_runner.bind_const(
            routing_logits=None,
            routing_bias=None,
            gemm1_bias=None,
            gemm1_alpha=None,
            gemm1_beta=None,
            gemm2_bias=None,
            num_experts=core.num_experts,
            top_k=core.topk,
            n_group=None,
            topk_group=None,
            intermediate_size=core.i_loc,
            local_expert_offset=core.e_off,
            local_num_experts=core.e_loc,
            routed_scaling_factor=None,
            routing_method_type=_ROUTING_METHOD_INERT,
            do_finalize=True,
            act_type=_ACT_SWIGLU,
        )
        self._silu = SiluAndMul()
        self._silu.bind_const(swiglu_limit=_SWIGLU_LIMIT)

        # Collectives: the TP (or attention-DP) group of this stage.
        self._allgather = Allgather()
        self._allgather.bind_const(sizes=None, group=core.group)
        self._reducescatter = Reducescatter()
        self._reducescatter.bind_const(sizes=None, group=core.group)
        # fp32 sums on AUTO (the IPC workspace) when the group is on one node,
        # NCCL across nodes; bf16 sums on NCCL.
        self._ar32_strategy = _AR_AUTO if core._ar_ws is not None else _AR_NCCL
        self._allreduce32 = Allreduce()
        self._allreduce16 = Allreduce()
        for op, ws, strategy in (
            (self._allreduce32, core._ar_ws, self._ar32_strategy),
            (self._allreduce16, None, _AR_NCCL),
        ):
            op.bind_const(
                residual=None,
                norm_weight=None,
                scale=None,
                bias=None,
                workspace=ws,
                group=core.group,
                strategy=strategy,
                op=0,
                eps=1e-6,
                trigger_completion_at_end=True,
            )
        self._pp_send = PpSendTensors()
        self._pp_recv = PpRecvTensors()

        # Engram: the hash constants, the table gather, the gate.
        self._eng_hash = EngramHash()
        self._eng_hash.bind_const(
            token_map=e["token_map"],
            multipliers=e["multipliers"],
            primes=e["primes"],
            offsets=e["offsets"],
            pad_id=e["pad_id"],
        )
        self._eng_gather = EngramGather()
        self._eng_gate = EngramGate()
        self._eng_gate.bind_const(eps=eps, clamp_value=1e-6)

        # Hyper-connections: per sublayer the mix GEMM + squared sums, the fused
        # sinkhorn, and the lagged-pre collapse; the post mapping.
        self._hc_attn_gemm = MhcGemmSqrsumFma()
        self._hc_ffn_gemm = MhcGemmSqrsumFma()
        for op in (self._hc_attn_gemm, self._hc_ffn_gemm):
            op.bind_const(N=_MIX, K=core.hc_k)
        self._hc_attn_fuse = MhcBigFuse()
        self._hc_ffn_fuse = MhcBigFuse()
        for op in (self._hc_attn_fuse, self._hc_ffn_fuse):
            op.bind_const(
                K=core.hc_k,
                hidden_size=0,
                rms_eps=eps,
                hc_pre_eps=core.hc_eps,
                hc_sinkhorn_eps=core.hc_eps,
                hc_post_mult_value=2.0,
                sinkhorn_repeat=20,
                num_splits=1,
                block_size=128,
            )
        self._hc_pre_attn = MhcHcHeadApply()
        self._hc_pre_ffn = MhcHcHeadApply()
        self._hc_pre_final = MhcHcHeadApply()
        for op in (self._hc_pre_attn, self._hc_pre_ffn, self._hc_pre_final):
            op.bind_const(
                mult=_HC, hidden_size=core.hidden, K=core.hc_k, norm_eps=eps, eps=core.hc_eps
            )
        self._hc_post = MhcPostMapping()

        for i in core.pp_layer_ids:
            d = L[i]
            for name in _MX_LINEARS:
                if name in d:
                    self._bind_mx(name, i, d[name])
            self._attn_norm.bind_layered(i, weight=d["attn_norm"])
            self._ffn_norm.bind_layered(i, weight=d["ffn_norm"])
            self._q_norm.bind_layered(i, weight=d["q_norm"])
            self._kv_norm.bind_layered(i, weight=d["kv_norm"])
            self._attn.bind_layered(i, sinks=d["sink"])
            if core.fp8_attn:
                s_kv = core._fp8_kv_scale[i]
                self._attn8.bind_layered(i, sinks=d["sink"], bmm2_scale=s_kv)
                # The call's QK scale is softmax scale x query scale x KV scale;
                # the quantizer writes it with its query scale folded in.
                self._q_e4m3.bind_layered(i, scale_mult=core.head_dim**-0.5 * s_kv)
            self._wo_a.bind_layered(i, b=d["wo_a_t"])
            self._router.bind_layered(i, mat_b=d["gate_w_t"])
            self._gate.bind_layered(i, bias=d["gate_bias"])
            self._gate_vl.bind_layered(i, bias=d["gate_bias_vl"])
            self._moe_quant.bind_layered(i, global_scale=d["g1"])
            self._moe_runner.bind_layered(
                i,
                gemm1_weights=d["fc1_w"],
                gemm1_weights_scale=d["fc1_s"],
                gemm1_clamp_limit=d["clamp"],
                gemm2_weights=d["fc2_w"],
                gemm2_weights_scale=d["fc2_s"],
                output1_scale_scalar=d["o1"],
                output1_scale_gate_scalar=d["o1g"],
                output2_scale_scalar=d["o2"],
            )
            fn, sc, bs = d["hc_attn"]
            self._hc_attn_gemm.bind_layered(i, w=fn)
            self._hc_attn_fuse.bind_layered(i, hc_scale=sc, hc_base=bs)
            # The FFN input collapses with this attention's mixes.
            self._hc_pre_ffn.bind_layered(i, scale=sc, base=bs)
            fn, sc, bs = d["hc_ffn"]
            self._hc_ffn_gemm.bind_layered(i, w=fn)
            self._hc_ffn_fuse.bind_layered(i, hc_scale=sc, hc_base=bs)
            if i > 0:
                # Lagged pre: the attention input collapses with the previous
                # layer's FFN mixes (the previous stage's, at a stage's first layer).
                _, psc, pbs = L[i - 1]["hc_ffn"]
                self._hc_pre_attn.bind_layered(i, scale=psc, base=pbs)
            if i in core.kv_sources:
                self._comp_mm.bind_layered(i, mat_b=d["comp_w_t"])
                self._comp_norm.bind_layered(i, weight=d["comp_norm"])
                self._idx_wk.bind_layered(i, mat_b=d["idx_wk_t"])
                self._idx_k_norm.bind_layered(i, weight=d["idx_k_norm"])
            if i in core.index_sources:
                self._idx_wproj.bind_layered(i, mat_b=d["idx_wproj_t"])
            if i in core.engram_layers:
                li = core.engram_layers.index(i)
                self._eng_gather.bind_layered(
                    i,
                    weight=d["eng_table"],
                    scale=d["eng_scale"],
                    row_start=core.eng_start[li],
                    row_end=core.eng_end[li],
                )
                self._eng_gate.bind_layered(i, q_weight=d["eng_q"], k_weight=d["eng_k"])
        if core.last_pp:
            # The final norm's input collapses with the last layer's FFN mixes.
            _, fsc, fbs = L[core.num_layers - 1]["hc_ffn"]
            self._hc_pre_final.bind_const(scale=fsc, base=fbs)

    def _bind_mx(self, name: str, layer: int, wsf) -> None:
        weight, weight_sf = wsf
        self._mx[name].bind_layered(layer, weight=weight, weight_sf=weight_sf)
        self._mx_shape[name, layer] = tuple(weight.shape)

    def step_args(self, md) -> dict:
        """The metadata surface this target reads, projected once per step.

        States the two context counts instead of reading them back: this target
        runs only when the step has no context request, so reading them would ask
        the metadata to confirm what the routing already decided. Also the
        step-contract probe (``Target._check_step_contract``)."""
        return dict(
            kv_cache_manager=md.kv_cache_manager,
            num_contexts=0,
            num_ctx_tokens=0,
            num_seqs=md.num_seqs,
            all_rank_num_tokens=md.all_rank_num_tokens,
            req_idx_per_token=md.req_idx_per_token,
            cu_seq_lens_cuda=md.cu_seq_lens_cuda,
            kv_lens_cuda_runtime=md.kv_lens_cuda_runtime,
            swa_local_indices_cuda=md.swa_local_indices_cuda,
            sparse_mla_topk_lens=md.sparse_mla_topk_lens,
            compressed_visible_lens_cuda=md.compressed_visible_lens_cuda,
            compressed_kv_lens_cuda=md.compressed_kv_lens_cuda,
            max_compressed_kv_lens=md.max_compressed_kv_lens,
            compress_block_tables=md.compress_block_tables,
            indexer_compress_block_tables=md.indexer_compress_block_tables,
            sliding_block_tables=md.sliding_block_tables,
            cu_new_comp_kv_cuda=md.cu_new_comp_kv_cuda,
            compressed_position_ids_cuda=md.compressed_position_ids_cuda,
            num_total_compressed_tokens=md.num_total_compressed_tokens,
            engram_token_block_table=md.engram_token_block_table,
            v41_swa_slot_mapping=md.v41_swa_slot_mapping,
            v41_compress_slot_mapping=md.v41_compress_slot_mapping,
            v41_index_k_slot_data=md.v41_index_k_slot_data,
            v41_index_k_slot_scale=md.v41_index_k_slot_scale,
            tokens_per_block=md.tokens_per_block,
            max_seq_len=md.max_seq_len,
        )

    def _mx_linear(self, name: str, x, layer: int, out_dtype=torch.bfloat16):
        """MXFP8 x MXFP8 linear: 1x32 UE8M0 activation quantization, then the
        GEMM against the checkpoint's 32x32 blocks expanded to 1x32."""
        m = x.shape[0]
        xq, xsf = self._mxq(x)
        be = _gemm_backend(m)
        _rv("mxfp8_mxfp8_gemm", m, self._mx_shape[name, layer], be)
        return self._mx[name](xq, xsf, out_dtype=out_dtype, backend=be, layer=layer)

    def _allreduce_fp32(self, x):
        """TP-member sum of bf16 per-rank partials in fp32 (the reference's
        fp32 all-reduce of bf16-rounded partials); returns fp32. AUTO on the IPC
        workspace when the group is on one node, NCCL across nodes."""
        x32 = torch.empty(list(x.shape), dtype=torch.float32, device=x.device)
        x32.copy_(x)
        _rv("allreduce", _sh(x32), "fp32", self._ar32_strategy)
        return self._allreduce32(x32)[0]

    def _engram(self, i, h, hashes, md, num_tokens, dev, keep=None):
        """Engram at layer i, in place on the stream h [T, 4, 5120]. ``keep``
        [T] bool is False on image-span rows (their gate is 0), or None."""
        core = self.core
        li = core.engram_layers.index(i)
        ids = hashes[:, li, :]  # [T, 24] int64, row stride 48
        if core.eng_host:
            # Every row is in the shared pinned copy: this rank's own tokens,
            # no zero fill and no collective, on every member.
            vals = self._eng_gather(ids, layer=i)
            red = torch.reshape(vals, [num_tokens, core.eng_in])
        elif core.dp:
            rows = core._dp_rows(md, num_tokens)
            ids_c = torch.empty([num_tokens, core.eng_cols], dtype=torch.int64, device=dev)
            ids_c.copy_(ids)
            ids32 = ids_c.view(torch.int32)
            if rows > num_tokens:
                ids32 = nn.functional.pad(ids32, [0, 0, 0, rows - num_tokens])
            _rv("allgather", _sh(ids32), "int32")
            ids_all = self._allgather(ids32).view(torch.int64)
            vals = self._eng_gather(ids_all, layer=i)
            flat = torch.reshape(vals, [vals.shape[0], core.eng_in])
            _rv("reducescatter", _sh(flat), "bf16")
            red = self._reducescatter(flat)
            if rows > num_tokens:
                red = torch.split(red, [num_tokens, rows - num_tokens], 0)[0]
        else:
            vals = self._eng_gather(ids, layer=i)
            flat = torch.reshape(vals, [num_tokens, core.eng_in])
            _rv("allreduce", _sh(flat), "bf16", _AR_NCCL)
            red = self._allreduce16(flat)[0]
        _rv("engram_gather", num_tokens, li)
        kv = self._mx_linear("eng_wkv", red, i)  # [T, 25600]
        key, value = torch.split(kv, [_HC * core.hidden, core.hidden], -1)
        key3 = torch.reshape(key, [num_tokens, _HC, core.hidden])
        _rv("engram_gate" if keep is None else "engram_gate.masked", num_tokens)
        return self._eng_gate(h, key3, value, token_mask=keep, out=h, layer=i)

    def _attention(self, i, x, st, pools, cache, dev):
        """One attention sublayer on the normed input x [T, 5120] -> [T, 5120] bf16,
        every row a generation row."""
        core = self.core
        r = core.ratios[i]
        T, G, B, next_n = st["T"], st["G"], st["B"], st["next_n"]
        md = st["md"]
        pos = st["pos"]
        table = core._table(r)
        H = core.h_loc
        hd = core.head_dim

        # Low-rank query + the shared window latent.
        qkv = self._mx_linear("wqkv_a", x, i)  # [T, 1792]
        q_lr, kv_raw = torch.split(qkv, [core.q_lora, hd], -1)
        qr = self._q_norm(q_lr, layer=i)
        kv = self._kv_norm(kv_raw, layer=i)
        qrq, qrsf = self._mxq(qr)
        q = self._mx["wq_b"](qrq, qrsf, out_dtype=torch.bfloat16, backend=_gemm_backend(T), layer=i)
        _rv("mla_rope_inplace", T, H, core.nope, r)
        q3 = self._rope(torch.reshape(q, [T, H, hd]), pos, table, H, core.nope, _ROPE, False, False)
        kv = torch.reshape(
            self._rope(
                torch.reshape(kv, [T, 1, hd]), pos, table, 1, core.nope, _ROPE, False, False
            ),
            [T, hd],
        )
        # Window KV: FP8 E4M3 x UE8M0/32 value set into this layer's SWA slots
        # (native: the codes and UE8M0 bytes of the 528 B rows). A pool write.
        native = pools["native"]
        swa_slots = md.v41_swa_slot_mapping[pools["local"][i]][:T]
        if native:
            codes, scales = pools["swa_q"][i]
            _rv("mxfp8_fake_quantize.native", T, _sh(codes))
            self._swa_write(kv, codes, swa_slots, scales)
        else:
            _rv("mxfp8_fake_quantize", T, _sh(pools["swa3"][i]))
            self._swa_write(kv, pools["swa3"][i], swa_slots)

        # Compressed KV + index K (the four KV sources).
        per_query = native and st["comp_form"] == "topk"
        if i in core.kv_sources:
            self._compress(i, x, st, pools, dev)
            if native and not per_query:
                self._native_compressed(i, st, pools)

        # Compressed positions per query (index sources); readers reuse.
        if i in core.index_sources:
            cache["topk"][r] = self._index(i, x, qrq, qrsf, st, pools, cache, dev)
        topk = cache["topk"].get(r) if r else None

        # Pool offsets -> the attention's index table (see the prefill target).
        req = md.req_idx_per_token[:T]
        swa_bt = md.sliding_block_tables[pools["local"][i], DeepseekV4AttentionType.SWA.value]
        swa_local = md.swa_local_indices_cuda[:T]
        if r:
            _rv("dsv4_local_to_global_indices", T, 128, _TOPK, _PAGE // r)
            gidx = self._l2g(
                req,
                swa_bt,
                swa_local,
                _PAGE,
                0,
                pools["comp_tbl"][r] if native and not per_query else md.compress_block_tables[r],
                topk,
                _PAGE // r,
                0,
            )
            if per_query:
                gidx = self._native_topk(i, gidx, topk, T, pools, cache)
            if native:
                comp_pool = pools["comp_c4_q"][r] if per_query else pools["comp_c4_req"][r]
            else:
                comp_pool = pools["comp4"][core.kv_src[i]]
        else:
            _rv("dsv4_local_to_global_indices", T, 128)
            gidx = self._l2g(req, swa_bt, swa_local, _PAGE, 0)
            comp_pool = core._const["dummy_comp8" if core.fp8_attn else "dummy_comp"]
        if native:
            gidx, swa_pool = self._native_window(i, gidx, T, pools)
        else:
            swa_pool = pools["swa4"][i]
        lens = md.sparse_mla_topk_lens[r][:T]
        kv_lens = md.kv_lens_cuda_runtime
        o = torch.empty([T, H, hd], dtype=torch.bfloat16, device=dev)
        if core.fp8_attn:
            # E4M3 query, one power-of-two scale s_q from this step's max|q| (on the device);
            # the call's QK scale is scale * s_q * s_kv (written by the quantizer) and its
            # PV scale s_kv.
            qa = torch.empty([T, H, hd], dtype=torch.float8_e4m3fn, device=dev)
            bmm1 = torch.empty([1], dtype=torch.float32, device=dev)
            _rv("e4m3_quantize_pow2_dynamic", T, H)
            qa, bmm1 = self._q_e4m3(q3, qa, bmm1, layer=i)
            attn = self._attn8
            fp8 = dict(bmm1_scale=bmm1)
        else:
            qa = q3
            attn = self._attn
            fp8 = {}
        kind = "trtllm_sparse_mla_dsv4_fp8" if core.fp8_attn else "trtllm_sparse_mla_dsv4"
        if T:
            # The call writes `o`, which the inverse RoPE reads.
            _rv(kind + ".decode", G, next_n, H, gidx.shape[1])
            attn(
                torch.reshape(qa, [G, next_n, H, hd]),
                swa_pool,
                comp_pool,
                sparse_indices=gidx,
                sparse_topk_lens=lens,
                seq_lens=kv_lens[:B],
                out=torch.reshape(o, [G, next_n, H, hd]),
                layer=i,
                **fp8,
            )
        # Remove the query rotation from the output's rope tail.
        o = self._rope(o, pos, table, H, core.nope, _ROPE, True, False)
        # Grouped low-rank output: per group [T, 4096] x [4096, 1024], into `oa`.
        oa = torch.empty([T, core.g_loc * core.o_lora], dtype=torch.bfloat16, device=dev)
        _rv("bmm_out", core.g_loc, T, core.group_in, core.o_lora)
        self._wo_a(
            torch.transpose(torch.reshape(o, [T, core.g_loc, core.group_in]), 0, 1),
            out=torch.transpose(torch.reshape(oa, [T, core.g_loc, core.o_lora]), 0, 1),
            layer=i,
        )
        y = self._mx_linear("wo_b", oa, i)  # [T, 5120] bf16 (per-rank partial under TP)
        if core.dp:
            return y
        y32 = self._allreduce_fp32(y)
        yb = torch.empty([T, core.hidden], dtype=torch.bfloat16, device=dev)
        yb.copy_(y32)
        return yb

    def _compress(self, i, x, st, pools, dev):
        """KV source i: this step's compressed entries -> index K (native
        MXFP4 pool) and main KV (E2M1 x E4M3/16 value set, bf16 pool), over
        generation rows only."""
        core = self.core
        r = core.ratios[i]
        md = st["md"]
        T, G, B, next_n = st["T"], st["G"], st["B"], st["next_n"]
        hd = core.head_dim
        cu = md.cu_seq_lens_cuda
        if r == 2:
            n = md.num_total_compressed_tokens[2]
            x32 = torch.empty([T, core.hidden], dtype=torch.float32, device=dev)
            x32.copy_(x)
            kv_score = self._comp_mm(x32, layer=i)  # fp32 [T, 1024] = [kv || score]
            pooled = torch.empty([n, hd], dtype=torch.bfloat16, device=dev)
            # Reserved decode rows the op leaves unwritten stay finite.
            pooled.copy_(core._const["zero_bf16"].expand([n, hd]))
            local = pools["local"][i]
            bt_kv = md.sliding_block_tables[local, DeepseekV4AttentionType.COMPRESSOR_KV.value]
            bt_sc = md.sliding_block_tables[local, DeepseekV4AttentionType.COMPRESSOR_SCORE.value]
            cu_comp = md.cu_new_comp_kv_cuda[2]
            if G:
                # Fills `pooled` and updates the compressor state pools.
                _rv("compressor_paged_kv_compress", G, next_n, n)
                self._comp_paged(
                    kv_score,
                    paged_kv=pools["ckv"][i],
                    paged_score=pools["csc"][i],
                    block_table_kv=bt_kv[:B],
                    block_table_score=bt_sc[:B],
                    output=pooled,
                    kv_lens=md.kv_lens_cuda_runtime[:B],
                    cu_seq_lens=cu[: B + 1],
                    cu_kv_comp=cu_comp[: B + 1],
                    batch_size=G,
                    next_n=next_n,
                )
            # The op above ran even with no completed group: it stores this
            # step's pending odd token in the compressor state.
            if n == 0:
                return
            latent = self._comp_norm(pooled, layer=i)
        else:
            n = T
            assert md.num_total_compressed_tokens[1] == T, (md.num_total_compressed_tokens[1], T)
            latent = self._comp_norm(self._comp_mm(x, layer=i), layer=i)
        cpos = md.compressed_position_ids_cuda[r][:n]
        tbl = core._rope_tables[1]
        # Index K from the RoPE-free latent (before the rotation below).
        k = self._idx_k_norm(self._idx_wk(latent, layer=i), layer=i)
        k = torch.reshape(
            self._rope(
                torch.reshape(k, [n, 1, _IDX_DIM]),
                cpos,
                tbl,
                1,
                _IDX_DIM - _ROPE,
                _ROPE,
                False,
                False,
            ),
            [n, _IDX_DIM],
        )
        kd, ksf = self._mxfp4(k)
        _rv("indexer_k_cache_scatter", n, r, _sh(pools["idxk"][i]))
        self._idx_scatter(
            kd,
            torch.reshape(ksf.view(torch.int32), [n, 1]),
            pools["idxk"][i],
            md.v41_index_k_slot_data[r][:n],
            md.v41_index_k_slot_scale[r][:n],
            n,
        )
        # Main KV: rope at the group-first position j*r, then the E2M1 x E4M3/16
        # value set into the owner's COMPRESS slots. A pool write.
        latent = torch.reshape(
            self._rope(
                torch.reshape(latent, [n, 1, hd]), cpos, tbl, 1, core.nope, _ROPE, False, False
            ),
            [n, hd],
        )
        slots = md.v41_compress_slot_mapping[r][:n]
        if pools["native"]:
            codes, scales = pools["comp_q"][i]
            _rv("nvfp4_fake_quantize.native", n, r, _sh(codes))
            self._comp_write(latent, codes, slots, scales)
        else:
            _rv("nvfp4_fake_quantize", n, r, _sh(pools["comp3"][i]))
            self._comp_write(latent, pools["comp3"][i], slots)

    def _native_compressed(self, i, st, pools):
        """Native storage, KV source i (after its write): every compressed
        entry of every request of the step, dequantized into the compact
        buffer at row ``b * L_r + e`` -- all the rows the layers reading this
        source can name (``compressed_kv_lens`` = kv_len // r per request).
        The buffer is read through ``comp_c4_req``."""
        core = self.core
        r = core.ratios[i]
        md = st["md"]
        # Rows are the engine's request indices (block-table rows); in the
        # decoder replay pass st["B"] counts its compacted request list.
        B = st.get("B_engine", st["B"])
        codes, scales = pools["comp_q4"][i]
        out = pools["comp_out"][r][:B]
        if core.fp8_attn:
            _rv("nvfp4_kv_to_e4m3", B, out.shape[1], r, _sh(codes))
            self._comp_read8(
                codes,
                scales,
                md.compress_block_tables[r][:B],
                md.compressed_kv_lens_cuda[r][:B],
                out,
                1.0 / core._fp8_kv_scale[i],
            )
            return
        _rv("nvfp4_kv_to_bf16", B, out.shape[1], r, _sh(codes))
        self._comp_read(
            codes,
            scales,
            md.compress_block_tables[r][:B],
            md.compressed_kv_lens_cuda[r][:B],
            core._const["one_f32"],
            out,
        )

    def _native_topk(self, i, gidx, topk, T, pools, cache):
        """Native storage, per-query read buffer, layer i: the index table
        with its compressed half renumbered onto ``topk_out`` (query t's
        column k at row ``t * 512 + k``). The rows are dequantized from the
        pool slots the index op named (-1: left unwritten) when the top-k set
        or its KV source changes; the layers that share both reuse them.

        Renumbering every column is exact: the index op packs a row's valid
        entries first and ``sparse_mla_topk_lens`` counts exactly those, so
        every -1 column is masked, and a masked column carries no weight
        whatever finite row it names (every row of the buffer is finite: zero
        at allocation, dequantized values after)."""
        core = self.core
        win, comp = torch.split(gidx, [_WINDOW, _TOPK], 1)
        src = core.kv_src[i]
        last = cache.get("comp_rows")
        if last is None or last[0] is not topk or last[1] != src:
            slots = pools["topk_slots"][:T]
            slots.copy_(comp)
            codes, scales = pools["comp_q1"][src]
            if core.fp8_attn:
                # Every layer reading this source shares its KV scale (``_derive_fp8_kv_scales``).
                _rv("nvfp4_kv_to_e4m3.topk", T, _TOPK, core.ratios[i], _sh(codes))
                self._comp_read8(
                    codes,
                    scales,
                    slots,
                    pools["topk_cols"][:T],
                    pools["topk_out"][:T],
                    1.0 / core._fp8_kv_scale[i],
                )
            else:
                _rv("nvfp4_kv_to_bf16.topk", T, _TOPK, core.ratios[i], _sh(codes))
                self._comp_read(
                    codes,
                    scales,
                    slots,
                    pools["topk_cols"][:T],
                    core._const["one_f32"],
                    pools["topk_out"][:T],
                )
            # Holding the top-k tensor keeps its identity unique for the step.
            cache["comp_rows"] = (topk, src)
        return torch.cat([win, pools["topk_ren"][:T]], 1)

    def _native_window(self, i, gidx, T, pools):
        """Native storage, layer i: dequantize the window rows each query's
        index row names into the compact buffer -- query t's column c to row
        ``t * 128 + c``, a -1 column to a zero row -- and return the index
        table with its window half renumbered onto that buffer (the compressed
        half is unchanged), and the buffer as the attention's window pool.

        A zero row stands in for -1 exactly: the kernel reads an attended -1
        as an all-zero row, and a masked column (past the query's window count)
        carries no weight whatever finite row it names."""
        core = self.core
        codes, scales = pools["swa_q"][i]
        if gidx.shape[1] == _WINDOW:
            win, rest = gidx, None
        else:
            win, rest = torch.split(gidx, [_WINDOW, gidx.shape[1] - _WINDOW], 1)
        if core.fp8_attn:
            _rv("mxfp8_to_e4m3", T * _WINDOW, _sh(pools["swa_c"]), _sh(codes))
            self._swa_read8(
                codes,
                scales,
                pools["swa_c"],
                1.0 / core._fp8_kv_scale[i],
                torch.reshape(win, [T * _WINDOW]),
            )
        else:
            _rv("mxfp8_dequantize", T * _WINDOW, _sh(pools["swa_c"]), _sh(codes))
            self._swa_read(codes, scales, pools["swa_c"], torch.reshape(win, [T * _WINDOW]))
        ren = pools["win_idx"][:T]
        return (ren if rest is None else torch.cat([ren, rest], 1)), pools["swa_c4"]

    def _index(self, i, x, qrq, qrsf, st, pools, cache, dev):
        """Index source i: top-512 compressed positions per generation row,
        [T, 512] int32, valid entries first in ascending order, -1 after."""
        core = self.core
        r = core.ratios[i]
        md = st["md"]
        T, G, B, next_n = st["T"], st["G"], st["B"], st["next_n"]
        src = core.kv_src[i]
        iq = self._mx["idx_wq_b"](
            qrq, qrsf, out_dtype=torch.bfloat16, backend=_gemm_backend(T), layer=i
        )
        iq3 = self._rope(
            torch.reshape(iq, [T, _IDX_HEADS, _IDX_DIM]),
            st["pos"],
            core._rope_tables[1],
            _IDX_HEADS,
            _IDX_DIM - _ROPE,
            _ROPE,
            False,
            False,
        )
        qd, qsf = self._mxfp4(iq3)  # [T,32,64] u8, [T*128] u8
        q8 = qd.view(torch.int8)
        qs32 = torch.reshape(qsf.view(torch.int32), [T, _IDX_HEADS])
        wb = self._idx_wproj(x, layer=i)  # bf16 [T, 32], 2^-6 folded in
        w32 = torch.empty([T, _IDX_HEADS], dtype=torch.float32, device=dev)
        w32.copy_(wb)
        idx = torch.empty([T, _TOPK], dtype=torch.int32, device=dev)
        reindex = core.cand_src < i
        if T:
            vis = md.compressed_visible_lens_cuda[r][:T]
            ctx_lens = torch.reshape(vis, [G, next_n])
            sched = cache["sched"].get(r)
            if sched is None:
                sched = self._paged_meta(ctx_lens, _PAGE // r)
                cache["sched"][r] = sched
            max_ctx = md.max_compressed_kv_lens[r]
            _rv("deepgemm_fp8_fp4_paged_mqa_logits", G, next_n, r, max_ctx)
            logits = self._paged_logits(
                torch.reshape(q8, [G, next_n, _IDX_HEADS, _IDX_DIM // 2]),
                torch.reshape(qs32, [G, next_n, _IDX_HEADS]),
                pools["idxk"][src],
                w32,
                ctx_lens,
                md.indexer_compress_block_tables[r][:B],
                sched,
                max_ctx,
            )
            if core._cand_binds_decode:
                if i == core.cand_src:
                    _rv("candidate_block_select.decode", T, max_ctx)
                    cache["cand_gen"] = self._cand(logits, None, vis)
                elif reindex:
                    logits = torch.where(cache["cand_gen"], logits, float("-inf"))
            _rv("flashinfer_top_k_ragged_deterministic.decode", T, max_ctx)
            idx.copy_(self._topk(logits, core._const["zero_i32_rows"][:T], vis, None))
        return idx

    def _route(self, i, chunk, img, dev):
        """Router of one token chunk [n, 5120] bf16: the gate GEMM (bf16 x and
        the checkpoint's bf16 weight on tensor cores, fp32 accumulation, fp32
        logits; every product is exact in fp32, the summation order is the
        tensor cores'), then top-6 by ``sqrt(softplus(logit)) + bias``. With
        ``img`` ([n] bool, True on image-span rows) those rows select with
        ``gate.bias_vl`` instead: a second routing call over the same logits
        and a per-row select of both results. Returns (weights [n, 6] fp32,
        ids [n, 6] int32)."""
        core = self.core
        n = chunk.shape[0]
        logits = self._router(chunk, layer=i)  # fp32 [n, 384]
        _rv("gate_forward", n, core.num_experts, core.topk)
        tw, tid = self._gate(logits, layer=i)
        if img is None:
            return tw, tid
        _rv("gate_forward.bias_vl", n, core.num_experts, core.topk)
        tw_vl, tid_vl = self._gate_vl(logits, layer=i)
        sel = torch.reshape(img, [n, 1])
        return torch.where(sel, tw_vl, tw), torch.where(sel, tid_vl, tid)

    def _moe(self, i, x, rows, dev, img=None):
        """Routed NVFP4 experts + FP8 shared expert on the normed input x.
        ``img`` is the image-span mask of the rows the router sees (this rank's
        rows, or the gathered rows under attention DP), or None. ``rows`` is
        the dep4 per-rank row count every rank pads to (None under attention
        TP). ``x`` None is a dep4 rank with no row in this pass: it sends zero
        rows and returns None."""
        core = self.core
        T = 0 if x is None else x.shape[0]
        assert T or core.dp, "only a dep4 rank runs the MoE without rows"
        if T:
            # Shared expert (replicated, this rank's own tokens): [gate ; up].
            h13 = self._mx_linear("sh_w13", x, i)
            _rv("silu_and_mul", T, 2 * core.moe_inter)
            act = self._silu(h13)
            shared = self._mx_linear("sh_w2", act, i)
        # Routed: dep4 gathers the token set ahead of the router, so every rank
        # routes the identical set and the four expert windows tile it once.
        pad_rows = 0
        xr = x
        if core.dp:
            assert rows is not None and T <= rows, (T, rows)
            pad_rows = rows - T
            if not T:
                xp = torch.empty([rows, core.hidden], dtype=torch.bfloat16, device=dev)
                xp.copy_(core._const["zero_bf16"].expand([rows, core.hidden]))
            else:
                xp = x if not pad_rows else nn.functional.pad(x, [0, 0, 0, pad_rows])
            _rv("allgather", _sh(xp), "bf16")
            xr = self._allgather(xp)
        parts = []
        sizes = _moe_chunks(xr.shape[0])
        if img is not None:
            assert img.shape[0] == xr.shape[0], (tuple(img.shape), tuple(xr.shape))
        img_chunks = torch.split(img, sizes, 0) if img is not None else [None] * len(sizes)
        for chunk, img_c in zip(torch.split(xr, sizes, 0), img_chunks):
            n = chunk.shape[0]
            tw, tid = self._route(i, chunk, img_c, dev)
            twb = torch.empty([n, core.topk], dtype=torch.bfloat16, device=dev)
            twb.copy_(tw)
            hs, hsf = self._moe_quant(chunk, layer=i)
            out = torch.empty([n, core.hidden], dtype=torch.bfloat16, device=dev)
            _rv("fp4_block_scale_moe_runner", n, core.e_loc, core.e_off, core.i_loc)
            # Given `output`, the runner finalizes into it and returns an empty tensor.
            self._moe_runner(
                hidden_states=hs,
                hidden_states_scale=hsf.view(torch.float8_e4m3fn),
                topk_weights=twb,
                topk_ids=tid,
                output=out,
                layer=i,
            )
            parts.append(out)
        routed = parts[0] if len(parts) == 1 else torch.cat(parts, 0)
        if core.dp:
            r32 = torch.empty(list(routed.shape), dtype=torch.float32, device=dev)
            r32.copy_(routed)
            _rv("reducescatter", _sh(r32), "fp32")
            r32 = self._reducescatter(r32)
            if not T:
                return None
            if pad_rows:
                r32 = torch.split(r32, [T, pad_rows], 0)[0]
        else:
            r32 = self._allreduce_fp32(routed)
        m32 = torch.add(r32, shared)  # fp32: the shared expert joins after the reduce
        m = torch.empty([T, core.hidden], dtype=torch.bfloat16, device=dev)
        m.copy_(m32)
        return m

    def _dp_gather_mask(self, loc, rows, T, dev):
        """dep4: the image-span mask of the all-gathered token set the router
        sees, each rank padded to ``rows`` as the MoE input is. ``loc`` is this
        rank's [T] mask, or None for T text rows (T may be 0: a pass that holds
        none of this rank's rows)."""
        assert T <= rows, (T, rows)
        if loc is None or not T:
            loc = torch.empty([rows if not T else T], dtype=torch.bool, device=dev)
            loc.copy_(self.core._const["false_b"].expand([loc.shape[0]]))
        u8 = loc.view(torch.uint8)
        if rows > loc.shape[0]:
            u8 = nn.functional.pad(u8, [0, rows - loc.shape[0]])
        _rv("allgather", _sh(u8), "uint8")
        return self._allgather(u8).view(torch.bool)

    def _dp_late_rows(self, mine: list, dev) -> list:
        """dep4 decoder replay: the passes every rank runs over the bounded
        layers, as each pass's row count padded to its max over ranks.

        ``mine`` is this rank's pass row counts (a bounded rank's slices, or its
        step rows as one pass). The bounded layers' MoE is a collective, so the
        ranks must run the same passes in lockstep; a rank's decoder rows can
        also outnumber its step rows (a small last chunk still runs its whole
        tail, the rest from the stash). Every rank calls this in every eager
        step under decoder replay, prefill and decode alike: ``[count, rows_0,
        rows_1, ...]`` rides in the int32 [1, 48] form the Engram ids gather
        in, and the result has one entry per pass of the rank with the most
        (0-row entries never occur: some rank holds the rows of every pass)."""
        core = self.core
        width = 2 * core.eng_cols  # 48
        vals = [len(mine), *mine]
        assert len(vals) <= width, (len(mine), width)
        core._ensure_ar(max(vals) + 1, dev)
        cols = [core._ar32[v : v + 1] for v in vals] + [core._ar32[0:1]] * (width - len(vals))
        v = torch.reshape(torch.cat(cols, 0), [1, width])
        _rv("allgather", _sh(v), "int32")
        got = self._allgather(v).tolist()  # [world, 48]
        assert len(got) == core.tp and got[core.tp_rank][: len(vals)] == vals, (
            core.tp_rank,
            vals,
            got,
        )
        return _dp_pass_rows(got)

    def _dp_idle_pass(self, rows: int, dev) -> None:
        """dep4 decoder replay: this rank's place in a pass over the bounded
        layers that holds none of its rows -- the mask gather and every
        bounded layer's routed round trip, with zero rows."""
        img = self._dp_gather_mask(None, rows, 0, dev)
        for i in range(self.core.late, self.core.num_layers):
            self._moe(i, None, rows, dev, img)

    def _scratch(self, s, T: int, dev) -> None:
        """The per-row scratch of one pass over the T rows of stream ``s.h``."""
        core = self.core
        f32 = torch.float32
        s.T = T
        s.h_flat = torch.reshape(s.h, [T, core.hc_k])
        s.tile = _tile_n(T)
        s.yA = torch.empty([T, _MIX], dtype=f32, device=dev)
        s.rA = torch.empty([T], dtype=f32, device=dev)
        s.mix4 = torch.empty([T, _HC], dtype=f32, device=dev)
        s.post = torch.empty([T, _HC], dtype=f32, device=dev)
        s.comb = torch.empty([T, _HC, _HC], dtype=f32, device=dev)
        s.no_res = torch.empty([T, _HC, 0], dtype=torch.bfloat16, device=dev)
        s.no_li = torch.empty([T, 0], dtype=torch.bfloat16, device=dev)
        s.coll = torch.empty([T, core.hidden], dtype=torch.bfloat16, device=dev)
        _rv("mhc", T, s.tile)

    def _layer(self, i: int, s, dev) -> None:
        """Layer i over the pass ``s`` (its stream, scratch, step dict, cache)."""
        core = self.core
        T, h, h_flat = s.T, s.h, s.h_flat
        # DSpark reads the INPUT stream of its target layers (37-39), every
        # token; the metadata averages the 4 hc copies itself.
        if s.spec is not None and s.spec.is_layer_capture(i):
            s.spec.maybe_capture_hidden_states(i, h_flat)
        if i in core.engram_layers:
            h = self._engram(i, h, s.hashes, s.md, T, dev, s.keep)
        # --- attention sublayer ------------------------------------
        s.yA, s.rA = self._hc_attn_gemm(h_flat, y=s.yA, r=s.rA, M=T, tile_n=s.tile, layer=i)
        s.post, s.comb, s.no_li = self._hc_attn_fuse(
            s.yA,
            s.rA,
            s.no_res,
            post_mix=s.post,
            comb_mix=s.comb,
            layer_input=s.no_li,
            M=T,
            layer=i,
        )
        if i == 0:
            a_in = s.emb  # one-hot pre on copy 0: the embedding itself
        else:
            # Lagged pre: the previous FFN's mixes collapse this input.
            s.mix4.copy_(torch.split(s.yF, [_HC, _MIX - _HC], 1)[0])
            s.coll = self._hc_pre_attn(s.mix4, s.rF, h, s.coll, M=T, layer=i)
            a_in = s.coll
        x = self._attn_norm(a_in, layer=i)
        a = self._attention(i, x, s.st, s.pools, s.cache, dev)
        h = self._hc_post(h, a, s.post, s.comb, h)
        # --- FFN sublayer --------------------------------------------
        s.yF, s.rF = self._hc_ffn_gemm(h_flat, y=s.yF, r=s.rF, M=T, tile_n=s.tile, layer=i)
        s.post, s.comb, s.no_li = self._hc_ffn_fuse(
            s.yF,
            s.rF,
            s.no_res,
            post_mix=s.post,
            comb_mix=s.comb,
            layer_input=s.no_li,
            M=T,
            layer=i,
        )
        s.mix4.copy_(torch.split(s.yA, [_HC, _MIX - _HC], 1)[0])
        s.coll = self._hc_pre_ffn(s.mix4, s.rA, h, s.coll, M=T, layer=i)
        x = self._ffn_norm(s.coll, layer=i)
        m = self._moe(i, x, s.dp_rows, dev, s.img)
        self._hc_post(h, m, s.post, s.comb, h)

    def _final(self, s):
        """The final collapse with layer 39's FFN mixes, then the final norm."""
        s.mix4.copy_(torch.split(s.yF, [_HC, _MIX - _HC], 1)[0])
        s.coll = self._hc_pre_final(s.mix4, s.rF, s.h, s.coll, M=s.T)
        return self._final_norm(s.coll)

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: torch.IntTensor | None = None,
        position_ids: torch.IntTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        lora_params: dict | None = None,
        **kwargs,
    ) -> torch.Tensor:
        core = self.core
        self._check_step_contract(attn_metadata)
        md = attn_metadata
        step = self.step_args(md)
        spec_metadata = kwargs.get("spec_metadata")
        pools = core._get_pools(md)
        dev = input_ids.device
        T = input_ids.shape[0]
        B = step["num_seqs"]
        next_n = T // B if B else 1
        assert next_n * B == T, (T, B)
        # A DSpark generation row verifies its 5 drafts: 1 + 5 tokens.
        assert next_n == 1 or (spec_metadata is not None and next_n == 1 + core.ds.block), (
            f"generation rows of {next_n} tokens (spec_metadata {'set' if spec_metadata is not None else 'None'})"
        )
        st = {
            "md": md,
            "T": T,
            "tc": step["num_ctx_tokens"],
            "gen": T,
            "nc": step["num_contexts"],
            "G": B,
            "B": B,
            "next_n": next_n,
            "pos": torch.reshape(position_ids, [-1]),
            "max_ctx_q": 0,
            "replay": 0,
            "tail_bound": False,
        }
        cache = {"topk": {}, "ctx_k": {}, "sched": {}}

        emb = nn.functional.embedding(input_ids, core.w["embed"]) if core.first_pp else None
        eager = not torch.cuda.is_current_stream_capturing()
        # dep4: every rank pads its MoE rows to dp_rows (past layer L under
        # decoder replay, to its pass's count instead: _dp_late_rows).
        dp_rows = core._dp_rows(md, T) if core.dp else None
        img_moe = None
        if core.dp and eager:
            # A peer rank may hold image rows in a prefill step: the router's
            # gathered mask needs this rank's (all text) rows too.
            img_moe = self._dp_gather_mask(None, dp_rows, T, dev)

        # Engram hash ids of every token (int64 [T, 2, 24]); also writes this
        # step's compressed ids into the paged history pool. The history is read
        # and written at the rows' own positions (see the prefill target for why
        # cached_token_lens_cuda is not a substitute).
        hashes = None
        if core.eng_local:
            eng_start = torch.reshape(
                nn.functional.embedding(
                    step["cu_seq_lens_cuda"][:B], torch.reshape(st["pos"], [T, 1])
                ),
                [B],
            )
            _rv("engram_hash", T, B)
            hashes = self._eng_hash(
                input_ids,
                step["req_idx_per_token"][:T],
                step["cu_seq_lens_cuda"][: B + 1],
                eng_start,
                pools["tok"],
                step["engram_token_block_table"],
            )

        h = torch.empty([T, _HC, core.hidden], dtype=torch.bfloat16, device=dev)
        yF = torch.empty([T, _MIX], dtype=torch.float32, device=dev)
        rF = torch.empty([T], dtype=torch.float32, device=dev)
        if core.first_pp:
            h.copy_(torch.reshape(emb, [T, 1, core.hidden]).expand([T, _HC, core.hidden]))
        else:
            # A later pipeline stage starts from the previous stage's stream and
            # its last FFN mixes (the lagged pre of this stage's first attention).
            _rv("pp_recv_tensors", T)
            self._pp_recv([h, yF, rF])

        s = SimpleNamespace(
            h=h,
            yF=yF,
            rF=rF,
            st=st,
            cache=cache,
            img=img_moe,
            spec=spec_metadata,
            emb=emb,
            hashes=hashes,
            keep=None,
            md=md,
            pools=pools,
            dp_rows=dp_rows,
        )
        self._scratch(s, T, dev)
        # Native KV storage: the compressed read buffer's form for the whole step.
        st["comp_form"] = core._comp_form(pools, md, st, None) if pools["native"] else None
        # dep4 decoder replay: a decode rank holds one pass over the bounded
        # layers (its step rows) and joins any further pass a prefill peer runs.
        dp_late = None
        if core.dp and core.replay and eager:
            dp_late = self._dp_late_rows([T], dev)

        stop = core.late if dp_late is not None else core.num_layers
        for i in core.pp_layer_ids:
            if i >= stop:
                break
            self._layer(i, s, dev)
        if not core.last_pp:
            # Hand the stream and the last FFN mixes to the next stage; this
            # stage's output rows are never read (its lm_head is skipped).
            _rv("pp_send_tensors", T)
            self._pp_send([s.h, s.yF, s.rF])
            return s.coll
        if dp_late is not None:
            s.dp_rows = dp_late[0]
            s.img = self._dp_gather_mask(None, dp_late[0], T, dev)
            for i in range(core.late, core.num_layers):
                self._layer(i, s, dev)
            for k in range(1, len(dp_late)):
                self._dp_idle_pass(dp_late[k], dev)
        return self._final(s)


class VisionTower:
    """DeepSeek-ViT + aligner + span assembly (the checkpoint's
    ``inference/vision.py`` and ``model.py`` ``merge_image_embeddings``).

    Holds no parameters of its own: its weights are the ``vis*`` rows of the
    core's weight table, and it binds its own entries to them in
    ``post_load_weights``. Replicated on every rank, no collective. It runs per
    context chunk, over the images whose spans meet that chunk, ONE IMAGE PER
    CALL: an image's rows depend on that image alone -- not on the chunking, nor
    on the request's other images -- and a step's vision memory is one image's
    working set.

    Input: one request's ``multimodal_data["image"]`` from the sibling
    ``input_processor`` (``P`` patches, ``C`` aligner cells, ``S`` span positions
    over all its images, in prompt order): ``pixel_values`` [P, 3, 14, 14]
    bf16, ``vit_cu_seqlens`` / ``aligner_cu_cells`` / ``span_lengths`` (host
    lists), ``vit_rope_cos`` / ``vit_rope_sin`` [P, 32] fp32, and the
    image-local tables ``aligner_index_local`` [C, 9] (value P_i = image i's
    zero pad) and ``span_source_local`` [S] (delimiters at C_i, C_i + 1,
    C_i + 2).

    Per image (attention stays within it)::

        x = patch_w . pixels + b                                 [P, 1024]
        32 x (pre-norm, eps 1e-6):
            qkv = wqkv . rmsnorm(x) + b; 2D rotate-half RoPE on q, k
            x  += wo . attention(qkv) + b          (bidirectional, 16 x 64)
            x  += w2 . silu_and_mul(w1 . rmsnorm(x))
        v = rmsnorm(x)
        cells = v rows gathered 3x3 per cell (zero pad)          [C, 9216]
        out   = w2 . gelu(w1 . cells + b1) + b2                  [C, 5120]
        span  = [out; image_start; image_end; image_newline][span_source_local]

    The packed qkv is roped in place as ``[3P, 16, 64]`` rows: q and k of
    patch t read table row t, v reads an identity row (cos 1, sin 0) and is
    left unchanged. An image has at most 9189 patches and 1021 cells (the
    1024-position span limit), inside the per-call envelope the entries
    certify. Every per-call length / position tensor is a view or a concat
    of views of load-time ``arange`` tables.

    **CUDA graphs (switch ``TRTLLM_DSV41_VIT_CUDA_GRAPH``, unset = on; none
    under ``llm_args.disable_mm_encoder``).** The
    tower call -- patch embed, 32 blocks, final norm -- is captured at load
    time once per padded patch count ``B`` in ``_VIT_GRAPH_BUCKETS`` (all
    graphs in one memory pool), and an image of P patches replays the
    smallest ``B >= P``. Every row-wise op runs over B rows (rows P..B-1 are
    padding and never reach a real row); the attention is the same
    ``thop_attention`` call over two sequence slots -- the image, then a pad
    sequence over rows [P, B) -- with ``max_context_q_len_override = B``,
    which keeps the FMHA's launch shape independent of the split, so a replay
    follows whatever lengths and offsets the device buffers hold. Before a
    replay the image's pixels, RoPE rows and the slot lengths / offsets are
    copied into the graphs' static buffers; after it the P output rows are
    copied out. The aligner and the span gathers stay eager. With the switch
    at 0 every tower call runs eagerly over its P rows.

    Replicated ranks compute the same bits here without any tactic pin: no
    call of the tower, the aligner or the span table goes through the
    AutoTuner (``cublas_mm`` picks its cuBLASLt algorithm from the shape
    alone -- a fixed table, else cuBLASLt's heuristic; the norms, RoPE,
    ``thop`` attention, activations and gathers have one kernel per shape),
    so nothing is chosen per rank."""

    def __init__(self, core: "DeepseekV41UnifiedModelingV2"):
        self._core = core
        self._c: dict | None = None
        # Graph mode: {bucket: CUDAGraph} (None = every call eager) and the
        # static buffers the graphs read and write.
        self._g: dict | None = None
        self._g_bufs: dict | None = None

    def post_load_weights(self) -> None:
        """Bind every entry the tower, aligner and span table call, then capture
        the tower's CUDA graphs (unless disabled)."""
        w = self._core.w
        dev = w["vis_norm"].device
        self._patch = CublasMm()
        self._patch.bind_const(mat_b=w["vis_patch_w"].t(), bias=w["vis_patch_b"])
        self._norm1 = FlashinferRmsnorm()
        self._norm1.bind_const(weight=w["vis0_norm1"], eps=_VIT_EPS)
        self._qkv = CublasMm()
        self._wo = CublasMm()
        self._norm2 = FlashinferRmsnorm()
        self._norm2.bind_const(eps=_VIT_EPS)
        self._w1 = CublasMm()
        self._w2 = CublasMm()
        # Each block ends with the next block's norm1 (the tower's final norm
        # after the last block).
        self._next_norm = FlashinferRmsnorm()
        self._next_norm.bind_const(eps=_VIT_EPS)
        for L in range(_VIT_LAYERS):
            p = f"vis{L}_"
            self._qkv.bind_layered(L, mat_b=w[p + "wqkv_w"].t(), bias=w[p + "wqkv_b"])
            self._wo.bind_layered(L, mat_b=w[p + "wo_w"].t(), bias=w[p + "wo_b"])
            self._norm2.bind_layered(L, weight=w[p + "norm2"])
            self._w1.bind_layered(L, mat_b=w[p + "w1"].t())
            self._w2.bind_layered(L, mat_b=w[p + "w2"].t())
            nxt = w[f"vis{L + 1}_norm1"] if L + 1 < _VIT_LAYERS else w["vis_norm"]
            self._next_norm.bind_layered(L, weight=nxt)
        self._silu = FlashinferSiluAndMul()
        self._rope = MlaRopeInplace()
        self._attn = ThopAttention()
        # Bidirectional attention within each sequence, no pool, no RoPE: every
        # argument but the step's lengths, offsets and buffers is constant.
        self._attn.bind_const(
            k=None,
            v=None,
            output_sf=None,
            kv_cache_block_offsets=None,
            host_kv_cache_pool_pointers=None,
            host_kv_cache_pool_mapping=None,
            cache_indirection=None,
            kv_scale_orig_quant=None,
            kv_scale_quant_orig=None,
            out_scale=None,
            rotary_inv_freq=None,
            rotary_cos_sin=None,
            latent_cache=None,
            q_pe=None,
            block_ids_per_seq=None,
            attention_sinks=None,
            is_fused_qkv=True,
            update_kv_cache=True,
            predicted_tokens_per_seq=1,
            local_layer_idx=0,
            num_heads=_VIT_HEADS,
            num_kv_heads=_VIT_HEADS,
            head_size=_VIT_HEAD_DIM,
            tokens_per_block=None,
            max_context_length=_VIT_ATTN_MAX_LEN,
            max_seq_len=_VIT_ATTN_MAX_LEN,
            attention_window_size=_VIT_ATTN_MAX_LEN,
            beam_width=1,
            mask_type=_VIT_MASK_BIDIRECTIONAL,
            quant_mode=0,
            q_scaling=1.0,
            position_embedding_type=0,
            rope_dim=0,
            rope_base=10000.0,
            rope_scale_type=0,
            rope_scale=1.0,
            rope_short_m_scale=1.0,
            rope_long_m_scale=1.0,
            rope_max_positions=1024,
            rope_original_max_positions=1024,
            use_paged_context_fmha=False,
            attention_input_type=0,
            is_mla_enable=False,
            chunked_prefill_buffer_batch_size=1,
            q_lora_rank=None,
            kv_lora_rank=None,
            qk_nope_head_dim=None,
            qk_rope_head_dim=None,
            v_head_dim=None,
            rope_append=None,
            mrope_rotary_cos_sin=None,
            mrope_position_deltas=None,
            helix_position_offsets=None,
            helix_is_inactive_rank=None,
            attention_chunk_size=None,
            softmax_stats_tensor=None,
            is_spec_decoding_enabled=False,
            use_spec_decoding=False,
            is_spec_dec_tree=False,
            spec_decoding_generation_lengths=None,
            spec_decoding_position_offsets_for_cpp=None,
            spec_decoding_packed_mask=None,
            spec_decoding_bl_tree_mask_offset=None,
            spec_decoding_bl_tree_mask=None,
            spec_bl_tree_first_sparse_mask_offset_kv=None,
            sparse_kv_indices=None,
            sparse_kv_offsets=None,
            sparse_attn_indices=None,
            sparse_attn_offsets=None,
            sparse_attn_indices_block_size=0,
            cu_kv_seqlens=None,
        )
        self._al1 = CublasMm()
        self._al1.bind_const(mat_b=w["vis_al_w1"].t(), bias=w["vis_al_b1"])
        self._al2 = CublasMm()
        self._al2.bind_const(mat_b=w["vis_al_w2"].t(), bias=w["vis_al_b2"])
        self._gelu = MoeGelu()
        ident = torch.zeros((1, 2, _VIT_ROPE_HALF), dtype=torch.float32, device=dev)
        ident[:, 0] = 1.0
        self._c = {
            "span_tail": w["vis_span_tail"],
            # Row v of either table holds the value v: per-call lengths,
            # offsets and positions are its views.
            "ar": torch.arange(_VIT_MAX_PATCHES + 1, dtype=torch.int32, device=dev),
            "ar_host": torch.arange(_VIT_MAX_PATCHES + 1, dtype=torch.int32),
            "zeros_host": torch.zeros(_VIT_GRAPH_SLOTS, dtype=torch.int32),
            "ident": ident,
            # thop_attention grows this in place on its first calls (eager
            # calls only; the graphs have their own).
            "workspace": torch.empty(0, dtype=torch.int8, device=dev),
        }
        if getattr(self._core, "disable_mm_encoder", False):
            # The ViT weights stay loaded, so an image that arrives anyway is
            # still answered, by eager tower calls.
            sys.stdout.write(
                f"MODELING_V2 deepseek_v41 rank {self._core.rank}: disable_mm_encoder: no ViT CUDA "
                "graphs captured (an image, if one arrives, runs the tower eagerly)\n"
            )
            sys.stdout.flush()
        elif _env_switch(_VIT_GRAPH_ENV) is not False:
            self._capture_graphs(dev)

    # -- metadata -------------------------------------------------------------
    def _meta(self, image: dict, i: int) -> dict:
        """The attention / RoPE metadata of image ``i`` of a request (eager call)."""
        c = self._c
        ar, arh = c["ar"], c["ar_host"]
        cu = image["vit_cu_seqlens"]
        p0, P = int(cu[i]), int(cu[i + 1]) - int(cu[i])
        cos = torch.reshape(image["vit_rope_cos"][p0 : p0 + P], [P, 1, _VIT_ROPE_HALF])
        sin = torch.reshape(image["vit_rope_sin"][p0 : p0 + P], [P, 1, _VIT_ROPE_HALF])
        table = torch.cat([torch.cat([cos, sin], 1), c["ident"]], 0)  # [P + 1, 2, 32]
        t = torch.reshape(ar[:P], [P, 1])
        pos = torch.reshape(
            torch.cat([t.expand([P, 2]), torch.reshape(ar[P : P + 1], [1, 1]).expand([P, 1])], 1),
            [3 * P],
        )
        return {
            "P": P,
            "N": 1,
            "rope_table": table,
            "rope_pos": pos,  # (t, t, P) per patch t
            "seq_lens": ar[P : P + 1],
            "cu_seqlens": torch.cat([ar[0:1], ar[P : P + 1]], 0),
            "seq_lens_host": arh[P : P + 1],
            "total_kv_host": torch.cat([arh[P : P + 1], arh[0:1]], 0),
            "types_host": c["zeros_host"][:1],
            "q_len_override": None,
            "workspace": c["workspace"],
        }

    # -- pieces ---------------------------------------------------------------
    def _patch_embed(self, px):
        """[P, 588] pixels -> [P, 1024]; a P in the split-K band runs on
        zero-padded rows and keeps the first P output rows."""
        P = px.shape[0]
        rows = _patch_embed_rows(P)
        _rv("vision.cublas_mm", rows, _VIT_PATCH_IN, _VIT_DIM)
        if rows == P:
            return self._patch(px)
        x = self._patch(nn.functional.pad(px, [0, 0, 0, rows - P]))
        return torch.split(x, [P, rows - P], 0)[0]

    def _attention(self, qkv, m):
        """Bidirectional attention within each sequence of the call."""
        P, N = m["P"], m["N"]
        out = torch.empty([P, _VIT_DIM], dtype=torch.bfloat16, device=qkv.device)
        if m["q_len_override"] is None:
            _rv("vision.thop_attention", P, N)
        else:
            _rv("vision.thop_attention.graph", P, N, m["q_len_override"])
        return self._attn(
            q=qkv,
            output=out,
            workspace_=m["workspace"],
            sequence_length=m["seq_lens"],
            host_past_key_value_lengths=m["seq_lens_host"],
            host_total_kv_lens=m["total_kv_host"],
            context_lengths=m["seq_lens"],
            host_context_lengths=m["seq_lens_host"],
            host_request_types=m["types_host"],
            max_context_q_len_override=m["q_len_override"],
            max_num_requests=N,
            cu_q_seqlens=m["cu_seqlens"],
            num_contexts=N,
            num_ctx_tokens=P,
        )

    def _block(self, L, x, h, m):
        """Block L on the residual stream ``x`` with ``h = norm1(x)``; returns
        ``(x_out, next_norm(x_out))``."""
        P = m["P"]
        qkv = self._qkv(h, layer=L)  # [P, 3072]
        qkv = torch.reshape(
            self._rope(
                torch.reshape(qkv, [3 * P, _VIT_HEADS, _VIT_HEAD_DIM]),
                m["rope_pos"],
                m["rope_table"],
                _VIT_HEADS,
                0,
                _VIT_HEAD_DIM,
                False,
                True,
            ),
            [P, 3 * _VIT_DIM],
        )
        a = self._wo(self._attention(qkv, m), layer=L)
        x = torch.add(x, a)
        h2 = self._norm2(x, layer=L)
        act = self._silu(self._w1(h2, layer=L))
        x = torch.add(x, self._w2(act, layer=L))
        return x, self._next_norm(x, layer=L)

    def _tower(self, px, m):
        """One tower call: [P, 588] pixels -> ViT output [P, 1024]."""
        x = self._patch_embed(px)
        h = self._norm1(x)
        for L in range(_VIT_LAYERS):
            x, h = self._block(L, x, h, m)
        return h

    def _vit(self, image: dict, i: int):
        """ViT output [P, 1024] of image ``i`` of a request (one tower call)."""
        cu = image["vit_cu_seqlens"]
        p0, P = int(cu[i]), int(cu[i + 1]) - int(cu[i])
        assert 0 < P <= _VIT_MAX_IMAGE_PATCHES, (i, P)
        pv = image["pixel_values"]
        px = torch.reshape(pv, [pv.shape[0], _VIT_PATCH_IN])[p0 : p0 + P]
        _rv("vision.vit", P, 1)
        if self._g is not None:
            return self._vit_graph(image, px, p0)
        return self._tower(px, self._meta(image, i))

    # -- CUDA graphs ----------------------------------------------------------
    def _graph_meta(self, B: int) -> dict:
        """The metadata a bucket-B graph is captured with: the static device
        slot lengths / offsets (rewritten before each replay), and host values
        that hold the B rows as one sequence (read only at capture)."""
        g, arh = self._g_bufs, self._c["ar_host"]
        return {
            "P": B,
            "N": _VIT_GRAPH_SLOTS,
            "rope_table": g["table"],
            "rope_pos": g["pos"][: 3 * B],
            "seq_lens": g["seq"],
            "cu_seqlens": g["cu"],
            "seq_lens_host": torch.cat([arh[B : B + 1], arh[0:1]], 0),
            "total_kv_host": torch.cat([arh[B : B + 1], arh[0:1]], 0),
            "types_host": self._c["zeros_host"],
            "q_len_override": B,
            "workspace": g["workspace"],
        }

    def _set_graph_slots(self, P: int, B: int) -> None:
        """Slot lengths (P, B - P) and offsets (0, P, B) of an image of P
        patches in bucket B."""
        g, ar = self._g_bufs, self._c["ar"]
        g["seq"].copy_(torch.cat([ar[P : P + 1], ar[B - P : B - P + 1]], 0))
        g["cu"].copy_(torch.cat([ar[0:1], ar[P : P + 1], ar[B : B + 1]], 0))

    def _graph_body(self, B: int, m: dict) -> None:
        """What a bucket-B graph runs: the tower over the static pixel rows,
        its output into the static output rows."""
        g = self._g_bufs
        g["h"][:B].copy_(self._tower(g["px"][:B], m))

    def _stage_graph_inputs(self, image: dict, px, p0: int) -> int:
        """Copy one image's inputs into the graphs' static buffers; returns its bucket."""
        g = self._g_bufs
        P = px.shape[0]
        B = next(b for b in _VIT_GRAPH_BUCKETS if b >= P)
        g["px"][:P].copy_(px)
        cos = torch.reshape(image["vit_rope_cos"][p0 : p0 + P], [P, 1, _VIT_ROPE_HALF])
        sin = torch.reshape(image["vit_rope_sin"][p0 : p0 + P], [P, 1, _VIT_ROPE_HALF])
        g["table"][:P].copy_(torch.cat([cos, sin], 1))
        self._set_graph_slots(P, B)
        return B

    def _vit_graph(self, image: dict, px, p0: int):
        """One tower call replayed from the smallest bucket that holds it."""
        P = px.shape[0]
        B = self._stage_graph_inputs(image, px, p0)
        _rv("vision.vit_graph", B, P)
        self._g[B].replay()
        out = torch.empty([P, _VIT_DIM], dtype=torch.bfloat16, device=px.device)
        out.copy_(self._g_bufs["h"][:P])
        return out

    def _capture_graphs(self, dev) -> None:
        """Capture one graph per bucket, largest first, into one shared pool.

        Every bucket runs once eagerly before any capture, so the graphs'
        attention workspace reaches its final size first (a later resize would
        leave an earlier graph pointing at freed memory); the eager calls keep
        their own workspace for the same reason."""
        BMAX = _VIT_GRAPH_BUCKETS[-1]
        table = torch.zeros((BMAX + 1, 2, _VIT_ROPE_HALF), dtype=torch.float32, device=dev)
        table[BMAX:] = self._c["ident"]
        t = torch.arange(BMAX, dtype=torch.int32, device=dev).view(BMAX, 1)
        pos = torch.cat(
            [t.expand(BMAX, 2), torch.full((BMAX, 1), BMAX, dtype=torch.int32, device=dev)], 1
        )
        self._g_bufs = {
            "px": torch.zeros((BMAX, _VIT_PATCH_IN), dtype=torch.bfloat16, device=dev),
            # Rows 0..BMAX-1: the staged image's RoPE rows; row BMAX: the identity.
            "table": table,
            "pos": pos.reshape(3 * BMAX).contiguous(),  # (t, t, BMAX) per patch t
            "seq": torch.zeros(_VIT_GRAPH_SLOTS, dtype=torch.int32, device=dev),
            "cu": torch.zeros(_VIT_GRAPH_SLOTS + 1, dtype=torch.int32, device=dev),
            "workspace": torch.empty(0, dtype=torch.int8, device=dev),
            "h": torch.zeros((BMAX, _VIT_DIM), dtype=torch.bfloat16, device=dev),
        }
        free0 = torch.cuda.mem_get_info(dev)[0]
        t0 = time.perf_counter()
        graphs = {}
        with torch.inference_mode():
            metas = {B: self._graph_meta(B) for B in _VIT_GRAPH_BUCKETS}
            for B in reversed(_VIT_GRAPH_BUCKETS):
                self._set_graph_slots(B, B)
                self._graph_body(B, metas[B])
            torch.cuda.synchronize(dev)
            pool = torch.cuda.graph_pool_handle()
            stream = torch.cuda.Stream(dev)
            for B in reversed(_VIT_GRAPH_BUCKETS):
                self._set_graph_slots(B, B)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(
                    graph, pool=pool, stream=stream, capture_error_mode="thread_local"
                ):
                    self._graph_body(B, metas[B])
                graphs[B] = graph
            torch.cuda.synchronize(dev)
        # The capture-time host tensors are read at capture only; kept with the graphs.
        self._g_metas = metas
        self._g = graphs
        torch.cuda.empty_cache()
        held = free0 - torch.cuda.mem_get_info(dev)[0]
        sys.stdout.write(
            f"MODELING_V2 deepseek_v41 rank {self._core.rank}: ViT CUDA graphs: {len(graphs)} buckets "
            f"{_VIT_GRAPH_BUCKETS[0]}..{BMAX} patches captured in {time.perf_counter() - t0:.1f} s; "
            f"device memory held {held / 2**20:.1f} MiB (static buffers + graph pool)\n"
        )
        sys.stdout.flush()

    # -- aligner and spans ------------------------------------------------------
    def _aligner(self, cells):
        """Gathered cells [C, 9, 1024] -> [C, 5120], in chunks of <= 2048 cells."""
        c = self._c
        C = cells.shape[0]
        outs = []
        for c0 in range(0, C, _AL_MAX_ROWS):
            n = min(_AL_MAX_ROWS, C - c0)
            flat = torch.reshape(cells[c0 : c0 + n], [n, _VIT_RATIO * _VIT_RATIO * _VIT_DIM])
            hh = self._al1(flat)
            _rv("vision.moe_gelu", n)
            g = self._gelu(hh, c["ar"][n : n + 1], c["ar"][1:2], n)
            outs.append(self._al2(g))
        return outs[0] if len(outs) == 1 else torch.cat(outs, 0)

    def image_rows(self, image: dict, i: int):
        """[S_i, 5120] bf16: the whole span of image ``i`` of a request."""
        ccu = image["aligner_cu_cells"]
        c0, c1 = int(ccu[i]), int(ccu[i + 1])
        s0 = sum(int(n) for n in image["span_lengths"][:i])
        s1 = s0 + int(image["span_lengths"][i])
        v = self._vit(image, i)
        # Row P_i of the padded ViT output is the zero pad.
        cells = nn.functional.embedding(
            image["aligner_index_local"][c0:c1], nn.functional.pad(v, [0, 0, 0, 1])
        )
        out = self._aligner(cells)
        # Rows C_i, C_i + 1, C_i + 2 = start, end, newline.
        table = torch.cat([out, self._c["span_tail"]], 0)
        return nn.functional.embedding(image["span_source_local"][s0:s1], table)

    def profile_largest_image(self, max_num_tokens: int) -> None:
        """The largest vision work of one context step, for the engine's memory
        profiling: the tower runs one image per call and each image's rows go
        into the step's embedding before the next image runs, so that work is
        one image of the largest size -- a 3 x 3063 patch grid (9189 patches,
        1021 aligner cells, a 1024-position span, the ``max_image_tokens``
        limit) -- next to the step's [max_num_tokens, 5120] embedding. Runs it
        (graph replay or eager, as the engine will) and keeps nothing."""
        from .input_processor import (
            ImageGrid,
            PreprocessedImage,
            VisionGeometry,
            build_vision_inputs,
        )

        dev = self._c["ar"].device
        geometry = VisionGeometry(
            patch_size=14, downsample_ratio=_VIT_RATIO, max_image_tokens=1024, min_pixels=544 * 544,
            max_wh_ratio=None, hidden_size=_VIT_DIM, num_heads=_VIT_HEADS, num_layers=_VIT_LAYERS,
            rope_theta=10000.0, image_token_id=_IMAGE_TOKEN,
        )  # fmt: skip
        grid = ImageGrid(
            n_llm_h=1, n_llm_w=1021, best_height=42, best_width=42882, n_vit_h=3, n_vit_w=3063
        )
        assert grid.num_span_tokens == geometry.max_image_tokens
        patches = torch.zeros((grid.n_vit_h * grid.n_vit_w, 3, 14, 14), dtype=torch.bfloat16)
        image = build_vision_inputs([PreprocessedImage(patches=patches, grid=grid)], [0], geometry)
        image = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in image.items()}
        S = grid.num_span_tokens
        R = max(int(max_num_tokens), 1)
        emb = torch.empty([R, self._core.hidden], dtype=torch.bfloat16, device=dev)
        rows = self.image_rows(image, 0)
        emb[: min(R, S)].copy_(rows[: min(R, S)])
        torch.cuda.synchronize(dev)
        sys.stdout.write(
            f"MODELING_V2 deepseek_v41 rank {self._core.rank}: memory profiling ran the largest image "
            f"({grid.n_vit_h * grid.n_vit_w} patches, {S}-position span, "
            f"{'graph' if self._g is not None else 'eager'} tower) beside a {R}-row embedding\n"
        )
        sys.stdout.flush()


# The drafter's MXFP8 x MXFP8 linears: per stage, then the two shared operands.
_DS_STAGE_LINEARS = ("wqkv_a", "wq_b", "wo_b", "sh_w13", "sh_w2")
_DS_SHARED_LINEARS = ("main_proj", "wkv3")


class DSparkDraftModel:
    """The object ``DSv4DSparkWorker`` drives as ``draft_model``
    (``tensorrt_llm/_torch/speculative/dspark.py``).

    Holds no parameters and no state of its own: its weights are the core's
    ``w_ds`` (``DRAFT_WEIGHTS``: stage ``s`` as ``l{s}_*``) and the trunk's
    embedding, its head is the shell's ``lm_head``, its RoPE table is the core's
    theta-10000 table 0, and the per-request rings ``[rows, 3, 128, 512]`` are
    the worker's. It binds its own entries in ``post_load_weights``, never at
    construction (the engine replaces the meta-device parameters when it
    materializes them). It takes no attention metadata: the draft step is
    generation-only and the prefill seeding writes rings, so it has no phase to
    route on.

    **Positions.** The worker hands frames (frame = position + 1). The drafter
    ropes at the position (frame - 1) everywhere and writes ring slot
    ``(frame - 1) & 127`` -- the reference's ``start_pos % window``. A ring's
    ``valid_len`` rows are the most recent contiguous positions ending at the
    bonus position ``p``, so a block query attends slots
    ``(p - valid_len + 1 + c) & 127`` for ``c < valid_len``: the first
    ``valid_len`` columns of its index row, which is all the kernel reads of
    the window segment. That is ``0..valid_len-1`` when the request's first
    ring row sat at a slot-0 position (a cold prefill), but not after a
    prefix-cache hit, whose seeding starts at the first uncached position; a
    full ring keeps plain slot order.

    **What one draft step computes** (``forward_batched``, G requests, N = 5G):
    ``main_x = main_norm(main_proj(main_hidden))`` once with stage 0's weights;
    every stage's ring row for the bonus position,
    ``fp8_qdq(rope(kv_norm_s(wkv_s(main_x))))``; block ids ``[bonus, noise x 4]``
    through the trunk embedding into 4 hc copies; three trunk-shaped blocks
    (lagged-pre mHC chain, attention, MoE) whose attention reads the stage's
    ring plus all 5 block rows, non-causal, with sinks; the head collapse, norm
    and the shell's ``lm_head``; then the sequential Markov correction
    ``logits[j] += W2 . W1[prev_j]`` in fp32 with ``prev_{j+1} = argmax``.
    """

    block_size = _DS_BLOCK
    num_stages = _DS_STAGES

    def __init__(self, core: "DeepseekV41UnifiedModelingV2", lm_head):
        assert core.ds is not None, "the core was built without the DSpark drafter"
        self.g = core.ds
        self._attn_params = {
            "window_size": self.g.window,
            "head_dim": self.g.head_dim,
            "n_heads": self.g.heads,
            "eps": self.g.eps,
        }
        self._core = core
        self._lm_head = lm_head
        # Written by the worker: the rows the RoPE table must hold.
        self._runtime_position_ceiling = 0
        self.dummy_loaded = False
        self._c: dict | None = None
        self._tabs: dict = {}

    # -- runtime surface ------------------------------------------------------
    def run_moe_lockstep_noop(self, all_rank_draft_tokens, device) -> None:
        """The drafter's MoE is replicated on every rank and has no collective:
        a rank with no generation rows owes its peers nothing."""
        return None

    def load_weights_from_target_model(self, model) -> None:
        """LoadFormat.DUMMY only: the drafter's rows are in the core, which the
        dummy loader already filled."""
        self.dummy_loaded = True

    # -- post-load binding ----------------------------------------------------
    def post_load_weights(self) -> None:
        """Bind every entry the drafter calls, build its load-time constants, and
        run one eager call of every drafter GEMM shape on each ``mxfp8_mxfp8_gemm``
        backend."""
        core, g = self._core, self.g
        w, wd = core.w, core.w_ds
        dev = wd["norm"].device
        eps = g.eps

        self._mxq = Mxfp8Quantize()
        self._mx = {name: Mxfp8Mxfp8Gemm() for name in _DS_STAGE_LINEARS + _DS_SHARED_LINEARS}
        self._mx_shape: dict = {}
        for name in _DS_SHARED_LINEARS:
            self._mx[name].bind_const(weight=wd[name], weight_sf=wd[name + "_sf"])
            self._mx_shape[name, None] = tuple(wd[name].shape)
        self._q_norm = FlashinferRmsnorm()
        self._kv_norm = FlashinferRmsnorm()
        self._attn_norm = FlashinferRmsnorm()
        self._ffn_norm = FlashinferRmsnorm()
        for op in (self._q_norm, self._kv_norm, self._attn_norm, self._ffn_norm):
            op.bind_const(eps=eps)
        self._main_norm = FlashinferRmsnorm()
        self._main_norm.bind_const(weight=wd["main_norm"], eps=eps)
        self._norm = FlashinferRmsnorm()
        self._norm.bind_const(weight=wd["norm"], eps=eps)
        self._rope = MlaRopeInplace()
        # The FP8 1x32 value set of a K/V row: ring rows and the block's rows.
        self._kv_write = Mxfp8FakeQuantize()
        self._attn = TrtllmSparseMlaDsv4()
        self._attn.bind_const(
            workspace_buffer=core._const["attn_ws"],
            softmax_scale=g.head_dim**-0.5,
            output_scale=1.0,
        )
        self._wo_a = BmmOut()
        # The router: fp32 x and the checkpoint's gate weight promoted to fp32.
        self._router = CublasMm()
        self._gate = GateForward()
        self._gate.bind_const(topk=g.topk, route_scale=g.route_scale)
        self._silu = SiluAndMul()
        self._silu.bind_const(swiglu_limit=g.swiglu)
        # The MXFP4 runner reads its MXFP8 activation linear, padded to 512 columns.
        self._moe_quant = Mxfp8Quantize()
        self._moe_quant.bind_const(swizzled_layout=_SF_LINEAR, alignment=_DS_MOE_ALIGN)
        self._moe_runner = Mxe4m3Mxe2m1BlockScaleMoeRunner()
        self._hc_attn_gemm = MhcGemmSqrsumFma()
        self._hc_ffn_gemm = MhcGemmSqrsumFma()
        for op in (self._hc_attn_gemm, self._hc_ffn_gemm):
            op.bind_const(N=_MIX, K=_HC * g.hidden)
        self._hc_attn_fuse = MhcBigFuse()
        self._hc_ffn_fuse = MhcBigFuse()
        for op in (self._hc_attn_fuse, self._hc_ffn_fuse):
            op.bind_const(
                K=_HC * g.hidden,
                hidden_size=0,
                rms_eps=eps,
                hc_pre_eps=core.hc_eps,
                hc_sinkhorn_eps=core.hc_eps,
                hc_post_mult_value=2.0,
                sinkhorn_repeat=20,
                num_splits=1,
                block_size=128,
            )
        self._hc_pre_attn = MhcHcHeadApply()
        self._hc_pre_ffn = MhcHcHeadApply()
        self._hc_pre_final = MhcHcHeadApply()
        for op in (self._hc_pre_attn, self._hc_pre_ffn, self._hc_pre_final):
            op.bind_const(
                mult=_HC, hidden_size=g.hidden, K=_HC * g.hidden, norm_eps=eps, eps=core.hc_eps
            )
        self._hc_post = MhcPostMapping()
        self._mk_head = CublasMm()
        self._mk_head.bind_const(mat_b=wd["mk_head"].t())

        for s in range(_DS_STAGES):
            p = f"l{s}_"
            for name in _DS_STAGE_LINEARS:
                self._mx[name].bind_layered(s, weight=wd[p + name], weight_sf=wd[p + name + "_sf"])
                self._mx_shape[name, s] = tuple(wd[p + name].shape)
            self._q_norm.bind_layered(s, weight=wd[p + "q_norm"])
            self._kv_norm.bind_layered(s, weight=wd[p + "kv_norm"])
            self._attn_norm.bind_layered(s, weight=wd[p + "attn_norm"])
            self._ffn_norm.bind_layered(s, weight=wd[p + "ffn_norm"])
            self._attn.bind_layered(s, sinks=wd[p + "sink"])
            self._wo_a.bind_layered(
                s,
                b=torch.transpose(
                    wd[p + "wo_a"].view(g.o_groups, g.o_lora, g.heads * g.head_dim // g.o_groups),
                    1,
                    2,
                ),
            )
            self._router.bind_layered(s, mat_b=wd[p + "gate_w"].t())
            self._gate.bind_layered(s, bias=wd[p + "gate_bias"])
            self._moe_runner.bind_layered(
                s,
                gemm1_weights=wd[p + "fc1_w"],
                gemm1_weights_scale=wd[p + "fc1_s"],
                gemm2_weights=wd[p + "fc2_w"],
                gemm2_weights_scale=wd[p + "fc2_s"],
            )
            self._hc_attn_gemm.bind_layered(s, w=wd[p + "hc_attn_fn"])
            self._hc_attn_fuse.bind_layered(
                s, hc_scale=wd[p + "hc_attn_scale"], hc_base=wd[p + "hc_attn_base"]
            )
            # The FFN input collapses with this stage's attention mixes.
            self._hc_pre_ffn.bind_layered(
                s, scale=wd[p + "hc_attn_scale"], base=wd[p + "hc_attn_base"]
            )
            self._hc_ffn_gemm.bind_layered(s, w=wd[p + "hc_ffn_fn"])
            self._hc_ffn_fuse.bind_layered(
                s, hc_scale=wd[p + "hc_ffn_scale"], hc_base=wd[p + "hc_ffn_base"]
            )
            if s > 0:
                # Lagged pre: the previous stage's FFN mixes collapse this input.
                q = f"l{s - 1}_"
                self._hc_pre_attn.bind_layered(
                    s, scale=wd[q + "hc_ffn_scale"], base=wd[q + "hc_ffn_base"]
                )
        # Head collapse with the last stage's FFN mixes.
        q = f"l{_DS_STAGES - 1}_"
        self._hc_pre_final.bind_const(scale=wd[q + "hc_ffn_scale"], base=wd[q + "hc_ffn_base"])

        # valid_len -> max(valid_len, 1) as an int32 row lookup: a sparse-MLA
        # row with seq_lens 0 attends the WHOLE 128-row window page.
        vl_floor = (
            torch.arange(g.window + 1, dtype=torch.int32, device=dev).clamp(min=1).view(-1, 1)
        )
        # bonus + 1 -> max(bonus, 0): rejection sampling can hand a -1 bonus,
        # which an embedding lookup would fault on.
        bonus_shift = (
            (torch.arange(g.vocab + 1, dtype=torch.int64, device=dev) - 1).clamp(min=0).view(-1, 1)
        )
        # valid_len -> (128 - max(valid_len, 1), slot mask): the first ring
        # slot a block query reads is ((start_pos + shift) & mask), i.e.
        # (p - valid + 1) & 127, or slot 0 for a full ring.
        v1 = torch.arange(g.window + 1, dtype=torch.int64, device=dev).clamp(min=1)
        ring_shift = (g.window - v1).view(-1, 1)
        ring_mask = torch.where(v1 < g.window, g.window - 1, 0).view(-1, 1)
        self._c = {
            "mk_embed": wd["mk_embed"],
            "embed": w["embed"],
            "noise": torch.full((1, 1), g.noise, dtype=torch.int64, device=dev),
            "neg1_i64": torch.full((1,), -1, dtype=torch.int64, device=dev),
            "one_i64": torch.full((1,), 1, dtype=torch.int64, device=dev),
            "zero_bf16": torch.zeros((1, 1), dtype=torch.bfloat16, device=dev),
            "j5": torch.arange(_DS_BLOCK, dtype=torch.int64, device=dev).view(1, _DS_BLOCK),
            "stage_base": (torch.arange(_DS_STAGES, dtype=torch.int64, device=dev) * _WINDOW).view(
                1, _DS_STAGES
            ),
            "vl_floor": vl_floor.contiguous(),
            "bonus_shift": bonus_shift.contiguous(),
            "ring_shift": ring_shift.contiguous(),
            "ring_mask": ring_mask.contiguous(),
            "c128": torch.arange(g.window, dtype=torch.int64, device=dev).view(1, g.window),
        }
        # The routed experts' SwiGLU clamp, one value for every expert.
        self._moe_runner.bind_const(
            routing_logits=None,
            routing_bias=None,
            gemm1_bias=None,
            gemm1_alpha=None,
            gemm1_beta=None,
            gemm1_clamp_limit=torch.full((g.experts,), g.swiglu, dtype=torch.float32, device=dev),
            gemm2_bias=None,
            num_experts=g.experts,
            top_k=g.topk,
            n_group=None,
            topk_group=None,
            intermediate_size=g.inter,
            valid_hidden_size=g.hidden,
            valid_intermediate_size=g.inter,
            local_expert_offset=0,
            local_num_experts=g.experts,
            routed_scaling_factor=None,
            routing_method_type=_ROUTING_METHOD_INERT,
            act_type=_ACT_SWIGLU,
        )
        self._tabs = {}
        self._warm_gemm_backends(dev)

    def _warm_gemm_backends(self, dev) -> None:
        """One eager call of every drafter GEMM shape on each backend
        (cute-dsl at M <= 32, cudnn above): a backend's first-ever call inside a
        CUDA-graph capture breaks the capture, and cudnn's first call moves
        FlashInfer's cached workspace once."""
        assert not torch.cuda.is_current_stream_capturing()
        linears = [(name, None) for name in _DS_SHARED_LINEARS]
        for s in range(_DS_STAGES):
            linears += [(name, s) for name in _DS_STAGE_LINEARS]
        seen = set()
        for name, s in linears:
            n, k = self._mx_shape[name, s]
            if (n, k) in seen:
                continue
            seen.add((n, k))
            for m in (1, 33):
                x = torch.empty([m, k], dtype=torch.bfloat16, device=dev)
                x.copy_(self._c["zero_bf16"].expand([m, k]))
                self._mx_linear(name, x, s)
        torch.cuda.synchronize()

    def _tables(self, rows: int, dev) -> dict:
        """Constant index tables for a ring of ``rows`` slots, built on the
        first (eager) call for that ring: stage-page bases, the block-row
        columns, the live-column count, and the zero-padded block K/V buffer."""
        t = self._tabs.get(rows)
        if t is not None:
            return t
        assert not torch.cuda.is_current_stream_capturing(), (
            "drafter index tables are built on the first eager call for a ring size; "
            "run one eager draft step (the graph runner's warm-up does) before capture"
        )
        r = torch.arange(rows, dtype=torch.int64, device=dev)
        s3 = torch.arange(_DS_STAGES, dtype=torch.int64, device=dev)
        base3 = (r[:, None] * _DS_STAGES + s3[None, :]) * _WINDOW  # [rows, 3]
        blk = torch.full((rows, _DS_BLK_COLS), -1, dtype=torch.int32, device=dev)
        blk[:, :_DS_BLOCK] = (
            r[:, None] * _DS_BLOCK + torch.arange(_DS_BLOCK, device=dev)[None, :]
        ).to(torch.int32)
        lens = torch.full((rows * _DS_BLOCK,), _DS_IDX_LEN, dtype=torch.int32, device=dev)
        blk_kv = torch.zeros(
            (rows * _DS_BLOCK + _DS_BLK_PAD_ROWS, self.g.head_dim), dtype=torch.bfloat16, device=dev
        )
        t = {"base3": base3.contiguous(), "blk": blk, "lens": lens, "blk_kv": blk_kv}
        self._tabs[rows] = t
        return t

    def _rope_table(self) -> torch.Tensor:
        """The core's theta-10000, no-YaRN table (grown only in its first forward)."""
        tab = self._core._rope_tables[0]
        need = max(int(self._runtime_position_ceiling), 1)
        assert tab.shape[0] >= need, (
            f"RoPE table has {tab.shape[0]} rows, the worker's position ceiling is {need}"
        )
        return tab

    # -- compositions ----------------------------------------------------------
    def _mx_linear(self, name: str, x, layer=None, out_dtype=torch.bfloat16):
        """MXFP8 x MXFP8 linear (the trunk's recipe and backend choice)."""
        m = x.shape[0]
        xq, xsf = self._mxq(x)
        be = _gemm_backend(m)
        _rv("dspark.mxfp8_mxfp8_gemm", m, self._mx_shape[name, layer], be)
        return self._mx[name](xq, xsf, out_dtype=out_dtype, backend=be, layer=layer)

    def _main_x(self, hidden):
        """main_norm(main_proj(captured rows)), stage 0's weights."""
        return self._main_norm(self._mx_linear("main_proj", hidden))

    def _ring_write(self, main_x, pos64, idx3, ring3, table) -> None:
        """Every stage's ``fp8_qdq(rope(kv_norm_s(wkv_s(main_x)), pos))`` into
        ring rows ``idx3[:, s]`` of ``ring3`` (pages of 128 rows); a negative
        row is skipped. A write into the worker's rings."""
        M = main_x.shape[0]
        hd = self.g.head_dim
        kv3 = self._mx_linear("wkv3", main_x)  # [M, 3 * 512]
        parts = torch.split(kv3, [hd] * _DS_STAGES, -1)
        p32 = torch.empty([M], dtype=torch.int32, device=main_x.device)
        p32.copy_(pos64)
        cols = torch.split(idx3, 1, 1)
        _rv("dspark.ring_write", M, _sh(ring3))
        for s in range(_DS_STAGES):
            kv = self._kv_norm(parts[s], layer=s)
            kv = torch.reshape(
                self._rope(
                    torch.reshape(kv, [M, 1, hd]), p32, table, 1, hd - _ROPE, _ROPE, False, False
                ),
                [M, hd],
            )
            self._kv_write(kv, ring3, torch.reshape(cols[s], [M]))

    def write_context_windows(self, hidden, window_positions, window) -> None:
        """Prefill seeding (eager): ``hidden`` [n <= 128, 15360] captured rows,
        ``window_positions`` [n] int64 frames, ``window`` the slot's
        ``[3, 128, 512]`` ring view."""
        n = hidden.shape[0]
        assert n <= _WINDOW and tuple(window.shape) == (_DS_STAGES, _WINDOW, self.g.head_dim), (
            n,
            tuple(window.shape),
        )
        pos64 = torch.add(window_positions, self._c["neg1_i64"])
        off = torch.bitwise_and(pos64, _WINDOW - 1)
        # [n, 3]: s * 128 + (p & 127)
        idx3 = torch.add(torch.reshape(off, [n, 1]), self._c["stage_base"])
        self._ring_write(self._main_x(hidden), pos64, idx3, window, self._rope_table())

    def write_context_windows_batched(
        self, interim_hidden, interim_pos, slots, interim_valid, kv_windows
    ) -> None:
        """Multi-accept back-fill (captured): ``interim_hidden`` [G, 5, 15360],
        ``interim_pos`` [G, 5] int64 frames, ``slots`` [G] int64,
        ``interim_valid`` [G, 5] bool (masked rows are not written),
        ``kv_windows`` [rows, 3, 128, 512]."""
        G = interim_hidden.shape[0]
        M = G * _DS_BLOCK
        dev = interim_hidden.device
        rows = kv_windows.shape[0]
        tabs = self._tables(rows, dev)
        ring3 = torch.reshape(kv_windows, [rows * _DS_STAGES, _WINDOW, self.g.head_dim])
        pos64 = torch.add(interim_pos, self._c["neg1_i64"])  # [G, 5]
        off = torch.bitwise_and(pos64, _WINDOW - 1)
        bases = nn.functional.embedding(slots, tabs["base3"])  # [G, 3]
        # [G, 5, 3]
        idx = torch.add(
            torch.reshape(bases, [G, 1, _DS_STAGES]), torch.reshape(off, [G, _DS_BLOCK, 1])
        )
        idx = torch.where(torch.reshape(interim_valid, [G, _DS_BLOCK, 1]), idx, self._c["neg1_i64"])
        hid = torch.reshape(interim_hidden, [M, interim_hidden.shape[-1]])
        self._ring_write(
            self._main_x(hid),
            torch.reshape(pos64, [M]),
            torch.reshape(idx, [M, _DS_STAGES]),
            ring3,
            self._rope_table(),
        )

    def _attention(self, s, x, st):
        """One stage's attention over its ring + the transient block."""
        g = self.g
        N, dev, table = st["N"], st["dev"], st["table"]
        H, hd = g.heads, g.head_dim
        qkv = self._mx_linear("wqkv_a", x, s)  # [N, 1792]
        q_lr, kv_raw = torch.split(qkv, [g.q_lora, hd], -1)
        qr = self._q_norm(q_lr, layer=s)
        kv = self._kv_norm(kv_raw, layer=s)
        qrq, qrsf = self._mxq(qr)
        q = self._mx["wq_b"](qrq, qrsf, out_dtype=torch.bfloat16, backend=_gemm_backend(N), layer=s)
        q3 = self._rope(
            torch.reshape(q, [N, H, hd]), st["bpos"], table, H, hd - _ROPE, _ROPE, False, False
        )
        kv = torch.reshape(
            self._rope(
                torch.reshape(kv, [N, 1, hd]), st["bpos"], table, 1, hd - _ROPE, _ROPE, False, False
            ),
            [N, hd],
        )
        # Block K/V: the FP8 1x32 value set, rows 0..N-1 of the padded buffer.
        blk_kv = self._kv_write(kv, st["blk_kv"])
        o = torch.empty([N, 1, H, hd], dtype=torch.bfloat16, device=dev)
        _rv("dspark.trtllm_sparse_mla_dsv4", N, H, _DS_IDX_W, _sh(st["ring4"]))
        o = self._attn(
            torch.reshape(q3, [N, 1, H, hd]),
            st["ring4"],
            torch.reshape(blk_kv, [N, 1, 1, hd]),
            sparse_indices=st["idx"][s],
            sparse_topk_lens=st["lens"],
            seq_lens=st["seq_lens"],
            out=o,
            layer=s,
        )
        o3 = self._rope(
            torch.reshape(o, [N, H, hd]), st["bpos"], table, H, hd - _ROPE, _ROPE, True, False
        )
        gin = H * hd // g.o_groups
        oa = torch.empty([N, g.o_groups * g.o_lora], dtype=torch.bfloat16, device=dev)
        # Grouped low-rank output, into `oa`.
        self._wo_a(
            torch.transpose(torch.reshape(o3, [N, g.o_groups, gin]), 0, 1),
            out=torch.transpose(torch.reshape(oa, [N, g.o_groups, g.o_lora]), 0, 1),
            layer=s,
        )
        return self._mx_linear("wo_b", oa, s)

    def _moe(self, s, x, dev):
        """FP8 shared expert + routed MXFP4 top-3 of 128, replicated (no collective)."""
        g = self.g
        N = x.shape[0]
        h13 = self._mx_linear("sh_w13", x, s)
        act = self._silu(h13)
        shared = self._mx_linear("sh_w2", act, s)
        x32 = torch.empty([N, g.hidden], dtype=torch.float32, device=dev)
        x32.copy_(x)
        logits = self._router(x32, layer=s)  # fp32 [N, 128]
        tw, tid = self._gate(logits, layer=s)
        twb = torch.empty([N, g.topk], dtype=torch.bfloat16, device=dev)
        twb.copy_(tw)
        hs, hsf = self._moe_quant(x)
        out = torch.empty([N, g.hidden], dtype=torch.bfloat16, device=dev)
        _rv("dspark.mxe4m3_mxe2m1_block_scale_moe_runner", N)
        # Every rank computes this drafter MoE on the same rows and must produce
        # the same bits, but the op's AutoTuner entry tunes per rank
        # (DistributedTuningStrategy.INDEPENDENT) inside the engine's tuning
        # pass, a cache file loaded at warm-up (TLLM_AUTOTUNER_CACHE_PATH) can
        # carry per-rank entries for it, and its tactics differ in output bits.
        # In fallback_tactic() the op neither tunes nor reads the cache: it takes
        # its fallback tactic on every rank, eager and under capture alike --
        # the cold state its contract certifies. Given `output`, the
        # runner writes it and returns an empty tensor.
        with fallback_tactic():
            self._moe_runner(
                hidden_states=hs,
                hidden_states_scale=hsf,
                topk_weights=twb,
                topk_ids=tid,
                output=out,
                layer=s,
            )
        return torch.add(out, shared)

    def _blocks(self, h, emb, st):
        """The three stages on the [N, 4, 5120] stream; returns the normed head
        input [N, 5120]."""
        g = self.g
        N, dev = st["N"], st["dev"]
        K = _HC * g.hidden
        tile = _tile_n(N)
        f32 = torch.float32
        yA, rA = (
            torch.empty([N, _MIX], dtype=f32, device=dev),
            torch.empty([N], dtype=f32, device=dev),
        )
        yF, rF = (
            torch.empty([N, _MIX], dtype=f32, device=dev),
            torch.empty([N], dtype=f32, device=dev),
        )
        mix4 = torch.empty([N, _HC], dtype=f32, device=dev)
        post = torch.empty([N, _HC], dtype=f32, device=dev)
        comb = torch.empty([N, _HC, _HC], dtype=f32, device=dev)
        no_res = torch.empty([N, _HC, 0], dtype=torch.bfloat16, device=dev)
        no_li = torch.empty([N, 0], dtype=torch.bfloat16, device=dev)
        coll = torch.empty([N, g.hidden], dtype=torch.bfloat16, device=dev)
        h_flat = torch.reshape(h, [N, K])
        _rv("dspark.mhc", N, tile)
        for s in range(_DS_STAGES):
            yA, rA = self._hc_attn_gemm(h_flat, y=yA, r=rA, M=N, tile_n=tile, layer=s)
            post, comb, no_li = self._hc_attn_fuse(
                yA, rA, no_res, post_mix=post, comb_mix=comb, layer_input=no_li, M=N, layer=s
            )
            if s == 0:
                a_in = emb  # one-hot pre on copy 0: the block embedding itself
            else:
                # Lagged pre: the previous stage's FFN mixes collapse this input.
                mix4.copy_(torch.split(yF, [_HC, _MIX - _HC], 1)[0])
                coll = self._hc_pre_attn(mix4, rF, h, coll, M=N, layer=s)
                a_in = coll
            x = self._attn_norm(a_in, layer=s)
            a = self._attention(s, x, st)
            h = self._hc_post(h, a, post, comb, h)
            yF, rF = self._hc_ffn_gemm(h_flat, y=yF, r=rF, M=N, tile_n=tile, layer=s)
            post, comb, no_li = self._hc_ffn_fuse(
                yF, rF, no_res, post_mix=post, comb_mix=comb, layer_input=no_li, M=N, layer=s
            )
            mix4.copy_(torch.split(yA, [_HC, _MIX - _HC], 1)[0])
            coll = self._hc_pre_ffn(mix4, rA, h, coll, M=N, layer=s)
            x = self._ffn_norm(coll, layer=s)
            m = self._moe(s, x, dev)
            h = self._hc_post(h, m, post, comb, h)
        # Head collapse with the last stage's FFN mixes, then its norm.
        mix4.copy_(torch.split(yF, [_HC, _MIX - _HC], 1)[0])
        coll = self._hc_pre_final(mix4, rF, h, coll, M=N)
        return self._norm(coll)

    def forward_batched(
        self,
        main_hidden,
        bonus,
        start_pos,
        *,
        kv_windows,
        slots,
        valid_len,
        temperature=0.0,
        confidence_threshold=0.0,
        return_logits=True,
        all_rank_num_tokens=None,
    ):
        """One draft step for G generation requests (captured).

        ``main_hidden`` [G, 15360] (the captured row of the last accepted
        position), ``bonus`` [G] int64, ``start_pos`` [G] int64 = the frame of
        ``main_hidden``, ``slots`` [G] int64, ``valid_len`` [G] int64 ring rows
        valid after this step's bonus write. Writes the bonus row of every
        stage's ring first. Returns ``(draft_ids [G, 5] int64, None,
        block_logits [G, 5, V] fp32)``; the worker samples from ``[2]``. The
        Markov feedback is greedy whatever ``temperature`` (the worker passes
        0.0); confidence truncation is not computed. ``all_rank_num_tokens`` is
        unused: the drafter has no collective.
        """
        g = self.g
        G = main_hidden.shape[0]
        N = G * _DS_BLOCK
        dev = main_hidden.device
        hd = g.head_dim
        rows = kv_windows.shape[0]
        assert G <= rows, (G, rows)
        tabs = self._tables(rows, dev)
        table = self._rope_table()
        ring3 = torch.reshape(kv_windows, [rows * _DS_STAGES, _WINDOW, hd])
        ring4 = torch.reshape(kv_windows, [rows * _DS_STAGES, 1, _WINDOW, hd])
        bonus = torch.reshape(
            nn.functional.embedding(torch.add(bonus, self._c["one_i64"]), self._c["bonus_shift"]),
            [G],
        )

        # The bonus position p = start_pos - 1 into every stage's ring.
        main_x = self._main_x(main_hidden)
        pos64 = torch.add(start_pos, self._c["neg1_i64"])
        off = torch.bitwise_and(pos64, _WINDOW - 1)
        bases = nn.functional.embedding(slots, tabs["base3"])  # [G, 3] stage-page bases
        self._ring_write(main_x, pos64, torch.add(bases, torch.reshape(off, [G, 1])), ring3, table)

        # Block ids [bonus, noise x 4] -> the trunk embedding -> 4 hc copies.
        ids = torch.cat(
            [torch.reshape(bonus, [G, 1]), self._c["noise"].expand([G, _DS_BLOCK - 1])], 1
        )
        emb = torch.reshape(nn.functional.embedding(ids, self._c["embed"]), [N, g.hidden])
        h = torch.empty([N, _HC, g.hidden], dtype=torch.bfloat16, device=dev)
        h.copy_(torch.reshape(emb, [N, 1, g.hidden]).expand([N, _HC, g.hidden]))

        # Block positions p + 1 + j = start_pos + j; per-row window counts >= 1.
        bpos = torch.empty([G, _DS_BLOCK], dtype=torch.int32, device=dev)
        bpos.copy_(torch.add(torch.reshape(start_pos, [G, 1]), self._c["j5"]))
        vl = torch.empty([G, _DS_BLOCK], dtype=torch.int32, device=dev)
        vl.copy_(nn.functional.embedding(valid_len, self._c["vl_floor"]).expand([G, _DS_BLOCK]))
        # Index rows [page base + ((first + c) & 127) | g * 5 + 0..4, -1 x 3] per
        # stage, first = (p - valid + 1) & 127 (0 for a full ring).
        first = torch.bitwise_and(
            torch.add(
                torch.reshape(start_pos, [G, 1]),
                nn.functional.embedding(valid_len, self._c["ring_shift"]),
            ),
            nn.functional.embedding(valid_len, self._c["ring_mask"]),
        )  # [G, 1]
        ring_off = torch.bitwise_and(torch.add(first, self._c["c128"]), _WINDOW - 1)  # [G, 128]
        blk = torch.reshape(
            torch.split(tabs["blk"], [G, rows - G], 0)[0], [G, 1, _DS_BLK_COLS]
        ).expand([G, _DS_BLOCK, _DS_BLK_COLS])
        stage_bases = torch.split(bases, 1, 1)
        idx = []
        for s in range(_DS_STAGES):
            ring_cols = torch.empty([G, _WINDOW], dtype=torch.int32, device=dev)
            ring_cols.copy_(torch.add(ring_off, stage_bases[s]))
            ring_cols = torch.reshape(ring_cols, [G, 1, _WINDOW]).expand([G, _DS_BLOCK, _WINDOW])
            idx.append(torch.reshape(torch.cat([ring_cols, blk], -1), [N, _DS_IDX_W]))
        st = {
            "N": N,
            "dev": dev,
            "table": table,
            "ring4": ring4,
            "bpos": torch.reshape(bpos, [N]),
            "seq_lens": torch.reshape(vl, [N]),
            "lens": torch.split(tabs["lens"], [N, rows * _DS_BLOCK - N], 0)[0],
            "blk_kv": torch.split(tabs["blk_kv"], [N, tabs["blk_kv"].shape[0] - N], 0)[0],
            "idx": idx,
        }
        xn = self._blocks(h, emb, st)

        # The shell's head: fp32 block logits [G, 5, V], gathered under TP.
        lg = self._lm_head(xn)
        V = lg.shape[-1]
        logits = torch.empty([G, _DS_BLOCK, V], dtype=torch.float32, device=dev)
        logits.copy_(torch.reshape(lg, [G, _DS_BLOCK, V]))

        # Sequential Markov correction with greedy feedback.
        toks = torch.empty([G, _DS_BLOCK], dtype=torch.int64, device=dev)
        lcols = torch.split(logits, 1, 1)
        tcols = torch.split(toks, 1, 1)
        prev = bonus
        for j in range(_DS_BLOCK):
            lj = torch.reshape(lcols[j], [G, V])
            e = nn.functional.embedding(prev, self._c["mk_embed"])  # [G, 256] bf16
            e32 = torch.empty([G, g.markov_rank], dtype=torch.float32, device=dev)
            e32.copy_(e)
            lj.copy_(torch.add(lj, self._mk_head(e32)))
            prev = torch.argmax(lj, dim=-1)
            torch.reshape(tcols[j], [G]).copy_(prev)
        return toks, None, logits


@register_auto_model("ModelingV2DeepseekV41FlashNvfp4Sm103Unified")
class ModelingV2DeepseekV41FlashNvfp4Sm103Unified(
    DecoderModelForCausalLM[DeepseekV41UnifiedModelingV2, PretrainedConfig]
):
    # `load_weights` streams the rank-local slices it declares out of the
    # mmapped safetensors handles, so the loader hands them over lazily and the
    # checkpoint is never materialized in host RAM.
    requires_lazy_safetensors = True

    def __init__(self, model_config: ModelConfig):
        cfg = model_config.pretrained_config
        assert cfg is not None
        super().__init__(
            DeepseekV41UnifiedModelingV2(model_config),
            config=model_config,
            hidden_size=cfg.hidden_size,
            vocab_size=cfg.vocab_size,
        )
        # The DSpark branch, built only when `speculative_config` asked for it
        # (the core already refused any mode but the embedded DSpark drafter).
        # It is written out rather than inherited from the in-tree one-engine
        # shell: that shell builds its drafter through a module-level function
        # that would construct TensorRT-LLM's own DSpark model instead of this
        # target's. The runtime finds the pieces by attribute: `spec_worker`
        # (the engine), `draft_config` (read by the loader without getattr;
        # None = the drafter's weights come from the same checkpoint) and
        # `load_draft_weights` (the loader's second, lazily mapped pass).
        self.spec_config = getattr(model_config, "spec_config", None)
        self.draft_config = None
        self.draft_model = None
        self.spec_worker = None
        self._draft_weights_loaded = False
        if self.spec_config is not None:
            assert self.model.ds is not None, "core built without the DSpark drafter"
            self.draft_model = DSparkDraftModel(self.model, self.lm_head)
            self.spec_worker = get_spec_worker(self.spec_config, model_config, model_config.mapping)
            self.spec_worker.set_draft_model(self.draft_model)

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: torch.IntTensor | None = None,
        position_ids: torch.IntTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        return_context_logits: bool = False,
        spec_metadata=None,
        lora_params: dict | None = None,
        **kwargs,
    ):
        """One engine step.

        Without a spec worker, a step with no `multimodal_params` delegates to
        the inherited base forward with every argument it was handed
        (`resource_manager` stays inside `**kwargs`), so the text path is the
        base forward exactly. The base forward forwards no `**kwargs` to the
        core, so a step that carries images calls the core itself with them and
        then the same logits processor.

        With one, the shell owns the logits gather: DSpark sets
        `without_logits`, so the engine applies no second gather, and the worker
        takes the trunk's hidden states ungathered beside the gathered logits.
        `position_ids` goes on in the engine's `[1, T]` shape; the worker
        squeezes it itself. `multimodal_params` reaches the core in `**kwargs`."""
        if self.spec_worker is None:
            assert spec_metadata is None, (
                "spec_metadata arrived without a spec worker; this engine was "
                "built without a speculative_config"
            )
            mm_params = kwargs.get("multimodal_params")
            if mm_params:
                hidden = self.model(
                    attn_metadata=attn_metadata,
                    input_ids=input_ids,
                    position_ids=position_ids,
                    inputs_embeds=inputs_embeds,
                    lora_params=lora_params,
                    multimodal_params=mm_params,
                )
                return self.logits_processor.forward(
                    hidden, self.lm_head, attn_metadata, return_context_logits
                )
            return super().forward(
                attn_metadata,
                input_ids,
                position_ids,
                inputs_embeds,
                return_context_logits,
                spec_metadata,
                lora_params,
                **kwargs,
            )
        assert spec_metadata is not None, (
            "the spec worker is built but the engine passed no spec_metadata"
        )
        hidden = self.model(
            attn_metadata=attn_metadata,
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            spec_metadata=spec_metadata,
            lora_params=lora_params,
            **kwargs,
        )
        if not self.model.last_pp:
            # An earlier pipeline stage: the drafter, the logits and the verify
            # run on the last stage, whose sampled and drafted tokens the engine
            # hands back to every stage. This stage's output is never read.
            return hidden
        self._seed_tail_rings()
        # `gather_ids`: one row per context request (its last token) and
        # 1 + 5 per generation request.
        logits = self.logits_processor.forward(
            nn.functional.embedding(spec_metadata.gather_ids, hidden),
            self.lm_head,
            attn_metadata,
            True,
        )
        return self.spec_worker(
            input_ids=input_ids,
            position_ids=position_ids,
            hidden_states=hidden,
            logits=logits,
            attn_metadata=attn_metadata,
            spec_metadata=spec_metadata,
            draft_model=self.draft_model,
            resource_manager=kwargs.get("resource_manager"),
        )

    def profile_inline_mm_encoder(self, max_num_tokens: int) -> None:
        """The engine's memory profiling (``KvCacheCreator``) hook for the
        vision tower, which runs inside the context step: run the largest
        vision work one step can hold, so the measured peak covers it. The
        step keeps nothing of it past the embedding the text dummy also holds.
        Nothing to profile under ``disable_mm_encoder``."""
        if self.model.disable_mm_encoder:
            return
        self.model._vision.profile_largest_image(max_num_tokens)

    def load_weights(self, weights, *args, **kwargs):
        _weights.MODEL_WEIGHTS.load(self, weights)

    def load_draft_weights(self, weights, weight_mapper=None):
        """The loader's second pass over the same checkpoint (lazily mapped):
        the drafter's `mtp.*` rows."""
        assert self.draft_model is not None, "load_draft_weights on an engine built without DSpark"
        _weights.DRAFT_WEIGHTS.load(self, weights)
        self._draft_weights_loaded = True

    def post_load_weights(self):
        super().post_load_weights()
        self.model.post_load_weights()
        if self.draft_model is not None:
            assert self._draft_weights_loaded or self.draft_model.dummy_loaded, (
                "DSpark is on but the loader never ran load_draft_weights: the drafter's "
                "parameters would be uninitialized memory"
            )
            self.draft_model.post_load_weights()

    def _seed_tail_rings(self) -> None:
        """Decoder replay: a tail's rows from earlier chunks exist only in this
        step's decoder pass. Their captured rows go into the request's ring
        (the worker's slot for it) before the worker seeds this chunk's rows,
        so the ring holds the whole tail."""
        pending = self.model._tail_ring
        if not pending:
            return
        self.model._tail_ring = []
        worker = self.spec_worker
        slots = getattr(worker, "_req_to_slot", None) or {}
        for rid, hidden, frames in pending:
            slot = slots.get(rid)
            if slot is not None:
                self.draft_model.write_context_windows(hidden, frames, worker._kv_windows[slot])
