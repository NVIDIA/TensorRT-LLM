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
"""Logical tensor contracts shared by the decode kernels of the glm_moe_dsa targets.

Some kernel packages add PDL inside the package, each with a documented
producer/consumer memory order. The contracts are written for GEN-only,
non-attention-DP, TP8, fixed MTP depth K=5. Each contract lists its logical
tensors; physical layouts follow the native TensorRT-LLM producer and
consumer at that seam. ``R`` is the row capacity of one call (live rows are
selected by a separate mask), ``H`` the local attention-head count.
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class GlmShape:
    """Model-shape constants checked against the GLM-5.2 NVFP4 config."""

    num_layers: int = 78
    num_dense_layers: int = 3
    hidden_size: int = 6144
    num_heads: int = 64
    q_lora_rank: int = 2048
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 192
    qk_rope_head_dim: int = 64
    v_head_dim: int = 256
    index_n_heads: int = 32
    index_head_dim: int = 128
    index_topk: int = 2048
    num_routed_experts: int = 256
    experts_per_token: int = 8
    moe_intermediate_size: int = 2048
    vocab_size: int = 154880
    tp_size: int = 8
    mtp_depth: int = 5

    @property
    def local_heads(self) -> int:
        return self.num_heads // self.tp_size


GLM52 = GlmShape()


@dataclass(frozen=True)
class Contract:
    """One seam between native TensorRT-LLM code and a package.

    ``rules`` are obligations of the seam beyond its tensors: communication,
    ordering and memory order.
    """

    name: str
    owner: str
    inputs: Tuple[str, ...]
    outputs: Tuple[str, ...]
    rules: Tuple[str, ...] = ()


# Rules of the attention_output_tail contract. They bind attention_output_tail and its tail
# continuations in ``attention_output_tail/`` (attention_output_tail_prefetch, and
# attention_output_into_hop apart from the into-hop publication named in the first rule).
ATTENTION_TAIL_COMMUNICATION_RULE = (
    "attention_output_tail owns W_UV + local o_proj and returns the unreduced rank-local "
    "partial. router_gemm/native all-reduce consumes the identical partial. No "
    "communication is hidden inside attention_output_tail. Only attention_output_into_hop (spec "
    "attention_output_into_hop, a declared "
    "cross-boundary fusion) adds a publication: after its griddepcontrol.wait "
    "the o_proj kernel also writes the partial (-0.0 mapped to +0.0) into "
    "the post_attention_hop hop's Lamport transport under the attention_output_into_hop protocol of "
    "glm_5_2_nvfp4__sm_103__tp8/csrc/postAttentionHopFromPartials.h, and the "
    "ordinary partial stays available."
)
ATTENTION_TAIL_PDL_RULE = (
    "attention_output_tail may be a PDL dependent of the latent's producer. Before its "
    "griddepcontrol.wait a attention_output_tail kernel reads only immutable weights (v_b_proj, "
    "local o_proj weight) and writes only its own shared memory "
    "(cp.async.bulk global->shared, completed through mbarriers before use; "
    "no async-proxy global writes). Every latent read and every global write "
    "(v scratch, partial) follows the wait. attention_output_tail's o_proj kernel executes "
    "griddepcontrol.launch_dependents before it writes the partial: every "
    "PDL-launched consumer of the partial must execute griddepcontrol.wait "
    "before reading the partial and before writing any global memory that "
    "may alias a buffer attention_output_tail still reads (v scratch and the latent are pool "
    "memory and may be reused after attention_output_tail returns). Kill switches: "
    "TRTLLM_ENABLE_PDL=0 (all, entry included), "
    "TailConfig(trigger_dependents=False) (exit)."
)

# Physical form of the routed_moe contract's unfinalized tuple as routed_moe returns it.
ROUTED_MOE_UNFINALIZED_LAYOUT_RULE = (
    "A routed-MoE package may return the unfinalized tuple in token-major "
    "form: fc2_output[R*8,6144] BF16 whose row t*8+k holds token t's k-th "
    "expert output (row pitch 6144 elements), expert scales [R,8] BF16 with "
    "the native dtype, values and top-k order, and "
    "expanded_idx_to_permuted_idx[R,8] int32 equal to the identity t*8+k "
    "(every row valid). The native runner's expert-grouped, padded layout is "
    "the other legal form; every unfinalized consumer (native MoEAllReduce "
    "finalize + all-reduce, the post_moe_hop hop) reads fc2_output only through the "
    "permutation, so it accepts either."
)

# Routed input (routed_moe_into_hop): routed_moe's routed result written into the post_moe_hop.
POST_MOE_HOP_ROUTED_INPUT_RULE = (
    "With routed_moe_into_hop selected, routed_moe writes its routed result into a routed-input "
    "buffer owned by the post_moe_hop hop instead of returning the routed_moe contract's output. "
    "PostMoeHop.enable_routed_input allocates the buffers zeroed in device "
    "memory and the mode is the routed_moe_into_hop numerical variant "
    "(TRTLLM_STAIRCASE_GLM_LL_ROUTED_MOE_INTO_HOP_ACCUMULATION, parsed only "
    "by dispatch.feature_variants), agreed by every rank of the group; a "
    "rank that disagrees on the mode, the buffers or the compile keeps every "
    "rank on the native seam. Modes: row = finalized rows [R,6144] BF16; "
    "identity = token-major raw rows [R*8,6144] BF16 and scales [R,8] "
    "(ROUTED_MOE_UNFINALIZED_LAYOUT_RULE); fp32 = an FP32 accumulator [R,6144] that is "
    "zero when routed_moe's kernel starts, to which routed_moe adds the eight weighted rows "
    "of a token with atomic FP32 adds (order run-to-run nondeterministic), "
    "and whose rows the hop clears after reading them. routed_moe never reuses a "
    "buffer it wrote into; the hop owns their lifetime. PDL edge routed_moe -> post_moe_hop: "
    "routed_moe's expert-cluster kernel executes griddepcontrol.launch_dependents "
    "after each CTA's last FC2 tile, and the post_moe_hop (or deferred_norm) hop reads the "
    "routed-input buffer only after its own griddepcontrol.wait."
)

# Deferred-norm packet (deferred_norm): the packet at the post_moe_hop -> query_front seam.
POST_MOE_HOP_DEFERRED_NORM_RULE = (
    "With deferred_norm selected, the post-MoE hop returns a DeferredNormHidden packet "
    "(deferred_norm/packet.py) in place of the normalized hidden state: a_frag = "
    "bf16(x * gamma) in the mma.m16n8k16 B-fragment layout of packet.py, "
    "sumsq[R,P] FP32 per-warp square sums in post_moe_hop's reduction order, residual "
    "[R,6144] BF16 bit-identical to post_moe_hop's, the next layer's gamma and eps. "
    "The packet is not a tensor: a consumer that does not know it fails at "
    "its first tensor operation, and every consumer other than deferred_norm's "
    "projection calls materialize(), which returns post_moe_hop's normalized output "
    "bit for bit. deferred_norm's projection applies r = rsqrt(sum(sumsq) / H + eps) "
    "after its GEMM. PDL: with trigger_early (the default) the hop executes "
    "griddepcontrol.launch_dependents right after its own "
    "griddepcontrol.wait; the dependent projection reads only immutable "
    "weights before its griddepcontrol.wait and reads the packet after it. "
    "Scope (deferred_norm variant; this target runs shared): all = "
    "every boundary into a target layer; shared = only boundaries into "
    "shared-indexer layers, "
    "decided per layer at hop construction and agreed by every rank, so a "
    "full-indexer layer receives post_moe_hop's normalized tensor."
)

# Shared communication state of the post_moe_hop contract (deferred_norm on the post_moe_hop
# workspace).
POST_MOE_HOP_SHARED_PROTOCOL_RULE = (
    "deferred_norm's hop kernel (deferred_norm/hop_kernel.py) runs on the post_moe_hop hop's workspace and "
    "implements post_moe_hop's protocol a second time: epoch % 3 buffer rotation, "
    "clearing the dirty buffer by the previous call's row count, the "
    "Lamport sentinel poll bounded by post_moe_hop's timeout_ns, the sticky per-buffer "
    "error words and the arrival count; it takes only the state-word indices "
    "from post_moe_hop/kernels.py (deferred_norm/ptx.py). post_moe_hop and deferred_norm hops of one group advance "
    "one epoch and one error chain, so a change to either kernel's protocol "
    "must change both in the same commit and pass the interleaved post_moe_hop/deferred_norm "
    "tests (test_deferred_norm_hop_mpi.py, test_deferred_norm_routed_mpi.py)."
)

CONTRACTS = {
    "query_front": Contract(
        "query/projection packet",
        "absorbed_query",
        ("hidden[R,6144] or DeferredNormHidden (deferred_norm)", "native q/kv-a/indexer weights"),
        (
            "q[R,H,256] or (q_nope_absorbed[R,H,512], q_rope[R,H,64])",
            "native compressed-kv/rope/indexer projection outputs",
        ),
        (POST_MOE_HOP_DEFERRED_NORM_RULE,),
    ),
    "index_selection": Contract(
        "index selection",
        "indexer_topk",
        ("native projected query/key/scale/weights", "immutable cache view", "live lengths"),
        ("topk_ids[R,2048]", "validity"),
    ),
    "sparse_attention": Contract(
        "sparse attention",
        "sparse_mla_decode",
        ("prepared query + scales", "FP8 paged KV", "selection/validity", "causal positions"),
        ("latent_out[R,H,512]",),
    ),
    "attention_output_tail": Contract(
        "attention tail",
        "attention_output_tail",
        ("latent_out[R,H,512]", "W_UV", "local W_o"),
        ("partial[R,6144]",),
        (ATTENTION_TAIL_COMMUNICATION_RULE, ATTENTION_TAIL_PDL_RULE),
    ),
    "router": Contract(
        "pre-MoE hop",
        "router_gemm",
        ("rank-local attention partial", "residual", "norm weight", "router weight"),
        ("normed_hidden", "residual", "router_logits[R,256] or None"),
    ),
    "routed_moe": Contract(
        "routed MoE",
        "routed_moe",
        ("hidden or quantized hidden", "router logits", "native expert weights/scales"),
        (
            "finalized local routed output",
            "or (fc2_output, expert_scale_factor, expanded_idx_to_permuted_idx)",
            "or the post_moe_hop hop's routed-input buffer (routed_moe_into_hop)",
        ),
        (ROUTED_MOE_UNFINALIZED_LAYOUT_RULE, POST_MOE_HOP_ROUTED_INPUT_RULE),
    ),
    "post_moe_hop": Contract(
        "post-MoE hop",
        "post_moe_hop",
        (
            "native raw MoE tuple or its routed-input buffer (routed_moe_into_hop)",
            "shared output or None",
            "residual",
            "norm weight",
        ),
        ("post-reduction hidden or DeferredNormHidden (deferred_norm)", "residual"),
        (
            POST_MOE_HOP_ROUTED_INPUT_RULE,
            POST_MOE_HOP_DEFERRED_NORM_RULE,
            POST_MOE_HOP_SHARED_PROTOCOL_RULE,
        ),
    ),
    "draft_front": Contract(
        "draft front",
        "draft_front",
        ("draft embedding", "previous hidden", "enorm/hnorm weights"),
        ("TP-local packed eh_proj input",),
    ),
}
