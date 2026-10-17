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
"""Real-weight fixtures and the native oracle for the routed-MoE tests.

The fixture is one GLM-5.2 NVFP4 MoE layer as TP rank ``r`` of TP8/EP1 sees
it: every routed expert's gate/up rows ``[256r, 256r+256)`` and down columns
``[256r, 256r+256)``, with block and global scales, the router weight and the
router bias. The native module is the stock TRT-LLM-Gen NVFP4 leaf loaded
through its own ``load_weights``/``post_load_weights``, so its buffers carry
exactly the shuffled/interleaved layouts and derived scalars of a served model.
Slicing the checkpoint before loading a ``tp_size=1`` module is equivalent to
the TP8 loader's ``load_weight_shard`` (COLUMN for w1/w3, ROW for w2).

Hidden-state proxies are real token embeddings passed through the layer's
``post_attention_layernorm`` (labelled semi-synthetic: the served MoE input is
the normalized residual stream of that layer, which needs a full forward).
"""

import json
import os
import tempfile
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from utils.llm_data import llm_models_root

CKPT_DIR = os.environ.get(
    "GLM_MOE_DSA_ROUTED_MOE_CKPT_DIR",
    os.path.join(str(llm_models_root()), "GLM-5.2-NVFP4"),
)
CACHE_DIR = os.environ.get(
    "GLM_MOE_DSA_ROUTED_MOE_FIXTURE_CACHE",
    os.path.join(tempfile.gettempdir(), "glm_moe_dsa_routed_moe_fixtures"),
)

HIDDEN = 6144
INTER = 2048
NUM_EXPERTS = 256
TOP_K = 8
TP_SIZE = 8
ROUTED_SCALING = 2.5
RMS_EPS = 1e-5

# A fixed English/code text; consecutive tokens of it stand in for the
# consecutive verify rows of one MTP request.
PROMPT_TEXT = (
    "def merge_sorted(a, b):\n    i = j = 0\n    out = []\n"
    "    while i < len(a) and j < len(b):\n        if a[i] <= b[j]:\n"
    "            out.append(a[i]); i += 1\n        else:\n"
    "            out.append(b[j]); j += 1\n    return out + a[i:] + b[j:]\n"
    "The quick brown fox jumps over the lazy dog while the compiler "
    "schedules tensor-core instructions across the streaming multiprocessors."
)


def _index() -> Dict[str, str]:
    with open(os.path.join(CKPT_DIR, "model.safetensors.index.json")) as f:
        return json.load(f)["weight_map"]


class _Reader:
    """Opens each safetensors shard once and slices tensors lazily."""

    def __init__(self):
        from safetensors import safe_open

        self._safe_open = safe_open
        self._map = _index()
        self._open = {}

    def _handle(self, name: str):
        shard = self._map[name]
        if shard not in self._open:
            self._open[shard] = self._safe_open(
                os.path.join(CKPT_DIR, shard), framework="pt", device="cpu"
            )
        return self._open[shard]

    def full(self, name: str) -> torch.Tensor:
        return self._handle(name).get_tensor(name)

    def rows(self, name: str, start: int, stop: int) -> torch.Tensor:
        # clone: a row slice may be a view of the whole tensor's storage,
        # which torch.save would otherwise write out in full
        return self._handle(name).get_slice(name)[start:stop].clone()

    def cols(self, name: str, start: int, stop: int) -> torch.Tensor:
        return self._handle(name).get_slice(name)[:, start:stop].clone()


def load_layer_rank(layer: int, tp_rank: int) -> Dict[str, torch.Tensor]:
    """Rank-sliced raw checkpoint tensors of one MoE layer (CPU, cached)."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"glm52_L{layer}_tp{TP_SIZE}r{tp_rank}.pt")
    if os.path.exists(path):
        return torch.load(path, map_location="cpu")
    rd = _Reader()
    inter_rank = INTER // TP_SIZE
    r0, r1 = tp_rank * inter_rank, (tp_rank + 1) * inter_rank
    pre = f"model.layers.{layer}.mlp"
    out: Dict[str, torch.Tensor] = {}
    for e in range(NUM_EXPERTS):
        for proj, key in (("gate_proj", "w1"), ("up_proj", "w3")):
            base = f"{pre}.experts.{e}.{proj}"
            out[f"{e}.{key}.weight"] = rd.rows(f"{base}.weight", r0, r1)
            out[f"{e}.{key}.weight_scale"] = rd.rows(f"{base}.weight_scale", r0, r1)
            out[f"{e}.{key}.weight_scale_2"] = rd.full(f"{base}.weight_scale_2")
            out[f"{e}.{key}.input_scale"] = rd.full(f"{base}.input_scale")
        base = f"{pre}.experts.{e}.down_proj"
        out[f"{e}.w2.weight"] = rd.cols(f"{base}.weight", r0 // 2, r1 // 2)
        out[f"{e}.w2.weight_scale"] = rd.cols(f"{base}.weight_scale", r0 // 16, r1 // 16)
        out[f"{e}.w2.weight_scale_2"] = rd.full(f"{base}.weight_scale_2")
        out[f"{e}.w2.input_scale"] = rd.full(f"{base}.input_scale")
    out["gate.weight"] = rd.full(f"{pre}.gate.weight")
    out["gate.e_score_correction_bias"] = rd.full(f"{pre}.gate.e_score_correction_bias")
    out["post_attention_layernorm.weight"] = rd.full(
        f"model.layers.{layer}.post_attention_layernorm.weight"
    )
    tmp = path + f".tmp{os.getpid()}"
    torch.save(out, tmp)
    os.replace(tmp, path)
    return out


def load_layer_full(layer: int) -> Dict[str, torch.Tensor]:
    """Unsliced checkpoint tensors of one MoE layer (CPU, not cached).

    Keys follow the MoE loader's convention (``{e}.w1.weight`` ...), so the
    TP8 module's own ``load_weights`` does the rank slicing.
    """
    rd = _Reader()
    pre = f"model.layers.{layer}.mlp"
    out: Dict[str, torch.Tensor] = {}
    for e in range(NUM_EXPERTS):
        for proj, key in (("gate_proj", "w1"), ("up_proj", "w3"), ("down_proj", "w2")):
            base = f"{pre}.experts.{e}.{proj}"
            for suffix in ("weight", "weight_scale", "weight_scale_2", "input_scale"):
                out[f"{e}.{key}.{suffix}"] = rd.full(f"{base}.{suffix}")
    out["gate.weight"] = rd.full(f"{pre}.gate.weight")
    out["gate.e_score_correction_bias"] = rd.full(f"{pre}.gate.e_score_correction_bias")
    out["post_attention_layernorm.weight"] = rd.full(
        f"model.layers.{layer}.post_attention_layernorm.weight"
    )
    return out


def _served_model_config(layer: int, tp_rank: int, max_num_tokens: int = 128):
    """ModelConfig of one served GEN rank: TP8/EP1 over two 4-GPU hosts,
    no attention DP, TRTLLM MoE backend, NVFP4 experts, BF16 shared expert.
    """
    from transformers import PretrainedConfig

    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm.mapping import Mapping
    from tensorrt_llm.models.modeling_utils import QuantConfig
    from tensorrt_llm.quantization.mode import QuantAlgo

    quant_config = QuantConfig(
        quant_algo=QuantAlgo.NVFP4,
        kv_cache_quant_algo=QuantAlgo.FP8,
        group_size=16,
        exclude_modules=[f"model.layers.{layer}.mlp.shared_experts*"],
    )
    pretrained = PretrainedConfig(
        hidden_size=HIDDEN,
        n_routed_experts=NUM_EXPERTS,
        num_experts_per_tok=TOP_K,
        moe_intermediate_size=INTER,
        n_shared_experts=1,
        n_group=1,
        topk_group=1,
        routed_scaling_factor=ROUTED_SCALING,
    )
    mapping = Mapping(
        world_size=TP_SIZE,
        rank=tp_rank,
        gpus_per_node=4,
        tp_size=TP_SIZE,
        moe_tp_size=TP_SIZE,
        moe_ep_size=1,
    )
    return ModelConfig(
        pretrained_config=pretrained,
        mapping=mapping,
        quant_config=quant_config,
        moe_backend="TRTLLM",
        max_num_tokens=max_num_tokens,
    )


def _walk_post_load(root: torch.nn.Module) -> None:
    """The model loader's full post-load walk over one module tree."""
    for mod in root.modules():
        hook = getattr(mod, "post_load_weights", None)
        if hook is not None and not getattr(mod, "_weights_removed", False):
            hook()


def _aux_streams():
    from tensorrt_llm._torch.utils import AuxStreamType

    return {t: torch.cuda.Stream() for t in AuxStreamType}


def build_served_experts(layer: int, tp_rank: int, full: Dict[str, torch.Tensor]):
    """The routed experts exactly as ``Deepseekv3MoE`` builds them: ``create_moe``
    with the served TP8 ``ModelConfig`` (a ``ConfigurableMoE`` around the
    resolved TRT-LLM-Gen leaf), loaded from unsliced checkpoint tensors.

    Returns ``(wrapper, bias)`` with the BF16 router bias the gate routes with.
    """
    from tensorrt_llm._torch.moe.fused_moe import create_moe
    from tensorrt_llm._torch.moe.fused_moe.routing import DeepSeekV3MoeRoutingMethod

    bias = full["gate.e_score_correction_bias"].to(torch.bfloat16).cuda()
    routing = DeepSeekV3MoeRoutingMethod(
        top_k=TOP_K,
        n_group=1,
        topk_group=1,
        routed_scaling_factor=ROUTED_SCALING,
        callable_e_score_correction_bias=lambda: bias,
        is_fused=True,
    )
    mc = _served_model_config(layer, tp_rank)
    wrapper = create_moe(
        num_experts=NUM_EXPERTS,
        routing_method=routing,
        hidden_size=HIDDEN,
        intermediate_size=INTER,
        dtype=torch.bfloat16,
        reduce_results=False,
        model_config=mc,
        override_quant_config=mc.quant_config,
        aux_stream_dict=_aux_streams(),
        layer_idx=layer,
    )
    weights = {k: v for k, v in full.items() if k[0].isdigit()}
    wrapper.load_weights([weights])
    _walk_post_load(wrapper)
    wrapper = wrapper.cuda()
    return wrapper, bias


class _NoAllReduce(torch.nn.Module):
    """Stand-in for the cross-rank AllReduce of a single-GPU TP8-shard test;
    the paths exercised here never reduce.
    """

    output_buffer_kind = 0

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, *args, **kwargs):
        raise RuntimeError("cross-rank reduction is not available in a single-GPU test")


def build_deepseek_moe(layer: int, tp_rank: int, full: Dict[str, torch.Tensor]):
    """One served MoE (TP8 rank shard): the built-in ``Deepseekv3MoE`` through its own
    constructor, adopted by the target's ``GlmMoE`` as the target's decoder layer adopts it.

    The NVFP4 experts and the gate carry the real checkpoint tensors; the BF16
    shared expert gets random weights (it is identical in both arms of a
    comparison). The cross-rank AllReduce is replaced by a stub, so only the
    paths that do not reduce (``forward(do_finalize=False)`` and
    ``compute_routed_output``) are usable. The target's routed fast paths run
    only inside a decode step (``dispatch.step(dispatch.Phase.DECODE)``).
    """
    import tensorrt_llm._torch.distributed as trtllm_dist
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.layers import (  # noqa: E501
        GlmMoE,
    )
    from tensorrt_llm._torch.models.modeling_deepseekv3 import Deepseekv3MoE

    mc = _served_model_config(layer, tp_rank)
    saved = trtllm_dist.AllReduce
    trtllm_dist.AllReduce = _NoAllReduce
    try:
        moe = Deepseekv3MoE(
            num_experts=NUM_EXPERTS,
            top_k=TOP_K,
            hidden_size=HIDDEN,
            intermediate_size=INTER,
            shared_expert_intermediate_size=INTER,
            aux_stream_dict=_aux_streams(),
            dtype=torch.bfloat16,
            model_config=mc,
            override_quant_config=mc.quant_config,
            layer_idx=layer,
        )
    finally:
        trtllm_dist.AllReduce = saved
    GlmMoE.adopt(moe, mc, layer)
    moe.gate.load_weights(
        [
            {
                "weight": full["gate.weight"],
                "e_score_correction_bias": full["gate.e_score_correction_bias"],
            }
        ]
    )
    weights = {k: v for k, v in full.items() if k[0].isdigit()}
    moe.experts.load_weights([weights])
    g = torch.Generator().manual_seed(1234)
    for p in moe.shared_experts.parameters():
        if p.dtype.is_floating_point:
            p.data.copy_(torch.randn(p.shape, generator=g) * 0.02)
    _walk_post_load(moe)
    return moe.cuda()


def load_router_bias(layer: int) -> torch.Tensor:
    """BF16 ``e_score_correction_bias`` of one MoE layer.

    BF16 is what the TRT-LLM-Gen backend routes with.
    """
    return (
        _Reader()
        .full(f"model.layers.{layer}.mlp.gate.e_score_correction_bias")
        .to(torch.bfloat16)
        .cuda()
    )


def load_embedding_rows(token_ids: List[int]) -> torch.Tensor:
    """Rows of ``model.embed_tokens.weight`` (BF16, CPU)."""
    rd = _Reader()
    name = "model.embed_tokens.weight"
    sl = rd._handle(name).get_slice(name)
    return torch.stack([sl[t : t + 1][0] for t in token_ids]).contiguous()


def prompt_token_ids(max_tokens: int = 256) -> List[int]:
    cache = os.path.join(CACHE_DIR, "prompt_ids.json")
    if os.path.exists(cache):
        with open(cache) as f:
            return json.load(f)[:max_tokens]
    # The checkpoint config does not validate under the image's transformers
    # AutoConfig, so read the tokenizer file directly.
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(os.path.join(CKPT_DIR, "tokenizer.json"))
    ids = tok.encode(PROMPT_TEXT).ids
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(cache, "w") as f:
        json.dump(ids, f)
    return ids[:max_tokens]


@dataclass
class NativeLayer:
    """A native TRT-LLM-Gen NVFP4 MoE layer plus its router (one TP rank)."""

    module: torch.nn.Module
    gate_weight: torch.Tensor  # [256, 6144] bf16
    bias: torch.Tensor  # [256] bf16, the native TRTLLM gate dtype
    norm_weight: torch.Tensor  # [6144] bf16
    layer: int
    tp_rank: int

    def router_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Native min-latency router GEMM (FP32 logits), as DeepseekV3Gate."""
        return torch.ops.trtllm.dsv3_router_gemm_op(
            x, self.gate_weight.t(), bias=None, out_dtype=torch.float32
        )

    def native_forward(
        self,
        x: torch.Tensor,
        logits: torch.Tensor,
        do_finalize: bool = True,
        bias: Optional[torch.Tensor] = None,
    ):
        """The native region of the routed_moe contract: FP4 quant + routing/FC1/FC2/finalize.

        Mirrors ``TRTLLMGenFp4BlockScaleBase.run_moe`` and
        ``TRTLLMGenNvfp4Base.quantize_input`` argument for argument.
        ``bias`` overrides the router bias (routing studies with another
        layer's logits and bias).
        """
        m = self.module
        x_fp4, x_sf = m.quantize_input(x)
        inter_padded = m.w3_w1_weight.shape[-2] // 2
        outputs = m.op_backend.run_fp4_block_scale_moe(
            logits,
            self.bias if bias is None else bias,
            x_fp4,
            x_sf,
            m.w3_w1_weight,
            m.w3_w1_weight_scale,
            None,
            m.act_alpha,
            m.act_beta,
            m.act_clamp,
            m.w2_weight,
            m.w2_weight_scale,
            None,
            m.fc31_scale_c.data,
            m.fc31_alpha.data,
            m.fc2_alpha.data,
            m.num_slots,
            TOP_K,
            1,
            1,
            inter_padded,
            m.slot_start,
            m.expert_size_per_partition,
            ROUTED_SCALING,
            int(m.routing_method.routing_method_type),
            do_finalize=do_finalize,
            topk_weights=None,
            topk_ids=None,
            valid_hidden_size=m.hidden_size,
            valid_intermediate_size=getattr(
                m.quant_method, "intermediate_size_per_partition_lean", None
            ),
            gated_act_type=0,
            output=None,
            tune_max_num_tokens=m.max_num_tokens,
            use_dp=False,
        )
        if do_finalize:
            return outputs[0] if isinstance(outputs, (list, tuple)) else outputs
        return outputs

    def hidden_proxy(self, token_ids: List[int]) -> torch.Tensor:
        """RMSNorm(embedding) * post_attention_layernorm.weight, BF16 on GPU."""
        emb = load_embedding_rows(token_ids).cuda().float()
        var = emb.pow(2).mean(-1, keepdim=True)
        y = emb * torch.rsqrt(var + RMS_EPS)
        return (y * self.norm_weight.float()).to(torch.bfloat16).contiguous()


def build_native_layer(layer: int = 10, tp_rank: int = 0, max_num_tokens: int = 128) -> NativeLayer:
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.moe.fused_moe.interface import MoEWeightLoadingMode
    from tensorrt_llm._torch.moe.fused_moe.routing import DeepSeekV3MoeRoutingMethod
    from tensorrt_llm._torch.moe.fused_moe.trtllm_gen import trtllm_gen_leaf
    from tensorrt_llm.mapping import Mapping
    from tensorrt_llm.models.modeling_utils import QuantConfig
    from tensorrt_llm.quantization.mode import QuantAlgo

    raw = load_layer_rank(layer, tp_rank)
    bias = raw["gate.e_score_correction_bias"].to(torch.bfloat16).cuda()
    routing = DeepSeekV3MoeRoutingMethod(
        top_k=TOP_K,
        n_group=1,
        topk_group=1,
        routed_scaling_factor=ROUTED_SCALING,
        callable_e_score_correction_bias=lambda: bias,
        is_fused=True,
    )
    quant_config = QuantConfig(
        quant_algo=QuantAlgo.NVFP4, kv_cache_quant_algo=QuantAlgo.FP8, group_size=16
    )
    model_config = ModelConfig(
        quant_config=quant_config,
        mapping=Mapping(world_size=1, tp_size=1, rank=0),
        moe_backend="TRTLLM",
        max_num_tokens=max_num_tokens,
    )
    leaf_cls = trtllm_gen_leaf(QuantAlgo.NVFP4)
    module = leaf_cls(
        routing_method=routing,
        num_experts=NUM_EXPERTS,
        hidden_size=HIDDEN,
        intermediate_size=INTER // TP_SIZE,
        dtype=torch.bfloat16,
        reduce_results=False,
        model_config=model_config,
        init_load_balancer=False,
        weight_loading_mode=MoEWeightLoadingMode.VANILLA,
    )
    weights = {k: v for k, v in raw.items() if k[0].isdigit()}
    module.load_weights([weights])
    module.post_load_weights()
    module = module.cuda()
    return NativeLayer(
        module=module,
        gate_weight=raw["gate.weight"].cuda().contiguous(),
        bias=bias,
        norm_weight=raw["post_attention_layernorm.weight"].cuda(),
        layer=layer,
        tp_rank=tp_rank,
    )


def _sum8(v: torch.Tensor, order: str) -> torch.Tensor:
    """Sum of the 8 top-k values in a fixed FP32 association order."""
    c = [v[:, i] for i in range(TOP_K)]
    if order == "xor_desc":
        # butterfly with offsets 16, 8, 4, 2, 1 over lanes 0..7 (zeros above)
        return ((c[0] + c[4]) + (c[2] + c[6])) + ((c[1] + c[5]) + (c[3] + c[7]))
    if order == "xor_asc":
        return ((c[0] + c[1]) + (c[2] + c[3])) + ((c[4] + c[5]) + (c[6] + c[7]))
    if order == "seq":
        acc = c[0]
        for x in c[1:]:
            acc = acc + x
        return acc
    raise ValueError(order)


ROUTING_SUM_ORDER = "xor_desc"


def routing_reference(logits: torch.Tensor, bias_bf16: torch.Tensor, sum_order: str = None):
    """Bit-faithful PyTorch model of the native SigmoidBias/ScaledSumNormalize.

    Returns ``(ids[R,8] int32 in descending-key order, weights[R,8] bf16)``.
    Selection key: FP32 ``0.5*tanh(0.5*x)+0.5 + float(bias)``; ties go to the
    lower expert index (the native 64-bit key packs ``65535-idx`` below the
    score bits).
    """
    s = 0.5 * torch.tanh(0.5 * logits.float()) + 0.5
    score = s + bias_bf16.float()[None, :]
    # Stable descending sort keeps the lower index first among equal scores.
    order = torch.sort(score, dim=-1, descending=True, stable=True).indices
    ids = order[:, :TOP_K]
    sel = torch.gather(score, 1, ids)
    sig = sel - bias_bf16.float()[ids]
    total = _sum8(sig, sum_order or ROUTING_SUM_ORDER)[:, None]
    w = sig * ROUTED_SCALING / total
    return ids.to(torch.int32), w.to(torch.bfloat16)


def logits_for_expert_sets(
    expert_sets: List[List[int]], bias_bf16: torch.Tensor, seed: int = 0
) -> torch.Tensor:
    """FP32 router logits whose native top-8 for row ``t`` is ``expert_sets[t]``.

    Chosen experts get logits that make ``sigmoid + bias`` clear every
    unchosen expert's maximum possible score by a margin; values within the
    chosen set are distinct so the top-8 order is well defined.
    """
    g = torch.Generator().manual_seed(seed)
    bias = bias_bf16.float().cpu()
    num_rows = len(expert_sets)
    logits = torch.full((num_rows, NUM_EXPERTS), -12.0)
    logits += 0.5 * torch.rand((num_rows, NUM_EXPERTS), generator=g)
    for t, chosen in enumerate(expert_sets):
        assert len(chosen) == TOP_K and len(set(chosen)) == TOP_K
        unchosen = [e for e in range(NUM_EXPERTS) if e not in set(chosen)]
        ceiling = max(bias[e].item() for e in unchosen) + 0.05
        for k, e in enumerate(chosen):
            # need sigmoid(l) + bias[e] > ceiling + small spread
            target = ceiling - bias[e].item() + 0.02 * (TOP_K - k)
            target = min(max(target, 1e-3), 1.0 - 1e-6)
            logit = torch.logit(torch.tensor(target, dtype=torch.float64))
            logits[t, e] = float(logit) + 0.5
    return logits.cuda()


def expert_sets_with_distinct(
    num_rows: int, num_distinct: int, seed: int = 0
) -> Optional[List[List[int]]]:
    """``num_rows`` top-8 sets whose union has exactly ``num_distinct`` experts."""
    if num_distinct < TOP_K or num_distinct > TOP_K * num_rows:
        return None
    if num_distinct > NUM_EXPERTS:
        return None
    g = torch.Generator().manual_seed(seed)
    pool = torch.randperm(NUM_EXPERTS, generator=g)[:num_distinct].tolist()
    sets = []
    # Cover the pool round-robin so every expert is used at least once.
    cursor = 0
    for _ in range(num_rows):
        row = []
        while len(row) < TOP_K:
            e = pool[cursor % num_distinct]
            cursor += 1
            if e not in row:
                row.append(e)
        sets.append(row)
    used = set(e for row in sets for e in row)
    assert len(used) == num_distinct, (num_rows, num_distinct, len(used))
    return sets


# ---------------------------------------------------------------------------
# Independent emulation of the native NVFP4 MoE arithmetic (reference only).
# ---------------------------------------------------------------------------
_E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


def e2m1_lut(device) -> torch.Tensor:
    vals = _E2M1 + [-v for v in _E2M1]
    return torch.tensor(vals, dtype=torch.float64, device=device)


def unpack_fp4(packed: torch.Tensor) -> torch.Tensor:
    """uint8 [..., K/2] -> float64 [..., K]; element 0 in the low nibble."""
    lut = e2m1_lut(packed.device)
    lo = lut[(packed & 0xF).long()]
    hi = lut[(packed >> 4).long()]
    return torch.stack([lo, hi], dim=-1).flatten(-2)


def unswizzle_sf(sf: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    """128x4-interleaved scale bytes -> linear [rows, cols] (uint8)."""
    flat = sf.contiguous().view(torch.uint8).flatten()
    r = torch.arange(rows, device=flat.device)[:, None]
    c = torch.arange(cols, device=flat.device)[None, :]
    idx = ((r // 128) * (cols // 4) + c // 4) * 512 + (r % 32) * 16 + ((r % 128) // 32) * 4 + c % 4
    return flat[idx]


def e4m3_bytes_to_float(b: torch.Tensor) -> torch.Tensor:
    return b.contiguous().view(torch.float8_e4m3fn).to(torch.float64)


def quant_e2m1_rn(v: torch.Tensor) -> torch.Tensor:
    """FP32 -> E2M1 value with round-to-nearest-even and saturation."""
    a = v.abs().double()
    grid = torch.tensor(_E2M1, dtype=torch.float64, device=v.device)
    # nearest grid point; ties go to the even code
    d = (a[..., None] - grid).abs()
    best = d.min(-1, keepdim=True).values
    cand = d == best
    codes = torch.arange(8, device=v.device)
    even = cand & (codes % 2 == 0)
    pick = torch.where(even.any(-1, keepdim=True), even, cand)
    code = (pick.int() * (8 - codes)).argmax(-1)  # lowest qualifying code
    q = grid[code]
    q = torch.where(a >= 6.0, torch.full_like(q, 6.0), q)
    return torch.copysign(q, v.double())


def quant_nvfp4_unit(h: torch.Tensor):
    """NVFP4 quantization with unit global scale, block 16 along the last dim.

    Returns the dequantized values (q * sf) as float64.
    """
    shp = h.shape
    hb = h.reshape(*shp[:-1], shp[-1] // 16, 16).float()
    amax = hb.abs().amax(-1, keepdim=True)
    sf = (amax * (1.0 / 6.0)).to(torch.float8_e4m3fn).to(torch.float32)
    osc = torch.where(amax > 0, 1.0 / sf, torch.zeros_like(sf))
    q = quant_e2m1_rn(hb * osc)
    return (q * sf.double()).reshape(shp)


def stored_row_maps():
    """Stored-row -> (is_gate, j) for w3_w1 and stored-row -> hidden for w2."""
    from tensorrt_llm.quantization.utils.fp4_utils import (
        get_reorder_rows_for_gated_act_gemm_row_indices,
        get_shuffle_matrix_a_row_indices,
    )

    dummy13 = torch.empty(2 * (INTER // TP_SIZE), 1)
    p0 = get_reorder_rows_for_gated_act_gemm_row_indices(dummy13)
    p1 = get_shuffle_matrix_a_row_indices(dummy13, epilogue_tile_m=128)
    orig13 = p0[p1]  # stored row -> row of cat([w3, w1])
    inter = INTER // TP_SIZE
    is_gate = orig13 >= inter
    j13 = torch.where(is_gate, orig13 - inter, orig13)
    dummy2 = torch.empty(HIDDEN, 1)
    orig2 = get_shuffle_matrix_a_row_indices(dummy2, epilogue_tile_m=128)
    return is_gate, j13, orig2


def dequant_expert(module, e: int):
    """Stored-row dequantized FC1/FC2 weights of expert ``e`` (no global)."""
    w13 = unpack_fp4(module.w3_w1_weight[e])
    sf13 = e4m3_bytes_to_float(
        unswizzle_sf(module.w3_w1_weight_scale[e], 2 * (INTER // TP_SIZE), HIDDEN // 16)
    )
    w13 = w13 * sf13.repeat_interleave(16, dim=1)
    w2 = unpack_fp4(module.w2_weight[e])
    sf2 = e4m3_bytes_to_float(
        unswizzle_sf(module.w2_weight_scale[e], HIDDEN, (INTER // TP_SIZE) // 16)
    )
    w2 = w2 * sf2.repeat_interleave(16, dim=1)
    return w13, w2


def _device_row_maps(device):
    is_gate, j13, orig2 = stored_row_maps()
    return is_gate.to(device), j13.to(device), orig2.to(device)


def emulate_expert_row(module, e: int, xd_row: torch.Tensor, maps, cache=None) -> torch.Tensor:
    """FC2 output row (BF16 values as float32) of expert ``e`` for one
    dequantized activation row, with FP64 accumulation and the native
    SwiGLU / NVFP4 / scaling steps.
    """
    is_gate, j13, orig2 = maps
    inter = INTER // TP_SIZE
    if cache is not None and e in cache:
        w13, w2 = cache[e]
    else:
        w13, w2 = dequant_expert(module, e)
        if cache is not None:
            cache[e] = (w13, w2)
    a13 = w13 @ xd_row  # stored-row accumulators
    up = torch.zeros(inter, dtype=torch.float64, device=xd_row.device)
    gate = torch.zeros(inter, dtype=torch.float64, device=xd_row.device)
    up[j13[~is_gate]] = a13[~is_gate]
    gate[j13[is_gate]] = a13[is_gate]
    gr = (gate * float(module.fc31_alpha[e])).float()
    us = (up * float(module.fc31_scale_c[e])).float()
    h = us * (gr * torch.sigmoid(gr))
    hd = quant_nvfp4_unit(h[None, :])[0]
    a2 = w2 @ hd  # stored rows
    y = torch.zeros(HIDDEN, dtype=torch.float64, device=xd_row.device)
    y[orig2] = a2
    return (y * float(module.fc2_alpha[e])).to(torch.bfloat16).float()


def dequant_activations(module, x: torch.Tensor) -> torch.Tensor:
    """The stock ``fp4_quantize`` of ``x``, dequantized to float64."""
    rows = x.shape[0]
    xq, xsf = module.quantize_input(x)
    return unpack_fp4(xq.view(torch.uint8)) * e4m3_bytes_to_float(
        xsf.view(torch.uint8).reshape(rows, HIDDEN // 16)
    ).repeat_interleave(16, dim=1)


def emulate_moe(layer: "NativeLayer", x: torch.Tensor, logits: torch.Tensor):
    """FP64-accumulated emulation of the native region (finalized output).

    Activations are quantized with the stock ``fp4_quantize`` op; weights are
    the stock module buffers; routing is ``routing_reference``.
    """
    m = layer.module
    rows = x.shape[0]
    xd = dequant_activations(m, x)
    ids, wts = routing_reference(logits, layer.bias)
    maps = _device_row_maps(x.device)
    out = torch.zeros(rows, HIDDEN, dtype=torch.float32, device=x.device)
    cache = {}
    for t in range(rows):
        acc = torch.zeros(HIDDEN, dtype=torch.float32, device=x.device)
        for k in range(TOP_K):
            y = emulate_expert_row(m, int(ids[t, k]), xd[t], maps, cache)
            acc = acc + wts[t, k].float() * y
        out[t] = acc
    return out.to(torch.bfloat16)


def error_stats(a: torch.Tensor, b: torch.Tensor) -> dict:
    a = a.float()
    b = b.float()
    diff = a - b
    ref_rms = b.pow(2).mean().sqrt().item()
    return {
        "nrms": (diff.pow(2).mean().sqrt().item() / max(ref_rms, 1e-30)),
        "max_abs": diff.abs().max().item(),
        "ref_rms": ref_rms,
        "cos": torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0).item(),
    }


# ---------------------------------------------------------------------------
# Native capture of real routed-MoE inputs (read-only shared fixture).
# ---------------------------------------------------------------------------
REAL_DIR = os.environ.get("GLM_MOE_DSA_REAL_MOE_INPUTS_DIR", "")


def load_real_capture(layer: int, real_dir: str = REAL_DIR):
    """Per prompt, ``(hidden [T, 6144] bf16 or None, logits [T, 256] fp32)``
    that the native ``compute_routed_output`` of ``layer`` received (CPU).
    Hidden states were captured for layers 10, 30, 50 and 77 only.
    """
    import glob

    out = []
    for path in sorted(glob.glob(os.path.join(real_dir, f"rows*_layer{layer}.pt"))):
        obj = torch.load(path, map_location="cpu")
        out.append((obj.get("hidden"), obj["logits"]))
    return out


def real_windows(capture, rows: int, stride: Optional[int] = None):
    """Consecutive-token windows ``(x, logits)`` (CPU); ``x`` is None when the
    capture has no hidden states for this layer.
    """
    stride = stride or rows
    win = []
    for hidden, logits in capture:
        for start in range(0, logits.shape[0] - rows + 1, stride):
            x = None if hidden is None else hidden[start : start + rows]
            win.append((x, logits[start : start + rows]))
    return win


def routing_keys(logits: torch.Tensor, bias_bf16: torch.Tensor) -> torch.Tensor:
    """FP32 selection keys ``sigmoid + bias`` of the native routing."""
    s = 0.5 * torch.tanh(0.5 * logits.float()) + 0.5
    return s + bias_bf16.float()[None, :].to(s.device)


def boundary_gap(logits: torch.Tensor, bias_bf16: torch.Tensor) -> torch.Tensor:
    """Per row, key(8th) - key(9th) in FP32 (0 means an exact boundary tie)."""
    k = routing_keys(logits, bias_bf16)
    top = torch.topk(k, TOP_K + 1, dim=-1).values
    return top[:, TOP_K - 1] - top[:, TOP_K]


def _key_of(logit: torch.Tensor, bias_val: torch.Tensor) -> torch.Tensor:
    s = 0.5 * torch.tanh(0.5 * logit.float()) + 0.5
    return s + bias_val.float()


def logits_for_key(target_key: float, bias_val: torch.Tensor, offset_ulps: int = 0):
    """All FP32 logits within +-2^16 logit ulps of the analytic solution whose
    native key equals ``target_key`` shifted by ``offset_ulps`` key ulps (the
    key arithmetic is FP32, evaluated on the device). May be empty.
    """
    dev = bias_val.device
    want = torch.tensor(target_key, dtype=torch.float32, device=dev)
    toward = torch.tensor(float("inf") if offset_ulps > 0 else -float("inf"), device=dev)
    for _ in range(abs(offset_ulps)):
        want = torch.nextafter(want, toward)
    s_target = float(want.double() - bias_val.double())
    s_target = min(max(s_target, 1e-4), 1 - 1e-4)
    l0 = torch.logit(torch.tensor(s_target, dtype=torch.float64)).float().reshape(1).to(dev)
    steps = torch.arange(-(1 << 16), 1 << 16, device=dev, dtype=torch.int32)
    cand = (l0.view(torch.int32) + steps).view(torch.float32)
    keys = _key_of(cand, bias_val)
    return cand[keys == want]


def logit_for_key(target_key: float, bias_val: torch.Tensor, offset_ulps: int = 0, avoid=None):
    """One logit from ``logits_for_key`` (the one farthest from ``avoid`` when
    given, else the middle hit) or ``None``.
    """
    hits = logits_for_key(target_key, bias_val, offset_ulps)
    if hits.numel() == 0:
        return None
    if avoid is None:
        return hits[hits.numel() // 2].item()
    return hits[(hits - avoid).abs().argmax()].item()


def raw_rows_by_topk(raw) -> torch.Tensor:
    """``[R, 8, 6144]`` FC2 rows in top-k order from a raw tuple
    ``(fc2_output, scales, expanded_idx_to_permuted_idx)``.
    """
    fc2, _, idx = raw
    return fc2[idx.long()]


def finalize_raw(raw) -> torch.Tensor:
    """The stock finalize arithmetic: ``sum_k scale[t,k] * fc2[idx[t,k]]`` in
    FP32, top-k order, rounded to BF16.
    """
    fc2, scale, idx = raw
    rows = scale.shape[0]
    fin = torch.zeros(rows, fc2.shape[1], device=fc2.device, dtype=torch.float32)
    for k in range(TOP_K):
        fin += scale[:, k : k + 1].float() * fc2[idx[:, k].long()].float()
    return fin.to(torch.bfloat16)


# ---------------------------------------------------------------------------
# routed_moe-versus-native classification.
#
# Discrete invariants are exact: the expert scales of every (t, k) are equal
# bit for bit and every (t, k) uses the same expert as native. Expert identity
# is read from the FC2 rows: rows within 5 % relative L2 of native's row are
# the same expert (rows of two different experts are far apart); any other
# window is decoded exactly by ``ExpertDecoder`` (a probe activation shared by
# all tokens turns every FC2 row into a per-expert fingerprint).
# Floating-point results fall in one of three classes:
#   exact    finalized outputs bitwise equal;
#   reorder  every FC2 row bitwise equal to native's and the finalized output
#            within the bound of the same eight exact products summed in a
#            different FP32 order (routed_moe sums per expert slot and across split
#            partials, the native finalize in top-k order):
#            |a - b| <= 2^-20 * sum_k |w_k y_k| + ulp_bf16(max(|a|, |b|));
#   budget   some FC2 rows differ (an FP4 rounding of the intermediate flipped
#            because the split-K FC1 accumulation is associated differently);
#            checked against the frozen error budget
#            (nrms <= 1.5 * nrms(emul - native) + 2e-3, no row above 4x).
# ---------------------------------------------------------------------------
SAME_EXPERT_REL = 0.05


def bf16_ulp(v: torch.Tensor) -> torch.Tensor:
    _, e = torch.frexp(v.float().abs())
    return torch.ldexp(torch.ones_like(v, dtype=torch.float32), e - 8)


def raw_report(raw_a, raw_n) -> dict:
    ra = raw_rows_by_topk(raw_a).float()
    rn = raw_rows_by_topk(raw_n).float()
    num = (ra - rn).pow(2).sum(-1).sqrt()
    den = rn.pow(2).sum(-1).sqrt()
    rel = torch.where(den > 0, num / den.clamp_min(1e-30), (num > 0).float() * 1e30)
    rows_exact = (ra == rn).all(-1)
    return {
        "scale_exact": bool(torch.equal(raw_a[1], raw_n[1])),
        "same_expert": bool((rel < SAME_EXPERT_REL).all()),
        "rows_exact": int(rows_exact.sum()),
        "rows": rows_exact.numel(),
        "max_row_rel": float(rel.max()),
    }


def reorder_close(fin_a: torch.Tensor, fin_n: torch.Tensor, raw_n) -> bool:
    fc2, scale, idx = raw_n
    terms = (scale.float()[:, :, None] * fc2[idx.long()].float()).abs().sum(1)
    a = fin_a.float()
    b = fin_n.float()
    bound = terms * 2.0**-20 + bf16_ulp(torch.maximum(a.abs(), b.abs()))
    return bool(((a - b).abs() <= bound).all())


def budget_check(out: torch.Tensor, native: torch.Tensor, emul: torch.Tensor):
    """The frozen error budget; returns ``(ok, stats)``."""
    e_native = error_stats(out, native)
    e_emul = error_stats(emul, native)
    budget = 1.5 * e_emul["nrms"] + 2e-3
    row_ref = native.float().pow(2).mean(-1).sqrt()
    row_err = (out.float() - native.float()).pow(2).mean(-1).sqrt()
    ok = e_native["nrms"] <= budget and bool(torch.all(row_err <= 4 * budget * row_ref + 1e-6))
    return ok, {"routed_moe_vs_native": e_native, "emul_vs_native": e_emul, "budget": budget}


class ExpertDecoder:
    """Exact expert IDs of a raw tuple, routed_moe's or native's.

    With every token row equal to one probe activation, the FC2 row of
    ``(t, k)`` depends only on the expert, so it is matched (nearest
    neighbour, L2) against per-expert fingerprints: native FC2 rows of every
    local expert for the probe, produced with constructed logits whose
    expert of every ``(t, k)`` is known. Routing comes from the logits and
    bias only, so a decoded call uses the window's logits with the probe as
    activation.
    """

    def __init__(self, layer: "NativeLayer", probe: torch.Tensor):
        self.layer = layer
        self.probe = probe.reshape(1, HIDDEN).to(torch.bfloat16).cuda()
        # Fingerprints: every expert's FC2 row for the probe, computed by the
        # FP64 emulation of the native arithmetic from the native buffers
        # (independent of any routing). A native or routed_moe row of the same
        # expert differs from it at most by an FP4 rounding of the
        # intermediate; rows of different experts are far apart.
        m = layer.module
        xd = dequant_activations(m, self.probe)[0]
        maps = _device_row_maps(xd.device)
        self.fps = torch.stack(
            [emulate_expert_row(m, e, xd, maps) for e in range(NUM_EXPERTS)]
        ).float()
        norms = self.fps.norm(dim=-1)
        d = torch.cdist(self.fps, self.fps) + torch.eye(NUM_EXPERTS, device="cuda") * 1e30
        self.min_norm_ratio = float(norms.min() / norms.median())
        self.min_separation = float(d.min() / norms.median())
        assert self.min_norm_ratio > 0.05 and self.min_separation > 0.2, (
            self.min_norm_ratio,
            self.min_separation,
        )

    def _x(self, rows: int) -> torch.Tensor:
        return self.probe.expand(rows, HIDDEN).contiguous()

    def decode(self, raw):
        rows = raw_rows_by_topk(raw).float()
        flat = rows.reshape(-1, HIDDEN)
        d = torch.cdist(flat, self.fps)
        best = d.topk(2, dim=-1, largest=False)
        margin = best.values[:, 1] / best.values[:, 0].clamp_min(1e-30)
        return best.indices[:, 0].reshape(rows.shape[:2]), float(margin.min())

    def compare(self, runner, logits, bias=None) -> dict:
        """Decode native's and routed_moe's experts for ``logits`` (and ``bias``)."""
        x = self._x(logits.shape[0])
        n_raw = self.layer.native_forward(x, logits, do_finalize=False, bias=bias)
        w_raw = runner(x, logits, do_finalize=False)
        n_ids, n_margin = self.decode(n_raw)
        w_ids, w_margin = self.decode(w_raw)
        return {
            "ids_equal": bool(torch.equal(n_ids, w_ids)),
            "scale_exact": bool(torch.equal(n_raw[1], w_raw[1])),
            "margin": min(n_margin, w_margin),
            "native_ids": n_ids,
            "routed_moe_ids": w_ids,
        }


def classify_vs_native(w_fin, w_raw, n_fin, n_raw, emul_fn=None, label="", id_check=None):
    """Assert the discrete invariants and return ``(class, report)``.

    ``emul_fn`` computes the FP64 emulation for the budget class; without it
    a window of that class is returned as ``"budget-unchecked"``.
    ``id_check`` (e.g. ``lambda: decoder.compare(runner, logits)``) decodes
    the experts exactly when some FC2 row is not within 5 % of native's.
    """
    rep = raw_report(w_raw, n_raw)
    assert rep["scale_exact"], f"{label}: expert scales differ from native"
    if not rep["same_expert"]:
        assert id_check is not None, f"{label}: FC2 rows far from native and no id check {rep}"
        dec = id_check()
        rep["decoded"] = {k: dec[k] for k in ("ids_equal", "scale_exact", "margin")}
        assert dec["ids_equal"] and dec["scale_exact"] and dec["margin"] > 4, (
            f"{label}: decoded experts differ from native {rep}"
        )
    if torch.equal(w_fin, n_fin):
        return "exact", rep
    if rep["rows_exact"] == rep["rows"]:
        assert reorder_close(w_fin, n_fin, n_raw), f"{label}: beyond the FP32 reorder bound"
        return "reorder", rep
    if emul_fn is None:
        return "budget-unchecked", rep
    ok, stats = budget_check(w_fin, n_fin, emul_fn())
    rep["budget"] = stats
    assert ok, f"{label}: outside the frozen error budget {stats}"
    return "budget", rep


def check_call(layer, runner, x, logits, fin=None, raw=None, emul=True, label="", decoder=None):
    """Run routed_moe (unless outputs are given) and native on one input; classify."""
    if fin is None:
        fin = runner(x, logits)
    if raw is None:
        raw = runner(x, logits, do_finalize=False)
    n_fin = layer.native_forward(x, logits)
    n_raw = layer.native_forward(x, logits, do_finalize=False)
    emul_fn = (lambda: emulate_moe(layer, x, logits)) if emul else None
    id_check = (lambda: decoder.compare(runner, logits)) if decoder is not None else None
    return classify_vs_native(fin, raw, n_fin, n_raw, emul_fn, label, id_check)
