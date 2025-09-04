# for now, this is only used as a benchmark and not a correctness test

import argparse
import json
import os
import pickle
import shutil
import sys
import tempfile
import traceback
import weakref
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Optional

import cloudpickle
import torch
from mpi4py import MPI
from mpi4py.futures import MPIPoolExecutor
from transformers import PretrainedConfig

import tensorrt_llm
from tensorrt_llm._torch.attention_backend.interface import KVCacheParams
from tensorrt_llm._torch.attention_backend.utils import get_attention_backend
from tensorrt_llm._torch.auto_deploy.utils.quantization_utils import \
    fp4_global_scale
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_deepseekv3 import \
    DeepseekV3DecoderLayer
from tensorrt_llm._torch.modules.linear import Linear, WeightMode
from tensorrt_llm._torch.modules.multi_stream_utils import with_multi_stream
from tensorrt_llm._torch.modules.rms_norm import RMSNorm
from tensorrt_llm._torch.pyexecutor.llm_request import (LlmRequest,
                                                        LlmRequestState,
                                                        SamplingConfig)
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm._torch.utils import AuxStreamType, model_extra_attrs
from tensorrt_llm._utils import str_dtype_to_binding, torch_dtype_to_str
from tensorrt_llm.bindings.executor import KvCacheConfig
from tensorrt_llm.mapping import CpType, Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo
from tensorrt_llm.sampling_params import SamplingParams

cloudpickle.register_pickle_by_value(sys.modules[__name__])
MPI.pickle.__init__(
    cloudpickle.dumps,
    cloudpickle.loads,
    pickle.HIGHEST_PROTOCOL,
)


def llm_models_root(check=False) -> Optional[Path]:
    root = Path("/home/scratch.trt_llm_data/llm-models/")

    if "LLM_MODELS_ROOT" in os.environ:
        root = Path(os.environ.get("LLM_MODELS_ROOT"))

    if not root.exists():
        root = Path("/scratch.trt_llm_data/llm-models/")

    if check:
        assert root.exists(), \
        "You shall set LLM_MODELS_ROOT env or be able to access /home/scratch.trt_llm_data to run this test"

    return root if root.exists() else None


def apply_exclude_modules(layer: DeepseekV3DecoderLayer,
                          quant_config: Optional[QuantConfig]):
    if quant_config is None or quant_config.exclude_modules is None:
        return
    for ie, exclude_module in enumerate(quant_config.exclude_modules):
        if exclude_module.startswith("model.layers."):
            i = exclude_module.find(".", len("model.layers."))
            quant_config.exclude_modules[ie] = exclude_module[i + 1:]
    new_config = QuantConfig(
        kv_cache_quant_algo=quant_config.kv_cache_quant_algo)

    for name, module in layer.named_modules():
        candidates = [name]
        if isinstance(module, Linear):
            weight_mode = module.weights_loading_config.weight_mode
            if weight_mode == WeightMode.FUSED_GATE_UP_LINEAR:
                # sometimes gate and up proj are not packed in the checkpoint,
                # but they still share the same exclusion rule
                candidates += [
                    name.replace('gate_up_proj', 'gate_proj'),
                    name.replace('gate_up_proj', 'up_proj')
                ]
            elif weight_mode == WeightMode.FUSED_QKV_LINEAR:
                # sometimes q_proj, k_proj and v_proj are not packed in the checkpoint,
                # but they still share the same exclusion rule
                candidates += [
                    name.replace('qkv_proj', 'q_proj'),
                    name.replace('qkv_proj', 'k_proj'),
                    name.replace('qkv_proj', 'v_proj')
                ]
        is_excluded = any(
            quant_config.is_module_excluded_from_quantization(n)
            for n in candidates)
        if is_excluded and getattr(module, "quant_config", None) is not None:
            module.quant_config = new_config
        if callable(getattr(module, "create_weights", None)):
            module.create_weights()


# set this to ensure we get DS Fp8
QUANTIZATION_CONFIG_FP8 = {
    "activation_scheme": "dynamic",
    "fmt": "e4m3",
    "quant_method": "fp8",
    "weight_block_size": [128, 128]
}
# set this to None / "nvfp4" / "dsfp8"
QUANTIZATION_TYPE = "nvfp4"
FP4_QUANT_CONFIG_PATH = llm_models_root(
) / "DeepSeek-R1" / "DeepSeek-R1-FP4" / "hf_quant_config.json"

NUM_HIDDEN_LAYERS = 61


# values for deepseek_v3_lite
@dataclass(kw_only=True, frozen=True)
class Scenario:
    # model config parameters
    dtype: torch.dtype = torch.bfloat16
    kv_cache_dtype: torch.dtype = torch.bfloat16
    num_layers: int = 1
    num_heads: int = 32
    num_kv_heads: int = 32
    q_lora_rank: int = None
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    hidden_size: int = 2560
    rope_theta: float = 10000.0
    rope_scaling: bool = False
    rope_beta_fast: int = 32
    rope_beta_slow: int = 1
    rope_factor: float = 40.0
    rope_mscale: float = 1.0
    rope_mscale_all_dim: float = 1.0
    rope_original_max_position_embeddings: int = 4096
    rope_type: str = "yarn"
    model_type: str = "deepseek_v3"
    intermediate_size: int = 12288
    moe_intermediate_size: int = 1536
    n_routed_experts: int = 72
    n_shared_experts: int = 2
    num_experts_per_tok: int = 6
    # set this to 1 to get a dense layer instead of MoE
    first_k_dense_replace: int = 0
    moe_layer_freq: int = 1
    n_group: int = 1
    topk_group: int = 1
    routed_scaling_factor: float = 2.0
    rms_norm_eps: float = 1e-6
    vocab_size: int = 129280
    bias: bool = False
    num_hidden_layers: int = 1
    # test setup parameters
    kv_cache_tokens_per_block: int = 32
    # TODO only 1 is supported for now here
    predicted_tokens_per_seq: int = 1
    batch: int = 8
    ctx_len: int = 1024
    # note: this is essentially the warm-up steps, as there is no correctness check for now
    ref_steps: int = 1
    tp_size: int = 1
    kvp_size: int = 4
    ep_size: int = 4

    @property
    def max_position_embeddings(self) -> int:
        # ensure that max_position_embeddings is set large enough for every scenario
        return self.ctx_len + 1

    @property
    def world_size(self) -> int:
        return self.tp_size * self.kvp_size


@dataclass(kw_only=True, frozen=True)
class ScenarioV3(Scenario):
    num_heads: int = 128
    num_kv_heads: int = 128
    q_lora_rank: int = 1536
    hidden_size: int = 7168
    # this enables RoPE scaling with all the right factors (from defaults)
    rope_scaling: bool = True
    intermediate_size: int = 18432
    moe_intermediate_size: int = 2048
    n_routed_experts: int = 256
    n_shared_experts: int = 1
    num_experts_per_tok: int = 8
    n_group: int = 8
    topk_group: int = 4
    routed_scaling_factor: float = 2.5
    rms_norm_eps: float = 1e-6


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run DeepSeek V3 Helix distributed benchmarks')
    parser.add_argument('--type',
                        type=str,
                        default='v3',
                        choices=['v3', 'lite'],
                        help='Model type: v3 or lite (default: v3)')
    parser.add_argument('--ctx_len_start',
                        type=int,
                        default=2**16,
                        help='Starting context length (default: 65536)')
    parser.add_argument('--ctx_len_end',
                        type=int,
                        default=2**22,
                        help='Ending context length (default: 4194304)')
    parser.add_argument('--batch',
                        type=int,
                        default=1,
                        help='Batch size (default: 1)')
    parser.add_argument('--tp',
                        type=int,
                        default=8,
                        help='Tensor parallelism size (default: 8)')
    parser.add_argument('--kvp',
                        type=int,
                        default=1,
                        help='KV parallelism size (default: 1)')
    parser.add_argument('--ep',
                        type=int,
                        default=2,
                        help='Expert parallelism size (default: 2)')
    parser.add_argument('--dense',
                        action='store_true',
                        help='Use dense layer instead of MoE layer')
    return parser.parse_args()


def generate_scenarios(args):
    """Generate scenarios based on command line arguments."""
    scenarios = []
    scenario_class = ScenarioV3 if args.type == 'v3' else Scenario
    first_k_dense_replace = 1 if args.dense else 0

    # Generate scenarios by doubling ctx_len from start to end
    ctx_len = args.ctx_len_start
    while ctx_len <= args.ctx_len_end:
        scenarios.append(
            scenario_class(batch=args.batch,
                           ctx_len=ctx_len,
                           tp_size=args.tp,
                           kvp_size=args.kvp,
                           ep_size=args.ep,
                           first_k_dense_replace=first_k_dense_replace))
        ctx_len *= 2

    return scenarios


def _setup_kv(scenario: Scenario, mapping: Mapping, gen_steps: int):
    # Set up KVCacheManager and attn_metadata for MLA
    n_gpu = mapping.cp_size
    assert scenario.ctx_len % n_gpu == 0
    ctx_len_per_gpu = scenario.ctx_len // n_gpu
    max_tokens = (
        ctx_len_per_gpu + gen_steps + scenario.kv_cache_tokens_per_block - 1
    ) // scenario.kv_cache_tokens_per_block * scenario.kv_cache_tokens_per_block * scenario.batch
    kv_cache_manager = KVCacheManager(
        KvCacheConfig(
            max_tokens=max_tokens,
            enable_block_reuse=False,
        ),
        # Use SELFKONLY for MLA
        tensorrt_llm.bindings.internal.batch_manager.CacheType.SELFKONLY,
        num_layers=scenario.num_layers,
        num_kv_heads=1,
        head_dim=scenario.kv_lora_rank + scenario.qk_rope_head_dim,
        tokens_per_block=scenario.
        kv_cache_tokens_per_block,  # for test, just use seq_len
        max_seq_len=ctx_len_per_gpu + gen_steps,
        max_batch_size=scenario.batch,
        mapping=mapping,
        dtype=str_dtype_to_binding(torch_dtype_to_str(scenario.kv_cache_dtype)),
    )
    for req_id in range(scenario.batch):
        req = LlmRequest(
            request_id=req_id,
            max_new_tokens=1,
            input_tokens=[1] *
            ctx_len_per_gpu,  # all requests have the same length here
            sampling_config=SamplingConfig(
                SamplingParams()._get_sampling_config()),
            is_streaming=False,
        )
        req.is_dummy_request = True
        req.paged_kv_block_ids = []
        beam_width = 1
        kv_cache_manager.impl.add_sequence(req_id, ctx_len_per_gpu, beam_width,
                                           req)
        req.state = LlmRequestState.GENERATION_IN_PROGRESS
        req.prompt_len = ctx_len_per_gpu
        req.py_prompt_len = req.prompt_len
    return kv_cache_manager


def _generate_random_weights(layer: DeepseekV3DecoderLayer):
    quant_dict = dict()

    # Helpers to init a tensor
    def init_low_precision(t, op, t_hash, no_fp4):
        if t.dtype.itemsize <= 1 and QUANTIZATION_TYPE == "nvfp4" and not no_fp4:
            t2 = torch.empty(t.shape[:-1] + (t.shape[-1] * 2, ),
                             dtype=torch.float32,
                             device=t.device)
            op(t2)
            t3 = t2.to(dtype=torch.float16)
            t4, x_scale = torch.ops.trtllm.fp4_quantize(t3,
                                                        fp4_global_scale(t2),
                                                        16, False)
            quant_dict[t_hash] = x_scale
            t.copy_(t4)
        elif t.dtype == torch.uint8:
            # sanity check for quantized weights (in uint8)
            raise ValueError("uint8 is not supported for non-fp4 weights")
        elif t.dtype.itemsize <= 1:
            t2 = torch.empty_like(t, dtype=torch.float32)
            op(t2)
            t.copy_(t2)
        else:
            op(t)

    def init_uniform(tensor, a=-1.0, b=1.0, use_kaiming=False, no_fp4=False):
        if tensor is not None:
            if use_kaiming:
                tv = tensor.view(-1, tensor.shape[-2], tensor.shape[-1])
                for t in tv:
                    init_low_precision(t, torch.nn.init.kaiming_uniform_,
                                       hash(tensor), no_fp4)
            else:
                init_low_precision(tensor,
                                   partial(torch.nn.init.uniform_, a=a, b=b),
                                   hash(tensor), no_fp4)

    def init_block_scale_fp8(tensor, orig_tensor, _no_fp4):
        if tensor is None or orig_tensor is None:
            return
        b1, b2 = 128, 128
        orig_tensor = orig_tensor.contiguous().to(tensor.dtype)
        exp1 = (orig_tensor.shape[-2] + b1 - 1) // b1
        exp2 = (orig_tensor.shape[-1] + b2 - 1) // b2
        if tensor.shape[-2] != exp1 or tensor.shape[-1] != exp2:
            # for some fused weights, this can happen
            # we simply adapt the size of the blocks and use that for the scale
            b1 = (orig_tensor.shape[-2] + tensor.shape[-2] -
                  1) // tensor.shape[-2]
            b2 = (orig_tensor.shape[-1] + tensor.shape[-1] -
                  1) // tensor.shape[-1]
        e1 = orig_tensor.shape[-2] // b1
        e2 = orig_tensor.shape[-1] // b2
        x = orig_tensor[..., :e1 * b1, :e2 * b2].view(*orig_tensor.shape[:-2],
                                                      e1, b1, e2, b2)
        scale = x.abs().amax(dim=(-3, -1)) / 448.
        if e1 * b1 != orig_tensor.shape[-2]:
            x2 = orig_tensor[...,
                             e1 * b1:, :e2 * b2].view(*orig_tensor.shape[:-2],
                                                      1, -1, e2, b2)
            scale2 = x2.abs().amax(dim=(-3, -1)) / 448.
            scale = torch.cat([scale, scale2], dim=-2)
        if e2 * b2 != orig_tensor.shape[-1]:
            x3 = orig_tensor[..., :e1 * b1,
                             e2 * b2:].view(*orig_tensor.shape[:-2], e1, b1, 1,
                                            -1)
            scale3 = x3.abs().amax(dim=(-3, -1)) / 448.
            if scale.shape[-2] == e1 + 1:
                x4 = orig_tensor[..., e1 * b1:,
                                 e2 * b2:].view(*orig_tensor.shape[:-2], 1, -1,
                                                1, -1)
                scale4 = x4.abs().amax(dim=(-3, -1)) / 448.
                scale3 = torch.cat([scale3, scale4], dim=-2)
            scale = torch.cat([scale, scale3], dim=-1)
        tensor.copy_(scale)

    def init_block_scale_fp4(tensor, orig_tensor, no_fp4):
        if no_fp4:
            init_block_scale_fp8(tensor, orig_tensor, None)
            return
        if tensor is None or orig_tensor is None:
            return
        if hash(orig_tensor) in quant_dict:
            tensor.copy_(quant_dict[hash(orig_tensor)])
            return
        print_dict = {k: (v.shape, v.dtype) for k, v in quant_dict.items()}
        raise ValueError(
            "init_block_scale_fp4 must be called after init_uniform "
            f"for original tensor, {hash(orig_tensor)} not in {print_dict}")

    if QUANTIZATION_TYPE == "nvfp4":
        init_block_scale = init_block_scale_fp4
    elif QUANTIZATION_TYPE == "dsfp8":
        init_block_scale = init_block_scale_fp8

    def init_linear(mod, no_fp4=False):
        if mod is None:
            return
        init_uniform(mod.weight, use_kaiming=True, no_fp4=no_fp4)
        if hasattr(mod, "weight_scale"):
            init_block_scale(mod.weight_scale, mod.weight, no_fp4=no_fp4)
        if hasattr(mod, "bias"):
            init_uniform(mod.bias, no_fp4=True)

    mla = layer.self_attn
    # Linear modules
    for name in ["kv_a_proj_with_mqa", "kv_b_proj", "o_proj", "q_b_proj"]:
        init_linear(getattr(mla, name))

    # RMSNorm modules
    for name in ["kv_a_layernorm", "q_a_layernorm"]:
        if name == "q_a_layernorm":
            mod = getattr(mla, name, None)
        else:
            mod = getattr(mla, name)
        if mod is not None:
            init_uniform(mod.weight, a=0.9, b=1.1, no_fp4=True)

    # k_b_proj_trans (created in create_weights)
    init_uniform(mla.k_b_proj_trans, use_kaiming=True, no_fp4=True)
    # k_b_proj_trans_scale (optional)
    if hasattr(mla, "k_b_proj_trans_scale"):
        init_block_scale(mla.k_b_proj_trans_scale,
                         mla.k_b_proj_trans,
                         no_fp4=True)
    init_uniform(mla.v_b_proj, no_fp4=True)
    # v_b_proj_scale (optional)
    if hasattr(mla, "v_b_proj_scale"):
        init_block_scale(mla.v_b_proj_scale, mla.v_b_proj, no_fp4=True)

    # Initialize the MoE weights / all remaining parameters from decoder layer
    # Initialize input and post-attention layer norms
    for name in ["input_layernorm", "post_attention_layernorm"]:
        mod = getattr(layer, name, None)
        if mod is not None and hasattr(mod, "weight"):
            init_uniform(mod.weight, a=0.9, b=1.1, no_fp4=True)

    # Initialize MLP weights
    if hasattr(layer, "mlp"):
        mlp = layer.mlp
        # Handle both GatedMLP and Deepseekv3MoE
        if hasattr(mlp, "gate_up_proj"):
            # GatedMLP case
            init_linear(mlp.gate_up_proj)
            init_linear(mlp.down_proj)
        elif hasattr(mlp, "gate"):
            # Deepseekv3MoE case
            # Initialize gate weights
            init_uniform(mlp.gate.weight, use_kaiming=True, no_fp4=True)
            if hasattr(mlp.gate, "e_score_correction_bias"):
                init_uniform(mlp.gate.e_score_correction_bias,
                             a=1.5,
                             b=2.5,
                             no_fp4=True)

            # Initialize experts weights
            if hasattr(mlp, "experts"):
                experts = mlp.experts
                init_uniform(experts.w3_w1_weight)
                init_uniform(experts.w2_weight)
                for name in [
                        "fc31_dequant",
                        "fc2_dequant",
                        "fc2_quant",
                        "fc31_input_dequant",
                        "w3_w1_weight_scaling_factor",
                        "w2_weight_scaling_factor",
                        "fc31_act_scale",
                        "fc2_act_scale",
                        "fc31_weight_scale",
                        "fc2_weight_scale",
                        "fc31_alpha",
                        "fc2_alpha",
                        "w3_w1_weight_scale",
                        "w2_weight_scale",
                        "fc31_input_scale",
                        "fc2_input_scale",
                        "fc31_scale_c",
                ]:
                    if hasattr(experts, name):
                        if "weight_scal" in name and (
                                "w3" in name or "fc31" in name) and (
                                    not hasattr(experts, "scaling_vector_size")
                                    or experts.scaling_vector_size == 128):
                            init_block_scale(getattr(experts, name),
                                             experts.w3_w1_weight)
                        elif "weight_scale" in name and (
                                "w2" in name or "fc2" in name) and (
                                    not hasattr(experts, "scaling_vector_size")
                                    or experts.scaling_vector_size == 128):
                            init_block_scale(getattr(experts, name),
                                             experts.w2_weight)
                        else:
                            init_uniform(getattr(experts, name), no_fp4=True)

            # Initialize shared experts weights
            if hasattr(mlp, "shared_experts"):
                shared_experts = mlp.shared_experts
                init_linear(shared_experts.gate_up_proj)
                init_linear(shared_experts.down_proj)


@torch.inference_mode
def _run_ds_layer_distributed(rank: int, world_size: int, scenario: Scenario,
                              gen_steps: int):
    torch.manual_seed(42 + rank)
    input_gen = torch.empty(scenario.batch * scenario.predicted_tokens_per_seq,
                            scenario.hidden_size,
                            dtype=scenario.dtype,
                            device="cuda").uniform_(-1, 1)
    input_gen_residual = torch.empty(scenario.batch *
                                     scenario.predicted_tokens_per_seq,
                                     scenario.hidden_size,
                                     dtype=scenario.dtype,
                                     device="cuda").uniform_(-1, 1)
    position_ids_gen = torch.full((scenario.batch, ),
                                  scenario.ctx_len,
                                  dtype=torch.int,
                                  device="cuda")

    extra_attrs = dict()
    # Set all values from scenario
    if scenario.rope_scaling:
        rope_scaling = {
            "beta_fast": scenario.rope_beta_fast,
            "beta_slow": scenario.rope_beta_slow,
            "factor": scenario.rope_factor,
            "mscale": scenario.rope_mscale,
            "mscale_all_dim": scenario.rope_mscale_all_dim,
            "original_max_position_embeddings":
            scenario.rope_original_max_position_embeddings,
            "type": scenario.rope_type,
        }
    else:
        rope_scaling = None
    pretrained_config = PretrainedConfig(
        architectures=["DeepseekV3ForCausalLM"],
        hidden_size=scenario.hidden_size,
        num_attention_heads=scenario.num_heads,
        num_key_value_heads=scenario.num_kv_heads,
        qk_nope_head_dim=scenario.qk_nope_head_dim,
        qk_rope_head_dim=scenario.qk_rope_head_dim,
        q_lora_rank=scenario.q_lora_rank,
        kv_lora_rank=scenario.kv_lora_rank,
        v_head_dim=scenario.v_head_dim,
        max_position_embeddings=scenario.max_position_embeddings,
        rope_theta=scenario.rope_theta,
        rope_scaling=rope_scaling,
        intermediate_size=scenario.intermediate_size,
        moe_intermediate_size=scenario.moe_intermediate_size,
        n_routed_experts=scenario.n_routed_experts,
        n_shared_experts=scenario.n_shared_experts,
        num_experts_per_tok=scenario.num_experts_per_tok,
        # ensure that we get MoE layer, so next two parameters need to be set correctly
        first_k_dense_replace=scenario.first_k_dense_replace,
        moe_layer_freq=scenario.moe_layer_freq,
        n_group=scenario.n_group,
        topk_group=scenario.topk_group,
        routed_scaling_factor=scenario.routed_scaling_factor,
        rms_norm_eps=scenario.rms_norm_eps,
        vocab_size=scenario.vocab_size,
        num_hidden_layers=scenario.num_hidden_layers,
        torch_dtype=scenario.dtype,
        model_type=scenario.model_type,
        use_bfloat16=scenario.dtype == torch.bfloat16,
    )
    # the mapping used for helix
    mapping_with_cp = Mapping(world_size=world_size,
                              rank=rank,
                              moe_ep_size=scenario.ep_size,
                              tp_size=scenario.tp_size,
                              cp_size=world_size // scenario.tp_size,
                              cp_config={'cp_type': CpType.HELIX},
                              gpus_per_node=tensorrt_llm.local_mpi_size())
    mapping_for_config = Mapping(world_size=world_size,
                                 rank=rank,
                                 moe_ep_size=scenario.ep_size,
                                 tp_size=world_size,
                                 cp_size=1,
                                 cp_config=dict(),
                                 gpus_per_node=tensorrt_llm.local_mpi_size())
    with tempfile.TemporaryDirectory() as tmpdir:
        config_dict = pretrained_config.to_dict()
        if QUANTIZATION_TYPE == "nvfp4":
            if not FP4_QUANT_CONFIG_PATH.exists():
                raise FileNotFoundError(
                    f"FP4 quantization config file not found at {FP4_QUANT_CONFIG_PATH}"
                )
            shutil.copy(FP4_QUANT_CONFIG_PATH,
                        os.path.join(tmpdir, "hf_quant_config.json"))
        elif QUANTIZATION_TYPE == "dsfp8":
            config_dict["quantization_config"] = QUANTIZATION_CONFIG_FP8
        config_dict["model_type"] = scenario.model_type
        with open(os.path.join(tmpdir, "config.json"), "w") as f:
            json.dump(config_dict, f)
        config = ModelConfig.from_pretrained(
            checkpoint_dir=tmpdir,
            mapping=mapping_for_config,
            max_num_tokens=scenario.ctx_len,
            attn_backend="TRTLLM",
            moe_backend="TRTLLM",
            use_cuda_graph=True,
            skip_create_weights_in_init=QUANTIZATION_TYPE == "nvfp4",
        )
        if QUANTIZATION_TYPE == "nvfp4":
            config.quant_config.mamba_ssm_cache_dtype = torch.bfloat16
            config.quant_config.kv_cache_quant_algo = QuantAlgo.FP8
    if rank == 0:
        print(f"Rank 0 using config: {config}")
    else:
        print(f"Rank {rank} using same config as rank 0")
    config.extra_attrs = extra_attrs
    aux_stream_list = [torch.cuda.Stream() for _ in range(3)]
    aux_stream_dict = {
        AuxStreamType.Attention: aux_stream_list[0],
        AuxStreamType.MoeShared: aux_stream_list[0],
        AuxStreamType.MoeChunkingOverlap: aux_stream_list[1],
        AuxStreamType.MoeBalancer: aux_stream_list[2],
    }
    layer = DeepseekV3DecoderLayer(
        model_config=config,
        layer_idx=0,
        aux_stream_dict=aux_stream_dict,
        mapping_with_cp=mapping_with_cp,
    )
    apply_exclude_modules(layer, config.quant_config)
    layer = layer.cuda()
    layer.next_layer_layernorm = RMSNorm(hidden_size=scenario.hidden_size,
                                         eps=scenario.rms_norm_eps,
                                         dtype=scenario.dtype).cuda()
    _generate_random_weights(layer)

    # Set up KVCacheManager and attn_metadata for distributed
    kv_cache_manager = _setup_kv(scenario, mapping_with_cp, gen_steps)
    # just explicitly generate KV cache values
    kv_cache_manager.get_buffers(0).uniform_(-1, 1)
    ctx_len_per_gpu = scenario.ctx_len // mapping_with_cp.cp_size

    outputs = []

    # CUDA graph setup for timing
    use_cuda_graph = gen_steps > scenario.ref_steps
    graph = None
    graph_output = None
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    for step in range(gen_steps):
        for req_id in range(scenario.batch):
            kv_cache_manager.impl.add_token(req_id)
        cache_add = step if rank == mapping_with_cp.cp_size - 1 else 0
        cached_tokens_per_seq = [
            ctx_len_per_gpu + cache_add for _ in range(scenario.batch)
        ]
        if step == 0:
            attn_metadata = get_attention_backend("TRTLLM").Metadata(
                seq_lens=torch.tensor([1] * scenario.batch, dtype=torch.int),
                request_ids=list(range(scenario.batch)),
                max_num_requests=scenario.batch,
                num_contexts=0,
                prompt_lens=[ctx_len_per_gpu] * scenario.batch,
                max_num_tokens=ctx_len_per_gpu,
                kv_cache_manager=kv_cache_manager,
                kv_cache_params=KVCacheParams(
                    use_cache=True,
                    num_cached_tokens_per_seq=cached_tokens_per_seq,
                ),
            )
        else:
            attn_metadata.kv_cache_params = KVCacheParams(
                use_cache=True,
                num_cached_tokens_per_seq=cached_tokens_per_seq,
            )
        attn_metadata.prepare()
        extra_attrs["attention_metadata"] = weakref.ref(attn_metadata)
        if not use_cuda_graph:
            # Original non-graph execution
            with model_extra_attrs(extra_attrs):
                result, _ = layer(
                    position_ids=position_ids_gen,
                    hidden_states=input_gen,
                    attn_metadata=attn_metadata,
                    residual=input_gen_residual,
                )
            if step < scenario.ref_steps:
                outputs.append(result)
            # update position_ids_gen
            position_ids_gen += 1
            continue

        # CUDA graph capture on first step when timing
        if step == 0:
            print(
                f"Rank {rank} {world_size}-GPU: Creating CUDA graph and capturing"
            )
            # Create CUDA graph metadata for capture
            attn_metadata = attn_metadata.create_cuda_graph_metadata(
                max_batch_size=scenario.batch)
            attn_metadata.prepare()
            extra_attrs["attention_metadata"] = weakref.ref(attn_metadata)

            # Warm-up runs before graph capture
            for _ in range(2):
                with model_extra_attrs(extra_attrs):
                    result, _ = layer(
                        position_ids=position_ids_gen,
                        hidden_states=input_gen,
                        attn_metadata=attn_metadata,
                        residual=input_gen_residual,
                    )

            # Capture the graph
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                with model_extra_attrs(extra_attrs):
                    with with_multi_stream(True):
                        for _ in range(NUM_HIDDEN_LAYERS):
                            graph_output, _ = layer(
                                position_ids=position_ids_gen,
                                hidden_states=input_gen,
                                attn_metadata=attn_metadata,
                                residual=input_gen_residual,
                            )
            result = graph_output
        elif step == scenario.ref_steps:
            start_event.record()

        graph.replay()
        result = graph_output

        # update position_ids_gen
        position_ids_gen += 1

        # Collect outputs for reference comparison
        if step < scenario.ref_steps:
            outputs.append(result)

    # synchronize to ensure all graphs are done
    end_event.record()
    torch.cuda.synchronize()

    if gen_steps == scenario.ref_steps:
        time_ms = 0.
        avg_gen_time = float('inf')
    else:
        time_ms = start_event.elapsed_time(end_event)
        avg_gen_time = time_ms / (gen_steps - scenario.ref_steps)
    throughput = 1000. * scenario.batch / avg_gen_time
    layer_str = "dense" if scenario.first_k_dense_replace > 0 else "moe"
    print(
        f"Rank {rank} {world_size}-GPU: time taken for "
        f"{gen_steps - scenario.ref_steps} steps of {layer_str}, ctx_len/ctx_len_per_gpu/tp/kvp/ep: "
        f"{scenario.ctx_len}/{ctx_len_per_gpu}/{scenario.tp_size}/{mapping_with_cp.cp_size}/{scenario.ep_size}: "
        f"{time_ms} ms, expected TPOT: {avg_gen_time} ms, "
        f"throughput: {throughput} DS decoder layer/s")
    output = torch.stack(outputs, dim=0)
    kv_cache_manager.shutdown()

    print(
        f"Rank {rank} {world_size}-GPU: output: {output[0, :8]} / {output[-1, -8:]}"
    )


def _run_single_rank(func, *args, **kwargs):
    rank = tensorrt_llm.mpi_rank()
    print(f"rank {rank} starting")
    torch.cuda.set_device(tensorrt_llm.local_mpi_rank())
    print(f"rank {rank} set device to {tensorrt_llm.local_mpi_rank()}")
    try:
        ret = func(rank, *args, **kwargs)
        print(f"rank {rank} done")
        return ret
    except Exception:
        traceback.print_exc()
        tb = traceback.format_exc()
        raise Exception(f"\n\nError occurred. Original traceback is\n{tb}\n")


def test_mla_helix_distributed(scenario: Scenario,
                               gen_steps: Optional[int] = None):
    world_size = scenario.world_size
    gen_steps = scenario.ref_steps if gen_steps is None else gen_steps
    with MPIPoolExecutor(max_workers=world_size) as executor:
        results = executor.map(
            _run_single_rank,
            *zip(*[(_run_ds_layer_distributed, world_size, scenario,
                    gen_steps)] * world_size))
        for r in results:
            assert r is None


if __name__ == "__main__":
    args = parse_args()
    test_scenarios = generate_scenarios(args)
    timing_steps = 10
    world_sizes = [scenario.world_size for scenario in test_scenarios]
    assert len(
        set(world_sizes)) == 1, "all scenarios must have the same world size"
    world_size = world_sizes[0]
    if world_size == 1 or (MPI.COMM_WORLD is not None
                           and MPI.COMM_WORLD.Get_size() > 1):
        # we already have a session setup by mpirun/srun, just call _run_single_rank
        if MPI.COMM_WORLD is not None:
            assert world_size == MPI.COMM_WORLD.Get_size(
            ), "world size must match"
        for scenario in test_scenarios:
            gen_steps = scenario.ref_steps + timing_steps
            _run_single_rank(_run_ds_layer_distributed, world_size, scenario,
                             gen_steps)
    else:
        for scenario in test_scenarios:
            gen_steps = scenario.ref_steps + timing_steps
            print(
                f"Running scenario: {scenario} and timing {timing_steps} steps")
            test_mla_helix_distributed(scenario, gen_steps=gen_steps)
