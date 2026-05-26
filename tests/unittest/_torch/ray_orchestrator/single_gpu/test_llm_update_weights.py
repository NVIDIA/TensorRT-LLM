import base64
import pickle
import re
from typing import Callable, List, Optional, Tuple

import pytest
import torch
from torch import nn
from torch.multiprocessing.reductions import reduce_tensor
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from utils.llm_data import llm_models_root
from utils.torch_ref import RefHFModel
from utils.util import getSMVersion, skip_pre_blackwell, skip_pre_hopper

from tensorrt_llm import LLM
from tensorrt_llm._torch.utils import get_device_uuid
from tensorrt_llm.llmapi import KvCacheConfig, MoeConfig, SamplingParams


# E2M1 boundary midpoints — round to nearest E2M1 magnitude:
#   0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0  (ord 0..7)
_E2M1_BOUNDS = torch.tensor(
    [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], dtype=torch.float32
)


def _quantize_w4a8_canonical(
    weight_float: torch.Tensor,
    block_size: int,
    weight_scale_2: torch.Tensor,
):
    """W4A8 NVFP4 FP8 quantization compatible with ``fp4_fp8_gemm_trtllmgen``.

    Reproduces the convention emitted by the C++ op
    ``float_to_e2m1_and_ufp8sf_scale`` (which the kernel was developed
    against):
    - per-block FP8 scale is stored as a *pure power of 2* (E4M3 byte with
      mantissa bits zeroed). The exponent is ``floor(log2(block_amax)) - 2``
      after scaling the input up by ``448 / amax_weight``.
    - per-tensor ``weight_scale_2`` is ``amax_weight / 448`` (NOT
      ``amax_weight / (6 * 448)`` as in modelopt 0.x / auto_deploy
      ``_quantize_nvfp4``).
    - FP4 values are recomputed AGAINST the power-of-2 block scale so the
      stored ``(packed_fp4, fp8_scale)`` pair is self-consistent.
    """
    n, k = weight_float.shape
    assert k % block_size == 0
    a_global_sf = 1.0 / weight_scale_2  # = 448 / amax_weight
    scaled = weight_float * a_global_sf  # values in ~[-448, 448]
    scaled_blocked = scaled.view(n, k // block_size, block_size)
    block_amax = scaled_blocked.abs().amax(dim=-1).clamp(min=1e-20)  # [n, k/block]
    # C++ uses ``scaleExp = floor(log2(amax)) - 2``; scale = 2^scaleExp ~= amax/4.
    scale_exp = (torch.floor(torch.log2(block_amax)).long() - 2).clamp(min=-6, max=7)
    # FP8 E4M3 byte: sign=0 (1 bit), exp (4 bits, bias 7), mantissa (3 bits)=0.
    e4m3_byte = (((scale_exp + 7) & 0xFF).to(torch.uint8) << 3)  # [n, k/block]
    block_scale_fp8 = e4m3_byte.view(torch.float8_e4m3fn)
    # Recompute FP4 against the (mantissa-zero) power-of-2 scale.
    scale_float = torch.pow(2.0, scale_exp.float())
    scaled_norm = (scaled_blocked / scale_float.unsqueeze(-1)).clamp(min=-6.0, max=6.0)
    abs_val = scaled_norm.abs()
    sign_bit = (scaled_norm < 0).to(torch.uint8)
    bounds = _E2M1_BOUNDS.to(weight_float.device)
    ord_val = (abs_val.unsqueeze(-1) > bounds).sum(dim=-1).to(torch.uint8).clamp(max=7)
    fp4 = ((sign_bit << 3) | ord_val).flatten(start_dim=-2)  # [n, k] uint8
    fp4_packed = ((fp4[..., 1::2] & 0x0F) << 4) | (fp4[..., 0::2] & 0x0F)
    return fp4_packed.to(torch.uint8), block_scale_fp8, weight_scale_2

# Ray-backed LLM teardown spawns the executor main-loop, GC and log/error
# listener threads in ray-core. These are torn down only when ``ray.shutdown()``
# fires, which only runs from RayExecutor.shutdown() — itself only called when
# the LLM object is explicitly closed. The transformers 5.5.x import graph
# enlarges the live reference set, delaying GC of the test-local LLM past
# pytest-threadleak's post-teardown snapshot. Disable the leak check for this
# file (matches the sibling pattern at
# tests/unittest/_torch/ray_orchestrator/multi_gpu/test_executor.py).
pytestmark = pytest.mark.threadleak(enabled=False)


class RefHFModelWithIPCHandles(RefHFModel):
    def __init__(self, model_dir: str, device_id: int = 0, num_hidden_layers: int = 4):
        self.device_id = device_id
        config = AutoConfig.from_pretrained(model_dir)
        config.num_hidden_layers = num_hidden_layers
        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir, config=config, torch_dtype=torch.bfloat16, attn_implementation="eager"
        ).to(f"cuda:{device_id}")
        self.all_weights = {}
        self.device_uuid = [get_device_uuid(i) for i in range(torch.cuda.device_count())]
        self._replicate_weights()

    def _replicate_weights(self):
        model_weights = []
        for n, p in self.model.named_parameters():
            model_weights.append((n, p.detach().clone()))

        self.all_weights[self.device_id] = model_weights
        for i in range(torch.cuda.device_count()):
            if i != self.device_id:
                cur_weights = []
                for n, p in self.all_weights[self.device_id]:
                    cur_weights.append((n, p.to("cuda:" + str(i))))
                self.all_weights[i] = cur_weights

    def get_weight_ipc_handles_serialized(
        self,
        device_ids: Optional[List[int]] = None,
        weight_filter: Optional[Callable[[str], bool]] = None,
    ):
        """
        Get base64-encoded serialized IPC handles for model weights.

        Args:
            device_ids: List of CUDA device indices to get weights from
            weight_filter: Optional function that takes weight name and returns True if weight should be included

        Returns:
            ret: Dictionary mapping device UUIDs to base64-encoded pickled handles
        """
        ret = {}
        device_list = list(range(torch.cuda.device_count())) if device_ids is None else device_ids

        for device in device_list:
            all_handles = []
            for item in self.all_weights[device]:
                name, p = item
                # Apply filter if provided
                if weight_filter is not None and not weight_filter(name):
                    continue
                handle = reduce_tensor(p)
                all_handles.append((name, handle))

            # Serialize with base64-encoded pickle
            serialized = base64.b64encode(pickle.dumps(all_handles)).decode("utf-8")
            ret[self.device_uuid[device]] = serialized

        return ret


def compare_logits(
    logits_list: List[torch.Tensor],
    ref_logits_list: List[torch.Tensor],
    topk: int = 20,
    threshold: float = 0.9,
):
    assert len(logits_list) == len(ref_logits_list)
    for i in range(len(logits_list)):
        assert logits_list[i].shape == ref_logits_list[i].shape
        lhs_idx = torch.topk(logits_list[i], topk, dim=-1).indices
        rhs_idx = torch.topk(ref_logits_list[i], topk, dim=-1).indices
        # Token wise comparison
        ratios = []
        for j in range(lhs_idx.shape[0]):
            lhs_idx_j = lhs_idx[j].tolist()
            rhs_idx_j = rhs_idx[j].tolist()
            overlap = set(lhs_idx_j) & set(rhs_idx_j)
            ratios.append(len(overlap) / len(lhs_idx_j))

        mean_ratio = sum(ratios) / len(ratios)
        assert mean_ratio > threshold, (
            f"Prompt {i}: overlap ratio: {mean_ratio:.2%} is less than {threshold:.2%}"
        )


def run_generate(
    llm: LLM,
    hf_model: RefHFModel,
    prompts: List[List[int]],
    sampling_params: SamplingParams,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    llm_responses = []
    llm_logits = []
    outputs = llm.generate(prompts, sampling_params)
    for output in outputs:
        llm_logits.append(output.outputs[0].generation_logits)
        llm_responses.append(output.outputs[0].token_ids)
    input_ids, attention_mask, position_ids = RefHFModel.pad_data(prompts, llm_responses)
    ref_logits = hf_model.generate_batch_with_padding(
        input_ids, attention_mask, position_ids, llm_responses, return_logits=True
    )
    return llm_logits, ref_logits


@skip_pre_hopper
@pytest.mark.parametrize(
    "model_dir",
    [
        "llama-models-v2/TinyLlama-1.1B-Chat-v1.0",
        "Qwen2.5-0.5B-Instruct",
        "Qwen3/Qwen3-8B",
        "Qwen3/Qwen3-30B-A3B",
        "Qwen3/Qwen3-8B-FP8",
        "Qwen3/Qwen3-30B-A3B-FP8",
    ],
)
@pytest.mark.part0
def test_llm_update_weights(model_dir):
    model_dir = str(llm_models_root() / model_dir)
    num_hidden_layers = 1
    hf_model = RefHFModelWithIPCHandles(model_dir, num_hidden_layers=num_hidden_layers)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    kv_cache_config = KvCacheConfig(enable_block_reuse=True, free_gpu_memory_fraction=0.1)
    moe_config = MoeConfig(backend="DEEPGEMM" if getSMVersion() >= 100 else "CUTLASS")
    with LLM(
        model=model_dir,
        ray_worker_extension_cls="tensorrt_llm.llmapi.rlhf_utils.WorkerExtension",
        tensor_parallel_size=1,
        load_format="dummy",
        pipeline_parallel_size=1,
        kv_cache_config=kv_cache_config,
        model_kwargs={"num_hidden_layers": num_hidden_layers},
        moe_config=moe_config,
    ) as llm:
        # Generate texts from the prompts.
        prompts_texts = [
            "Hello, my name is",
            "The president of the United States is",
            "The capital of France is",
            "The future of AI is",
        ]
        prompts = [tokenizer.encode(prompt) for prompt in prompts_texts]
        del tokenizer
        sampling_params = SamplingParams(
            temperature=0, return_generation_logits=True, max_tokens=1024
        )

        ipc_handles = hf_model.get_weight_ipc_handles_serialized([0])

        llm._collective_rpc("update_weights", (ipc_handles,))
        # Finalize the update weights
        llm._collective_rpc("update_weights", (None,))

        llm_logits, ref_logits = run_generate(llm, hf_model, prompts, sampling_params)
        compare_logits(llm_logits, ref_logits)


@skip_pre_hopper
@pytest.mark.parametrize(
    "model_dir",
    [
        "llama-models-v2/TinyLlama-1.1B-Chat-v1.0",
        "Qwen2.5-0.5B-Instruct",
        "Qwen3/Qwen3-8B",
        "Qwen3/Qwen3-30B-A3B",
        "Qwen3/Qwen3-8B-FP8",
        "Qwen3/Qwen3-30B-A3B-FP8",
    ],
)
@pytest.mark.part1
def test_llm_partial_update_weights(model_dir):
    model_dir = str(llm_models_root() / model_dir)
    num_hidden_layers = 1
    hf_model = RefHFModelWithIPCHandles(model_dir, num_hidden_layers=num_hidden_layers)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    kv_cache_config = KvCacheConfig(enable_block_reuse=True, free_gpu_memory_fraction=0.1)
    moe_config = MoeConfig(backend="DEEPGEMM" if getSMVersion() >= 100 else "CUTLASS")
    with LLM(
        model=model_dir,
        ray_worker_extension_cls="tensorrt_llm.llmapi.rlhf_utils.WorkerExtension",
        tensor_parallel_size=1,
        load_format="dummy",
        pipeline_parallel_size=1,
        kv_cache_config=kv_cache_config,
        model_kwargs={"num_hidden_layers": num_hidden_layers},
        moe_config=moe_config,
    ) as llm:
        # Generate texts from the prompts.
        prompts_texts = [
            "Hello, my name is",
            "The president of the United States is",
            "The capital of France is",
            "The future of AI is",
        ]
        prompts = [tokenizer.encode(prompt) for prompt in prompts_texts]
        del tokenizer

        sampling_params = SamplingParams(
            temperature=0, return_generation_logits=True, max_tokens=1024
        )

        def common_filter(filter_name: str) -> Callable[[str], bool]:
            def filter_fn(name: str) -> bool:
                return name.endswith(filter_name)

            return filter_fn

        # Generate filter_list from model weight keys by removing layer prefix
        # e.g., "model.layers.41.input_layernorm.weight" -> "input_layernorm.weight"
        layer_prefix_pattern = re.compile(r"^model\.layers\.\d+\.")
        filter_set = set()
        for name, _ in hf_model.all_weights[hf_model.device_id]:
            suffix = layer_prefix_pattern.sub("", name)
            filter_set.add(suffix)
        filter_list = list(filter_set)

        for filter_name in filter_list:
            weight_filter = common_filter(filter_name=filter_name)
            ipc_handles = hf_model.get_weight_ipc_handles_serialized(
                [0], weight_filter=weight_filter
            )
            llm._collective_rpc("update_weights", (ipc_handles,))
        # Finalize the update weights
        llm._collective_rpc("update_weights", (None,))

        llm_logits, ref_logits = run_generate(llm, hf_model, prompts, sampling_params)
        compare_logits(llm_logits, ref_logits)


@skip_pre_hopper
@pytest.mark.parametrize(
    "model_dir, fp8_model_dir",
    [
        ("Qwen3/Qwen3-8B", "Qwen3/Qwen3-8B-FP8"),
        ("Qwen3/Qwen3-30B-A3B", "Qwen3/Qwen3-30B-A3B-FP8"),
    ],
)
@pytest.mark.parametrize("kv_cache_dtype", ["auto", "fp8"])
@pytest.mark.part2
def test_llm_update_weights_with_quant_config(model_dir, fp8_model_dir, kv_cache_dtype):
    model_dir = str(llm_models_root() / model_dir)
    fp8_model_dir = str(llm_models_root() / fp8_model_dir)
    num_hidden_layers = 1
    hf_model = RefHFModelWithIPCHandles(fp8_model_dir, num_hidden_layers=num_hidden_layers)
    tokenizer = AutoTokenizer.from_pretrained(fp8_model_dir)
    kv_cache_config = KvCacheConfig(
        enable_block_reuse=True, free_gpu_memory_fraction=0.1, dtype=kv_cache_dtype
    )
    moe_config = MoeConfig(backend="DEEPGEMM" if getSMVersion() >= 100 else "CUTLASS")
    with LLM(
        model=model_dir,
        ray_worker_extension_cls="tensorrt_llm.llmapi.rlhf_utils.WorkerExtension",
        tensor_parallel_size=1,
        load_format="dummy",
        pipeline_parallel_size=1,
        kv_cache_config=kv_cache_config,
        model_kwargs={
            "num_hidden_layers": num_hidden_layers,
            "quantization_config": {
                "activation_scheme": "dynamic",
                "fmt": "e4m3",
                "quant_method": "fp8",
                "weight_block_size": [128, 128],
            },
        },
        moe_config=moe_config,
    ) as llm:
        # Generate texts from the prompts.
        prompts_texts = [
            "Hello, my name is",
            "The president of the United States is",
            "The capital of France is",
            "The future of AI is",
        ]
        prompts = [tokenizer.encode(prompt) for prompt in prompts_texts]
        del tokenizer
        sampling_params = SamplingParams(
            temperature=0, return_generation_logits=True, max_tokens=1024
        )

        ipc_handles = hf_model.get_weight_ipc_handles_serialized([0])

        llm._collective_rpc("update_weights", (ipc_handles,))
        # Finalize the update weights
        llm._collective_rpc("update_weights", (None,))

        llm_logits, ref_logits = run_generate(llm, hf_model, prompts, sampling_params)
        compare_logits(llm_logits, ref_logits)


# E2M1 lookup table for the 16 FP4 values, used by the W4A8 reference dequantize.
_E2M1_VALUES = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)


def _dequantize_nvfp4_block32(
    quantized_t: torch.Tensor,  # [N, K/2] uint8 (packed E2M1x2)
    scale_1: torch.Tensor,  # per-block FP8 scale (flat or shaped)
    scale_2: torch.Tensor,  # per-tensor fp32 scale
    orig_shape: tuple,
    orig_dtype: torch.dtype,
) -> torch.Tensor:
    """Dequantize NVFP4-packed weight with block size 32.

    The W4A8 NVFP4 FP8 path uses a 32-element block, whereas the existing
    ``_dequantize_nvfp4`` helper from ``auto_deploy.torch_quant`` hardcodes
    block size 16, so we need a local copy here.
    """
    device = quantized_t.device
    n, k = orig_shape
    block_size = 32
    num_blocks = n * (k // block_size)
    s1 = scale_1.reshape(-1)[:num_blocks]

    high = (quantized_t >> 4) & 0x0F
    low = quantized_t & 0x0F
    idx = torch.empty(n, (k // 2) * 2, dtype=torch.long, device=device)
    idx[..., 0::2] = low.long()
    idx[..., 1::2] = high.long()

    vals = _E2M1_VALUES.to(device)[idx]  # [N, K], float32

    scale_real = (s1.to(torch.float32) * scale_2.to(torch.float32)).view(
        n, k // block_size, 1
    )
    vals = vals.view(n, k // block_size, block_size) * scale_real
    return vals.view(n, k).to(orig_dtype)


class RefW4A8NVFP4FP8ModelWithIPCHandles(RefHFModel):
    """Reference model that loads bf16 weights from HuggingFace, quantizes
    them to the W4A8 NVFP4 FP8 weight format (FP4 packed weight + FP8 block
    scale + fp32 weight_scale_2), and keeps a round-tripped bf16 model for
    HF reference inference.

    Mirrors ``RefNVFP4ModelWithIPCHandles`` from the multi-GPU test file but:
    - uses block size 32 (vs NVFP4's 16) to match
      ``W4A8NVFP4FP8LinearMethod.scaling_vector_size``
    - calibrates per-Linear ``input_scale`` by hooking nn.Linear inputs
      while the bf16 HF model runs the calibration prompts. The resulting
      ``input_scale = amax/448`` is shipped via IPC alongside the FP4
      weight + FP8 block scale + FP32 weight_scale_2.

    Provides IPC handles with keys ``weight`` (FP4 packed uint8),
    ``weight_scale`` (FP8 per-block), ``weight_scale_2`` (fp32 per-tensor),
    and ``input_scale`` (fp32 per-tensor, static). q/k/v and gate/up
    projections in a fusion group share a unified ``weight_scale_2`` so
    the per-tensor cross-shard consistency check inside
    ``_finalize_w4a8_scales`` passes.
    """

    W4A8_BLOCK_SIZE = 32

    EXCLUDE_PATTERNS = [
        "embed_tokens",
        "lm_head",
        "layernorm",
        "norm",
        "ln_",
        "embeddings",
        "mlp.gate.weight",
    ]
    INCLUDE_PATTERNS = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "fc1",
        "fc2",
        "mlp",
    ]
    FUSION_GROUPS = (
        ("q_proj", "k_proj", "v_proj"),
        ("gate_proj", "up_proj"),
    )
    _PROJ_TO_GROUP = {proj: group for group in FUSION_GROUPS for proj in group}

    # 4 prompts used both for calibration and at test time (same input distribution).
    CALIBRATION_PROMPTS = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]

    def __init__(
        self, model_dir: str, device_id: int = 0, num_hidden_layers: Optional[int] = None
    ):
        self.device_id = device_id
        self.model_dir = model_dir
        config = AutoConfig.from_pretrained(model_dir)
        if num_hidden_layers is not None:
            config.num_hidden_layers = num_hidden_layers
        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir, config=config, torch_dtype=torch.bfloat16, attn_implementation="eager"
        ).to(f"cuda:{device_id}")
        self.all_weights = {}
        self._dequantized_weights = {}
        # input_scales[<linear-name>.weight] = FP32 scalar (amax/448) used for static
        # FP8 activation quantization. Filled by _calibrate_input_scales().
        self.input_scales: dict = {}
        self.device_uuid = [get_device_uuid(i) for i in range(torch.cuda.device_count())]
        self._calibrate_input_scales()
        self._quantize_and_replicate_weights()

    def _calibrate_input_scales(self):
        """Run the HF model on a few prompts, hooking every nn.Linear's input
        to collect per-tensor amax. ``input_scale = amax / 448`` (FP8 e4m3fn max).

        Stored under the same key the weight uses (``<name>.weight`` →
        ``<name>.input_scale``) so the IPC handle can ship them alongside the
        FP4 weights for static activation quantization on the LLM side.
        """
        amax: dict = {}
        hooks = []

        def make_hook(linear_name):
            def hook(_mod, inputs):
                x = inputs[0]
                # inputs may be a tuple (x, ...) for some Linear subclasses; first elem is the activation.
                if not isinstance(x, torch.Tensor):
                    return
                cur = x.detach().float().abs().amax().item()
                amax[linear_name] = max(amax.get(linear_name, 0.0), cur)
            return hook

        for name, mod in self.model.named_modules():
            if isinstance(mod, nn.Linear):
                hooks.append(mod.register_forward_pre_hook(make_hook(name)))

        tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        device = f"cuda:{self.device_id}"
        self.model.eval()
        with torch.no_grad():
            for prompt in self.CALIBRATION_PROMPTS:
                ids = torch.tensor(
                    [tokenizer.encode(prompt)], dtype=torch.long, device=device
                )
                self.model(ids)
        for h in hooks:
            h.remove()
        del tokenizer

        # Convert amax to input_scale (FP32 scalar) on CUDA so IPC reduce_tensor works.
        for linear_name, a in amax.items():
            if a <= 0.0:
                # No tokens routed through this expert during calibration — fall back to 1.0
                a = 1.0
            self.input_scales[f"{linear_name}.weight"] = torch.tensor(
                a / 448.0, dtype=torch.float32, device=device
            )

    @staticmethod
    def _unfuse_moe_expert_params(all_params):
        """Same MoE-expert unfusing logic as ``RefNVFP4ModelWithIPCHandles``."""
        unfused = []
        refuse_map: dict = {}
        for name, p in all_params:
            if name.endswith(".experts.gate_up_proj") and p.dim() == 3:
                prefix = name[: -len("gate_up_proj")]
                num_experts = p.shape[0]
                half = p.shape[1] // 2
                for i in range(num_experts):
                    g_name = f"{prefix}{i}.gate_proj.weight"
                    u_name = f"{prefix}{i}.up_proj.weight"
                    unfused.append((g_name, p[i, :half, :].clone()))
                    unfused.append((u_name, p[i, half:, :].clone()))
                    refuse_map[g_name] = (name, i, "gate")
                    refuse_map[u_name] = (name, i, "up")
            elif name.endswith(".experts.down_proj") and p.dim() == 3:
                prefix = name[: -len("down_proj")]
                num_experts = p.shape[0]
                for i in range(num_experts):
                    d_name = f"{prefix}{i}.down_proj.weight"
                    unfused.append((d_name, p[i].clone()))
                    refuse_map[d_name] = (name, i, "down")
            else:
                unfused.append((name, p))
        return unfused, refuse_map

    def _quantize_and_replicate_weights(self):
        all_params = [
            (name, param.detach().clone()) for name, param in self.model.named_parameters()
        ]
        all_params, moe_refuse_map = self._unfuse_moe_expert_params(all_params)

        fusion_buffer: dict = {}
        model_weights = []

        for name, p in all_params:
            if not self._should_quantize(name):
                model_weights.append((name, p))
                continue

            proj_type, group = self._get_proj_info(name)
            if proj_type is not None:
                layer_prefix = name[: name.index(proj_type)].rstrip(".")
                buf_key = (layer_prefix, group)
                fusion_buffer.setdefault(buf_key, {})[proj_type] = (name, p)
                if all(proj in fusion_buffer[buf_key] for proj in group):
                    projs = fusion_buffer.pop(buf_key)
                    model_weights.extend(self._quantize_fusion_group(group, projs))
            else:
                model_weights.extend(self._quantize_single_weight(name, p))

        assert not fusion_buffer, f"Incomplete fusion groups: {list(fusion_buffer.keys())}"

        self.all_weights[self.device_id] = model_weights
        for i in range(torch.cuda.device_count()):
            if i != self.device_id:
                self.all_weights[i] = [(n, p.to(f"cuda:{i}")) for n, p in model_weights]

        with torch.no_grad():
            param_dict = dict(self.model.named_parameters())
            for name, dequant_weight in self._dequantized_weights.items():
                if name in param_dict:
                    param_dict[name].copy_(dequant_weight)
                elif name in moe_refuse_map:
                    fused_name, expert_idx, kind = moe_refuse_map[name]
                    fused = param_dict[fused_name]
                    if kind == "gate":
                        half = fused.shape[1] // 2
                        fused[expert_idx, :half, :].copy_(dequant_weight)
                    elif kind == "up":
                        half = fused.shape[1] // 2
                        fused[expert_idx, half:, :].copy_(dequant_weight)
                    else:
                        fused[expert_idx].copy_(dequant_weight)
        del self._dequantized_weights

    @classmethod
    def _should_quantize(cls, name: str) -> bool:
        if not name.endswith(".weight"):
            return False
        name_lower = name.lower()
        for pattern in cls.EXCLUDE_PATTERNS:
            if pattern in name_lower:
                return False
        for pattern in cls.INCLUDE_PATTERNS:
            if pattern in name_lower:
                return True
        return False

    @classmethod
    def _get_proj_info(cls, name: str):
        for proj, group in cls._PROJ_TO_GROUP.items():
            if proj in name:
                return proj, group
        return None, None

    def _quantize_single_weight(self, name: str, weight: torch.Tensor) -> List[tuple]:
        return list(self._do_quantize(name, weight))

    def _quantize_fusion_group(self, group: tuple, projs: dict) -> List[tuple]:
        all_amaxes = [v.float().abs().amax() for _, v in projs.values()]
        # weight_scale_2 = amax/448 (NOT amax/(6*448)) — matches the convention
        # the kernel's canonical quantizer (float_to_e2m1_and_ufp8sf_scale) uses.
        unified_scale_2 = torch.stack(all_amaxes).amax() / 448.0

        results = []
        for proj_type in group:
            if proj_type in projs:
                name, weight = projs[proj_type]
                results.extend(self._do_quantize(name, weight, unified_scale_2))
        return results

    def _do_quantize(
        self,
        name: str,
        weight: torch.Tensor,
        weight_scale_2: Optional[torch.Tensor] = None,
    ) -> List[tuple]:
        weight_float = weight.float()
        if weight_scale_2 is None:
            weight_scale_2 = weight_float.abs().amax().float() / 448.0

        packed_uint8, block_scale_fp8, _ws2 = _quantize_w4a8_canonical(
            weight_float, self.W4A8_BLOCK_SIZE, weight_scale_2
        )

        self._dequantized_weights[name] = _dequantize_nvfp4_block32(
            packed_uint8,
            block_scale_fp8,
            weight_scale_2,
            weight_float.shape,
            torch.bfloat16,
        )

        entries = [
            (name, packed_uint8),
            (name + "_scale", block_scale_fp8),
            (name + "_scale_2", weight_scale_2),
        ]
        # Static activation quantization: ship per-tensor input_scale calibrated
        # offline. Required for the MoE TRTLLMGen W4A8 NVFP4 FP8 path
        # (fused_moe_trtllm_gen.py: it calls
        # ``static_quantize_e4m3_per_tensor(x, 1.0/fc31_input_scale)`` with no
        # dynamic-quant fallback). Dense W4A8NVFP4FP8LinearMethod consumes the
        # same key via load_weight_scales and stashes it in
        # tmp_w4a8_input_scales_list for the alpha computation in finalize.
        # Key format: replace ``.weight`` suffix with ``.input_scale``, mirroring
        # how modelopt-emitted checkpoints carry these per-Linear tensors.
        if name.endswith(".weight") and name in self.input_scales:
            entries.append(
                (name.replace(".weight", ".input_scale"),
                 self.input_scales[name].clone())
            )
        return entries

    def get_weight_ipc_handles_serialized(
        self,
        device_ids: Optional[List[int]] = None,
        weight_filter: Optional[Callable[[str], bool]] = None,
    ):
        ret = {}
        device_list = list(range(torch.cuda.device_count())) if device_ids is None else device_ids
        for device in device_list:
            all_handles = []
            for item in self.all_weights[device]:
                name, p = item
                if weight_filter is not None and not weight_filter(name):
                    continue
                handle = reduce_tensor(p)
                all_handles.append((name, handle))
            serialized = base64.b64encode(pickle.dumps(all_handles)).decode("utf-8")
            ret[self.device_uuid[device]] = serialized
        return ret


@skip_pre_blackwell
@pytest.mark.parametrize(
    "model_dir",
    [
        "Qwen3/Qwen3-8B-Base",
        "Qwen3-30B-A3B-Base",
    ],
)
@pytest.mark.part3
def test_llm_update_weights_w4a8_nvfp4_fp8(model_dir):
    model_dir = str(llm_models_root() / model_dir)
    num_hidden_layers = 1
    hf_model = RefW4A8NVFP4FP8ModelWithIPCHandles(
        model_dir, num_hidden_layers=num_hidden_layers
    )
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    kv_cache_config = KvCacheConfig(enable_block_reuse=True, free_gpu_memory_fraction=0.1)
    # TRTLLM (trtllm_gen) is the only MoE backend with a W4A8_NVFP4_FP8 dispatch
    # case today (W4A8NVFP4FP8TRTLLMGenFusedMoEMethod). Cutlass / DeepGemm
    # backends do not (would hit "Unsupported quantization mode: [16384]").
    moe_config = MoeConfig(backend="TRTLLM")
    with LLM(
        model=model_dir,
        ray_worker_extension_cls="tensorrt_llm.llmapi.rlhf_utils.WorkerExtension",
        tensor_parallel_size=1,
        load_format="dummy",
        pipeline_parallel_size=1,
        kv_cache_config=kv_cache_config,
        model_kwargs={
            "num_hidden_layers": num_hidden_layers,
            "quantization_config": {
                "producer": {"name": "modelopt"},
                "quant_algo": "W4A8_NVFP4_FP8",
                "group_size": 32,
                "exclude_modules": ["*mlp.gate", "lm_head"],
            },
        },
        moe_config=moe_config,
    ) as llm:
        prompts_texts = [
            "Hello, my name is",
            "The president of the United States is",
            "The capital of France is",
            "The future of AI is",
        ]
        prompts = [tokenizer.encode(prompt) for prompt in prompts_texts]
        del tokenizer
        sampling_params = SamplingParams(
            temperature=0, return_generation_logits=True, max_tokens=1024
        )

        ipc_handles = hf_model.get_weight_ipc_handles_serialized([0])
        llm._collective_rpc("update_weights", (ipc_handles,))
        llm._collective_rpc("update_weights", (None,))

        llm_logits, ref_logits = run_generate(llm, hf_model, prompts, sampling_params)
        # Looser threshold: W4A8 logits are compared against a BF16 reference.
        compare_logits(llm_logits, ref_logits, threshold=0.8)


@skip_pre_blackwell
@pytest.mark.parametrize(
    "model_dir",
    [
        "Qwen3/Qwen3-8B-Base",
        "Qwen3-30B-A3B-Base",
    ],
)
@pytest.mark.part4
def test_llm_partial_update_weights_w4a8_nvfp4_fp8(model_dir):
    model_dir = str(llm_models_root() / model_dir)
    num_hidden_layers = 1
    hf_model = RefW4A8NVFP4FP8ModelWithIPCHandles(
        model_dir, num_hidden_layers=num_hidden_layers
    )
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    kv_cache_config = KvCacheConfig(enable_block_reuse=True, free_gpu_memory_fraction=0.1)
    # See note in test_llm_update_weights_w4a8_nvfp4_fp8: only TRTLLM (trtllm_gen)
    # MoE backend has a W4A8_NVFP4_FP8 dispatch case today.
    moe_config = MoeConfig(backend="TRTLLM")
    with LLM(
        model=model_dir,
        ray_worker_extension_cls="tensorrt_llm.llmapi.rlhf_utils.WorkerExtension",
        tensor_parallel_size=1,
        load_format="dummy",
        pipeline_parallel_size=1,
        kv_cache_config=kv_cache_config,
        model_kwargs={
            "num_hidden_layers": num_hidden_layers,
            "quantization_config": {
                "producer": {"name": "modelopt"},
                "quant_algo": "W4A8_NVFP4_FP8",
                "group_size": 32,
                "exclude_modules": ["*mlp.gate", "lm_head"],
            },
        },
        moe_config=moe_config,
    ) as llm:
        prompts_texts = [
            "Hello, my name is",
            "The president of the United States is",
            "The capital of France is",
            "The future of AI is",
        ]
        prompts = [tokenizer.encode(prompt) for prompt in prompts_texts]
        del tokenizer

        sampling_params = SamplingParams(
            temperature=0, return_generation_logits=True, max_tokens=1024
        )

        def common_filter(filter_name: str) -> Callable[[str], bool]:
            def filter_fn(name: str) -> bool:
                return name.endswith(filter_name)

            return filter_fn

        layer_prefix_pattern = re.compile(r"^model\.layers\.\d+\.")
        filter_set = set()
        for name, _ in hf_model.all_weights[hf_model.device_id]:
            suffix = layer_prefix_pattern.sub("", name)
            filter_set.add(suffix)
        filter_list = list(filter_set)

        for filter_name in filter_list:
            weight_filter = common_filter(filter_name=filter_name)
            ipc_handles = hf_model.get_weight_ipc_handles_serialized(
                [0], weight_filter=weight_filter
            )
            llm._collective_rpc("update_weights", (ipc_handles,))
        llm._collective_rpc("update_weights", (None,))

        llm_logits, ref_logits = run_generate(llm, hf_model, prompts, sampling_params)
        # Looser threshold: W4A8 logits are compared against a BF16 reference.
        compare_logits(llm_logits, ref_logits, threshold=0.8)
