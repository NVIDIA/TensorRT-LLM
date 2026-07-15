import ast
import inspect
import json
import struct
import textwrap
from types import SimpleNamespace

import pytest
import torch
from transformers import PretrainedConfig

# from utils.util import default_dtype
from tensorrt_llm._torch.attention_backend.interface import AttentionForwardArgs
from tensorrt_llm._torch.attention_backend.sparse.deepseek_v4.compressor import Compressor
from tensorrt_llm._torch.attention_backend.sparse.deepseek_v4.deepseek_v4 import (
    DeepseekV4Indexer,
    DeepseekV4TrtllmAttention,
)
from tensorrt_llm._torch.attention_backend.trtllm import TrtllmAttention
from tensorrt_llm._torch.configs.deepseekv4 import DeepseekV4Config
from tensorrt_llm._torch.models.modeling_deepseekv4 import (
    DeepseekV4Attention,
    DeepseekV4ForCausalLM,
    DeepseekV4Gate,
    DeepseekV4WeightLoader,
    _copy_deepseek_v4_fused_a_weight_scale,
    _remap_deepseek_v4_checkpoint_keys,
    _resolve_enable_fused_hc,
)
from tensorrt_llm._torch.pyexecutor.model_loader import ModelLoader, initialize_dummy_weights

DEEPSEEK_V4_TINY_CONFIG = {
    "architectures": ["DeepseekV4ForCausalLM"],
    "model_type": "deepseek_v4",
    "hidden_size": 4096,
    "num_attention_heads": 64,
    "num_key_value_heads": 1,
    "qk_nope_head_dim": 448,
    "qk_rope_head_dim": 64,
    "v_head_dim": 512,
    "q_lora_rank": 1024,
    "kv_lora_rank": 448,
    "o_groups": 8,
    "o_lora_rank": 1024,
    "max_position_embeddings": 65536,
    "rms_norm_eps": 1e-6,
    "dtype": "bfloat16",
    "vocab_size": 129280,
    "num_hidden_layers": 7,
    "n_hash_layers": 3,
    "moe_intermediate_size": 2048,
    "n_routed_experts": 256,
    "n_shared_experts": 1,
    "num_experts_per_tok": 6,
    "n_group": 1,
    "topk_group": 1,
    "routed_scaling_factor": 1.5,
    "score_func": "sqrtsoftplus",
    "hc_mult": 4,
    "hc_sinkhorn_iters": 20,
    "hc_eps": 1e-6,
    "compress_rope_theta": 40000.0,
    "rope_theta": 10000.0,
    "rope_scaling": {
        "type": "yarn",
        "factor": 4.0,
        "original_max_position_embeddings": 65536,
        "beta_fast": 32,
        "beta_slow": 1,
    },
    "quantization_config": {
        "activation_scheme": "dynamic",
        "fmt": "e4m3",
        "quant_method": "fp8",
        "scale_fmt": "ue8m0",
        "weight_block_size": [128, 128],
    },
}


def _source_calls(source):
    return {
        ast.unparse(node)
        for node in ast.walk(ast.parse(textwrap.dedent(source)))
        if isinstance(node, ast.Call)
    }


def _write_safetensors_header(path, tensor_name, dtype, shape):
    header = {
        tensor_name: {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [0, 0],
        }
    }
    payload = json.dumps(header).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(payload)) + payload)


def test_deepseek_v4_config_aliases():
    config = DeepseekV4Config(
        num_hash_layers=5, sliding_window=256, head_dim=128, score_func="sigmoid", swiglu_limit=9.0
    )

    assert config.model_type == "deepseek_v4"
    assert config.n_hash_layers == 5
    assert config.window_size == 256
    assert config.v_head_dim == 128
    assert config.scoring_func == "sigmoid"
    assert config.swiglu_limit == 9.0


def test_deepseek_v4_fused_hc_default_enabled(monkeypatch):
    monkeypatch.delenv("TRTLLM_MHC_ENABLE_FUSED_HC", raising=False)
    config = PretrainedConfig()

    assert _resolve_enable_fused_hc(config) is True

    config.enable_fused_hc = False
    assert _resolve_enable_fused_hc(config) is False

    monkeypatch.setenv("TRTLLM_MHC_ENABLE_FUSED_HC", "1")
    assert _resolve_enable_fused_hc(config) is True

    monkeypatch.setenv("TRTLLM_MHC_ENABLE_FUSED_HC", "0")
    assert _resolve_enable_fused_hc(config) is False


def test_deepseek_v4_model_defaults():
    class LlmArgs:
        pass

    defaults = DeepseekV4ForCausalLM.get_model_defaults(LlmArgs())

    assert defaults == {
        "kv_cache_config": {
            "tokens_per_block": 128,
            "use_kv_cache_manager_v2": True,
            "enable_swa_scratch_reuse": True,
        }
    }


def test_deepseek_v4_weight_remap_for_mxfp4_routed_experts():
    weights = {
        "layers.0.ffn.experts.0.w1.weight": torch.tensor([[-1, 2], [3, -4]], dtype=torch.int8),
        "layers.0.ffn.experts.0.w1.scale": torch.tensor([1, 2], dtype=torch.int8),
    }

    remapped = _remap_deepseek_v4_checkpoint_keys(weights, num_hidden_layers=1, kv_lora_rank=448)

    assert remapped["model.layers.0.mlp.experts.0.w1.weight"].dtype == torch.uint8
    assert remapped["model.layers.0.mlp.experts.0.w1.weight_scale"].dtype == torch.uint8


def test_deepseek_v4_weight_remap_for_fp8_routed_experts():
    weights = {
        "layers.0.ffn.experts.0.w1.weight": torch.zeros((2, 2), dtype=torch.float32),
        "layers.0.ffn.experts.0.w1.scale": torch.ones((2, 2), dtype=torch.float32),
    }

    remapped = _remap_deepseek_v4_checkpoint_keys(weights, num_hidden_layers=1, kv_lora_rank=448)

    assert "model.layers.0.mlp.experts.0.w1.weight_scale_inv" in remapped
    assert "model.layers.0.mlp.experts.0.w1.weight_scale" not in remapped


def test_deepseek_v4_fused_a_weight_scale_rebuilds_fp8_shape():
    module = torch.nn.Module()
    module.weight = torch.nn.Parameter(torch.empty((2048, 7168), dtype=torch.float8_e4m3fn))
    module.weight_scale = torch.nn.Parameter(torch.empty((16, 16), dtype=torch.float32))
    module.rebuild_tensor_metadata = {}
    fused_a = torch.empty((2048, 7168), dtype=torch.float8_e4m3fn)
    fused_a_scale = torch.ones((16, 56), dtype=torch.float32)

    _copy_deepseek_v4_fused_a_weight_scale(module, fused_a, fused_a_scale)

    assert module.weight_scale.shape == fused_a_scale.shape
    assert torch.equal(module.weight_scale, fused_a_scale)


def test_deepseek_v4_fused_a_weight_scale_keeps_oversized_slice():
    module = torch.nn.Module()
    module.weight = torch.nn.Parameter(torch.empty((2176, 7168), dtype=torch.float8_e4m3fn))
    module.weight_scale = torch.nn.Parameter(torch.zeros((17, 56), dtype=torch.float32))
    module.rebuild_tensor_metadata = {}
    fused_a = torch.empty((2048, 7168), dtype=torch.float8_e4m3fn)
    fused_a_scale = torch.ones((16, 56), dtype=torch.float32)

    _copy_deepseek_v4_fused_a_weight_scale(module, fused_a, fused_a_scale)

    assert module.weight_scale.shape == (17, 56)
    assert torch.equal(module.weight_scale[:16], fused_a_scale)
    assert torch.equal(module.weight_scale[16], torch.zeros(56))


def test_deepseek_v4_kv_norm_keeps_full_head_dim():
    weights = {
        "layers.0.attn.kv_norm.weight": torch.arange(512, dtype=torch.float32),
    }

    remapped = _remap_deepseek_v4_checkpoint_keys(weights, num_hidden_layers=1, kv_lora_rank=448)

    tensor = remapped["model.layers.0.self_attn.kv_a_layernorm.weight"]
    assert tensor.shape == (512,)
    assert tensor[-1].item() == 511


def test_deepseek_v4_gate_bias_maps_to_score_correction_bias():
    weights = {
        "layers.0.ffn.gate.bias": torch.arange(4, dtype=torch.float32),
    }

    remapped = _remap_deepseek_v4_checkpoint_keys(weights, num_hidden_layers=1, kv_lora_rank=448)

    assert torch.equal(
        remapped["model.layers.0.mlp.gate.e_score_correction_bias"],
        weights["layers.0.ffn.gate.bias"],
    )


def test_deepseek_v4_gate_uses_fp32_reference_linear():
    if not torch.cuda.is_available():
        pytest.skip("dsv3_router_gemm_op requires CUDA")

    device = torch.device("cuda")
    gate = DeepseekV4Gate(
        hidden_size=4,
        num_experts=3,
        top_k=2,
        n_group=1,
        topk_group=1,
        routed_scaling_factor=1.0,
        is_hashed=False,
        dtype=torch.bfloat16,
        moe_backend="TRTLLM",
    ).to(device)
    hidden_states = torch.tensor([[1.0, -2.0, 3.0, -4.0]], dtype=torch.bfloat16, device=device)
    weight = torch.tensor(
        [[1.0, 2.0, 3.0, 4.0], [-1.0, 1.0, -1.0, 1.0], [0.5, -0.5, 0.25, -0.25]],
        dtype=torch.bfloat16,
        device=device,
    )
    gate.weight.copy_(weight)

    logits = gate(hidden_states)

    assert gate.e_score_correction_bias.dtype == torch.float32
    assert logits.dtype == torch.float32
    assert torch.equal(logits, torch.nn.functional.linear(hidden_states.float(), weight.float()))


def test_deepseek_v4_attn_sink_remap():
    weights = {
        "layers.0.attn.attn_sink": torch.arange(4, dtype=torch.float32),
    }

    remapped = _remap_deepseek_v4_checkpoint_keys(weights, num_hidden_layers=1, kv_lora_rank=448)

    assert torch.equal(
        remapped["model.layers.0.self_attn.attn_sink"], weights["layers.0.attn.attn_sink"]
    )

    # The MTP sink is re-prefixed to the extra layer index and routes through
    # the same self_attn loader branch (and thus load_attn_sink) as the main
    # layers.
    mtp_remapped = _remap_deepseek_v4_checkpoint_keys(
        {"mtp.0.attn.attn_sink": torch.arange(4, dtype=torch.float32)},
        num_hidden_layers=1,
        kv_lora_rank=448,
    )
    assert "model.layers.1.self_attn.attn_sink" in mtp_remapped


def test_deepseek_v4_flat_hc_weight_remap():
    weights = {
        "layers.0.hc_attn_fn": torch.tensor([1.0]),
        "layers.0.hc_ffn_scale": torch.tensor([2.0]),
        "hc_head_base": torch.tensor([3.0]),
    }

    remapped = _remap_deepseek_v4_checkpoint_keys(weights, num_hidden_layers=1, kv_lora_rank=448)

    assert torch.equal(remapped["model.layers.0.hc_attn_fn"], weights["layers.0.hc_attn_fn"])
    assert torch.equal(remapped["model.layers.0.hc_ffn_scale"], weights["layers.0.hc_ffn_scale"])
    assert torch.equal(remapped["model.hc_head_base"], weights["hc_head_base"])


def test_deepseek_v4_o_a_proj_scale_remap():
    weights = {
        "layers.0.attn.wo_a.weight": torch.zeros((8, 8), dtype=torch.float8_e4m3fn),
        "layers.0.attn.wo_a.scale": torch.ones((1, 1), dtype=torch.float32),
    }

    remapped = _remap_deepseek_v4_checkpoint_keys(weights, num_hidden_layers=1, kv_lora_rank=448)

    assert "model.layers.0.self_attn.o_a_proj" in remapped
    assert "model.layers.0.self_attn.o_a_proj.weight_scale_inv" in remapped


def test_deepseek_v4_q_b_layernorm_matches_per_head_reference():
    from tensorrt_llm._torch.modules.rms_norm import RMSNorm

    if not torch.cuda.is_available():
        pytest.skip("RMSNorm fast paths require CUDA")

    eps = 1e-6
    num_heads = 2
    head_dim = 4
    device = torch.device("cuda")
    norm = RMSNorm(
        hidden_size=head_dim, eps=eps, dtype=torch.bfloat16, device=device, has_weights=False
    )
    hidden_states = torch.arange(1, 17, dtype=torch.bfloat16, device=device).reshape(2, 8)

    output = norm(hidden_states.view(-1, head_dim)).view_as(hidden_states)

    ref = hidden_states.view(2, num_heads, head_dim)
    ref = ref * torch.rsqrt(ref.square().float().mean(dim=-1, keepdim=True) + eps).to(ref.dtype)
    torch.testing.assert_close(output, ref.reshape(2, 8), rtol=1e-2, atol=2e-2)
    assert list(norm.named_parameters()) == []


def test_deepseek_v4_q_b_layernorm_differs_from_joint_flat_rms():
    from tensorrt_llm._torch.modules.rms_norm import RMSNorm

    if not torch.cuda.is_available():
        pytest.skip("RMSNorm fast paths require CUDA")

    eps = 1e-6
    head_dim = 4
    device = torch.device("cuda")
    head_scales = torch.tensor([1.0, 10.0, 0.1, 1.0], dtype=torch.bfloat16, device=device)
    num_heads = head_scales.numel()
    base = torch.arange(1, 1 + 2 * num_heads * head_dim, dtype=torch.bfloat16, device=device).view(
        2, num_heads, head_dim
    )
    hidden_states = (base * head_scales.view(1, num_heads, 1)).reshape(2, num_heads * head_dim)

    per_head_norm = RMSNorm(
        hidden_size=head_dim, eps=eps, dtype=torch.bfloat16, device=device, has_weights=False
    )
    per_head = per_head_norm(hidden_states.view(-1, head_dim)).view_as(hidden_states)

    joint_norm = RMSNorm(
        hidden_size=num_heads * head_dim,
        eps=eps,
        dtype=torch.bfloat16,
        device=device,
        has_weights=False,
    )
    joint = joint_norm(hidden_states)

    assert not torch.allclose(per_head, joint, atol=0.1)


def test_deepseek_v4_mla_q_b_layernorm_init_and_forward_shape():
    from tensorrt_llm._torch.modules.mla import MLA

    init_src = inspect.getsource(MLA.__init__)
    helper_src = inspect.getsource(MLA._deepseek_v4_q_b_layernorm)
    forward_src = inspect.getsource(MLA.forward_impl_with_deepseek_v4)
    init_src_no_ws = "".join(init_src.split())

    assert "self.q_b_layernorm=RMSNorm(hidden_size=self.qk_head_dim" in init_src_no_ws
    assert "has_weights=False" in init_src
    assert "kv_a_layernorm_hidden_size = (" in init_src
    assert "self.kv_lora_rank + self.qk_rope_head_dim" in init_src
    assert "self.kv_a_layernorm=RMSNorm(hidden_size=kv_a_layernorm_hidden_size" in init_src_no_ws
    assert "q.dim() == 2" in helper_src
    assert "self.num_heads_tp * self.qk_head_dim" in helper_src
    assert "torch.ops.trtllm.deepseek_v4_q_norm" in helper_src
    assert ".is_cuda" not in helper_src
    assert ".is_contiguous" not in helper_src
    assert "q.dtype" not in helper_src
    assert "total_rows" not in helper_src
    assert "self.q_b_layernorm(" not in helper_src
    assert "self._deepseek_v4_q_b_layernorm(q_proj)" in _source_calls(forward_src)


def test_deepseek_v4_compressor_rotate_and_indexer_rope_contracts():
    assert inspect.signature(Compressor).parameters["rotate_activation"].default is False

    indexer_init = inspect.getsource(DeepseekV4Indexer.__init__)
    assert "is_neox=False" in indexer_init
    assert "rotate_activation=HAS_FAST_HADAMARD" in indexer_init

    attention_init = inspect.getsource(DeepseekV4TrtllmAttention.__init__)
    assert "rotate_activation=False" in attention_init


class _CudaMarkedTensor:
    """Proxy over a real CPU tensor that reports ``is_cuda=True``.

    The forward injection gate keys on ``attn_sink.data.is_cuda`` (the C++
    op takes the raw pointer without a device check), and these unit tests
    must stay runnable on CPU-only hosts.
    """

    is_cuda = True

    def __init__(self, tensor):
        object.__setattr__(self, "_tensor", tensor)

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_tensor"), name)


def test_deepseek_v4_attention_forward_injects_attn_sink(monkeypatch):
    captured = {}

    def fake_forward(self, *args, **kwargs):
        captured.update(kwargs)
        return "ok"

    monkeypatch.setattr(TrtllmAttention, "forward", fake_forward)
    monkeypatch.setattr(
        DeepseekV4TrtllmAttention,
        "_prepare_sparse_forward_args",
        lambda self, metadata, forward_args: None,
    )
    attn = object.__new__(DeepseekV4TrtllmAttention)
    sink = torch.ones(4, dtype=torch.float32)
    attn.attn_sink = SimpleNamespace(data=_CudaMarkedTensor(sink))

    metadata = object()
    assert DeepseekV4TrtllmAttention.forward(attn, "q", None, None, metadata) == "ok"
    assert "attention_sinks" not in captured
    assert captured["forward_args"].attention_sinks.data_ptr() == sink.data_ptr()

    captured.clear()
    forward_args = AttentionForwardArgs()
    assert (
        DeepseekV4TrtllmAttention.forward(
            attn, "q", None, None, metadata, forward_args=forward_args
        )
        == "ok"
    )
    assert "attention_sinks" not in captured
    assert captured["forward_args"].attention_sinks.data_ptr() == sink.data_ptr()
    assert forward_args.attention_sinks is None

    # Device gate: a still-on-CPU pre-registered sink (hook-less harnesses
    # that never ran the owning module's staged post-load hooks,
    # post_load_weights / cache_derived_state) must NOT be injected --
    # attentionOp.cpp reads the raw pointer without a device check, so a
    # host pointer would reach the CUDA kernel. Null-sink behavior is the
    # correct fallback.
    captured.clear()
    attn.attn_sink = torch.nn.Parameter(sink, requires_grad=False)
    assert DeepseekV4TrtllmAttention.forward(attn, "q", None, None, metadata) == "ok"
    assert captured["forward_args"].attention_sinks is None


def _make_attention_stub(num_heads_tp=4):
    """Bare DeepseekV4Attention carrying only the attn_sink surface (the
    real __init__ needs a GPU-backed MLA tree). ``_weights_transformed`` is
    pre-set so MLA's transform chain is a no-op on the stub."""
    stub = object.__new__(DeepseekV4Attention)
    torch.nn.Module.__init__(stub)
    stub.num_heads_tp = num_heads_tp
    stub.mqa = SimpleNamespace()
    stub._weights_transformed = True
    stub._register_attn_sink()
    return stub


def test_deepseek_v4_attention_module_owns_attn_sink():
    """attn_sink must be a Parameter of the DSv4-owned nn.Module.

    Module ownership is what makes it transportable: only
    module._parameters entries enter state_dict()/named_parameters(), which
    is exactly what the GMS writer commit (finalize_gms_write /
    _iter_module_tensors), GMS read-only materialization, and dummy init
    enumerate. A backend-held Parameter is invisible to all of them. It
    must also exist with stable storage at construction time: CUDA-graph
    warmup capture happens at LLM init, before any RLHF update_weights, and
    -inf is the neutral sink (kernels add exp(sink - max) to the softmax
    denominator)."""
    stub = _make_attention_stub(num_heads_tp=64)
    parent = torch.nn.Module()
    parent.self_attn = stub

    sink = stub.attn_sink
    assert isinstance(sink, torch.nn.Parameter)
    assert sink.requires_grad is False
    assert sink.dtype == torch.float32
    assert tuple(sink.shape) == (64,)
    assert torch.isneginf(sink.data).all()
    assert stub._attn_sink_loaded is False
    assert "self_attn.attn_sink" in parent.state_dict()
    assert "self_attn.attn_sink" in dict(parent.named_parameters())
    # The backend alias shares the Parameter object, so loader copy_ and
    # forward injection stay in sync.
    assert stub.mqa.attn_sink is stub.attn_sink

    # __init__ must register unconditionally (warmup capture needs the
    # storage before any reload could create it lazily).
    init_src = inspect.getsource(DeepseekV4Attention.__init__)
    assert "self._register_attn_sink()" in init_src


def test_deepseek_v4_cache_derived_state_rewires_rebound_attn_sink(monkeypatch):
    """GMS read-only materialization REBINDS module._parameters["attn_sink"]
    to a new Parameter bound zero-copy to writer-committed shared storage
    (the CPU-built param never sees model.to("cuda") on the GMS path), so
    cache_derived_state must re-wire the backend alias to the transported
    Parameter and must NOT write it: the receiver's local _attn_sink_loaded
    flag is still False while the storage holds real values shared with the
    writer and every peer."""
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self: self)
    stub = _make_attention_stub()
    values = torch.tensor([0.5, -1.0, 2.0, 0.0], dtype=torch.float32)
    transported = torch.nn.Parameter(values.clone(), requires_grad=False)
    ptr = transported.data_ptr()
    # Mirror of the upstream materializer's rebind statement.
    stub._parameters["attn_sink"] = transported

    DeepseekV4Attention.cache_derived_state(stub)

    assert stub.mqa.attn_sink is transported
    assert torch.equal(transported.data, values)  # no -inf refill
    assert transported.data_ptr() == ptr
    assert stub._attn_sink_loaded is False


def test_deepseek_v4_attn_sink_stage_preservation_semantics(monkeypatch):
    """Both staged hooks are VALUE-PRESERVING for attn_sink. Dummy init must
    skip it at the source (suffix skip-list in initialize_dummy_weights;
    exp(-inf) == 0 is the exact no-sink math) while still randomizing
    sibling params. Direct-write transports (MX P2P) deliver real values
    with _attn_sink_loaded still False, and repeated finalize sweeps must
    preserve them without rebinding .data (captured graphs bake the device
    pointer). cache_derived_state (read-only stage) never writes either."""
    if not torch.cuda.is_available():
        # The one-time CUDA migration is exercised on GPU runners; keep the
        # value/flag/pointer contract testable on CPU-only hosts.
        monkeypatch.setattr(torch.Tensor, "cuda", lambda self: self)

    stub = _make_attention_stub()
    parent = torch.nn.Module()
    parent.self_attn = stub
    parent.probe = torch.nn.Parameter(torch.zeros(4), requires_grad=False)

    # (a) dummy init: the ctor -inf survives via the source-level skip while
    # a sibling fp32 Parameter IS randomized (the skip is suffix-scoped).
    initialize_dummy_weights(parent)
    assert torch.isneginf(stub.attn_sink.data).all()
    assert not torch.equal(parent.probe.data, torch.zeros(4))
    DeepseekV4Attention.post_load_weights(stub)
    assert torch.isneginf(stub.attn_sink.data.cpu()).all()
    if torch.cuda.is_available():
        assert stub.attn_sink.data.is_cuda
    ptr = stub.attn_sink.data_ptr()

    # (b) MX-P2P-style direct Parameter write: real values land WITHOUT
    # load_attn_sink (diagnostic flag stays False); the full post-load walk
    # must preserve them across repeated sweeps, pointer stable. This is
    # the direct regression test for the flag-gated refill.
    transported = torch.tensor([0.5, -1.0, 2.0, 0.0], dtype=torch.float32)
    stub.attn_sink.data.copy_(transported)
    assert stub._attn_sink_loaded is False
    DeepseekV4Attention.post_load_weights(stub)
    DeepseekV4Attention.post_load_weights(stub)  # finalize sweeps repeat
    assert torch.equal(stub.attn_sink.data.cpu(), transported)
    assert stub.attn_sink.data_ptr() == ptr
    assert stub.mqa.attn_sink is stub.attn_sink

    # (c) loader-delivered values behave identically (flag is diagnostic
    # only, it gates nothing).
    loaded = torch.tensor([1.5, 2.5, 3.5, 4.5], dtype=torch.float32)
    stub.attn_sink.data.copy_(loaded)
    stub._attn_sink_loaded = True
    DeepseekV4Attention.post_load_weights(stub)
    assert torch.equal(stub.attn_sink.data.cpu(), loaded)
    assert stub.attn_sink.data_ptr() == ptr

    # (d) the read-only stage must not write even with the flag unset and
    # non-neutral values (RO receivers hold writer-transported real values).
    stub_ro = _make_attention_stub()
    stub_ro.attn_sink.data.copy_(transported)
    DeepseekV4Attention.cache_derived_state(stub_ro)
    assert torch.equal(stub_ro.attn_sink.data.cpu(), transported)


def test_deepseek_v4_attn_sink_ctor_value_survives_meta_init_mode():
    """Executable contract for what _register_attn_sink relies on:
    MetaInitMode intercepts only aten.empty/empty_like, so torch.full (no
    tensor args) executes for real on CPU -- the -inf ctor value exists as
    real bytes that later allocation stages must preserve."""
    from tensorrt_llm._torch.models.modeling_utils import MetaInitMode

    with MetaInitMode():
        sink = torch.nn.Parameter(
            torch.full((4,), float("-inf"), dtype=torch.float32), requires_grad=False
        )
        probe = torch.empty(4)

    assert sink.device.type == "cpu"
    assert torch.isneginf(sink).all()
    assert probe.device.type == "meta"  # the mode WAS active around both


def test_initialize_dummy_weights_skips_attn_sink_suffix():
    """Shared-layer contract for the dummy initializer: the ``.attn_sink``
    suffix skip preserves ctor values for ANY module registering the param
    (the neutral value is the correct dummy sink), while suffix scoping
    keeps the skip from swallowing similarly named params."""
    model = torch.nn.Module()
    inner = torch.nn.Module()
    model.self_attn = inner
    inner.attn_sink = torch.nn.Parameter(
        torch.full((3,), float("-inf"), dtype=torch.float32), requires_grad=False
    )
    inner.attn_sink_bias = torch.nn.Parameter(torch.zeros(3), requires_grad=False)
    model.weight = torch.nn.Parameter(torch.zeros(3), requires_grad=False)

    initialize_dummy_weights(model)

    assert torch.isneginf(inner.attn_sink.data).all()
    assert not torch.equal(inner.attn_sink_bias.data, torch.zeros(3))
    assert not torch.equal(model.weight.data, torch.zeros(3))


def test_deepseek_v4_attn_sink_reached_by_model_loader_walks(monkeypatch):
    """The GMS read-only loader branch runs ONLY ModelLoader._walk_cache_state
    and the full/dummy branches run _walk_full_post_load: with the round-9
    model-level backend sweep gone, both real walks must reach the
    module-level staged hooks (rebind re-wire on the cache walk,
    value-preserving migration + re-wire on the full walk)."""
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self: self)
    root = _make_causal_lm_stub()
    attn = _make_attention_stub()
    root.attn = attn

    rebound = torch.nn.Parameter(
        torch.tensor([9.0, 8.0, 7.0, 6.0], dtype=torch.float32), requires_grad=False
    )
    attn._parameters["attn_sink"] = rebound  # upstream RO rebind

    ModelLoader._walk_cache_state(root)
    # Read-only stage: alias re-wired to the transported Parameter, values
    # untouched.
    assert attn.mqa.attn_sink is rebound
    assert torch.equal(rebound.data, torch.tensor([9.0, 8.0, 7.0, 6.0]))

    ModelLoader._walk_full_post_load(root)
    # Writer stage is value-preserving too: directly transported values
    # (diagnostic flag never set) survive the full walk -- no -inf refill.
    assert torch.equal(attn.attn_sink.data, torch.tensor([9.0, 8.0, 7.0, 6.0]))
    assert attn.attn_sink is rebound
    assert attn.mqa.attn_sink is attn.attn_sink


def test_deepseek_v4_load_attn_sink_refreshes_in_place():
    """Reload must copy_ into the module-owned pre-registered storage (never
    rebind: the captured CUDA graph baked this pointer), apply the per-head
    TP split, and latch _attn_sink_loaded on the owner module."""
    loader = object.__new__(DeepseekV4WeightLoader)
    loader.model_config = SimpleNamespace(
        mapping=SimpleNamespace(enable_attention_dp=False, tp_size=2, tp_rank=1)
    )
    owner = _make_attention_stub(num_heads_tp=8)
    ptr = owner.attn_sink.data_ptr()

    sink_src = torch.arange(16, dtype=torch.float32)
    loader.load_attn_sink(owner, sink_src)
    assert owner.attn_sink.data_ptr() == ptr
    assert torch.equal(owner.attn_sink.data, sink_src[8:])
    assert owner._attn_sink_loaded is True
    # The backend alias observes the refresh through the shared object.
    assert owner.mqa.attn_sink is owner.attn_sink
    assert torch.equal(owner.mqa.attn_sink.data, sink_src[8:])

    # Shape drift across reloads must fail loudly, not corrupt storage.
    with pytest.raises(ValueError, match="attn_sink shape changed"):
        loader.load_attn_sink(owner, torch.arange(8, dtype=torch.float32))
    assert torch.equal(owner.attn_sink.data, sink_src[8:])

    # Under attention DP (tp_size == 1 mapping) the full sink is kept.
    loader.model_config = SimpleNamespace(
        mapping=SimpleNamespace(enable_attention_dp=True, tp_size=1, tp_rank=0)
    )
    owner_dp = _make_attention_stub(num_heads_tp=16)
    loader.load_attn_sink(owner_dp, sink_src)
    assert torch.equal(owner_dp.attn_sink.data, sink_src)


def _make_causal_lm_stub():
    """Bare DeepseekV4ForCausalLM without a GPU-backed model tree (the real
    __init__ needs one). ``config`` is a read-only property forwarding to
    ``model_config.pretrained_config``.
    """
    stub = object.__new__(DeepseekV4ForCausalLM)
    torch.nn.Module.__init__(stub)
    stub.model_config = SimpleNamespace(pretrained_config=SimpleNamespace(num_hidden_layers=0))
    stub.model = SimpleNamespace(layers=[])
    return stub
