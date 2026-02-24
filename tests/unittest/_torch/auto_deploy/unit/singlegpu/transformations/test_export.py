from abc import ABC, abstractmethod
from typing import Optional, Set, Type

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from _model_test_utils import MLP
from _torch_test_utils import all_close
from torch.export import Dim, export
from torch.fx import GraphModule

from tensorrt_llm._torch.auto_deploy.export import torch_export_to_gm


def _torch_export_non_strict(model, *args, **kwargs):
    kwargs["strict"] = False
    return export(model, *args, **kwargs)


class ModuleForExport(ABC, nn.Module):
    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def get_sample_input(self):
        pass

    def check_xfail(self, f_export, use_dynamic_shape, device) -> bool:
        return False

    @abstractmethod
    def get_dynamic_shapes(self):
        pass


class MLPForExport(ModuleForExport):
    def __init__(self):
        super().__init__()
        self.mlp = MLP(10, 10, 10)

    def forward(self, x):
        return self.mlp(x)

    def get_sample_input(self):
        return torch.randn(2, 10)

    def get_dynamic_shapes(self):
        return {0: Dim.DYNAMIC}


class MLPDuplicate(ModuleForExport):
    def __init__(self, device: Optional[str] = None):
        super(MLPDuplicate, self).__init__()
        self.fc1 = nn.Linear(10, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, 10)
        self.to(device)
        self.fc3.weight = self.fc1.weight

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

    def get_sample_input(self):
        return torch.randn(2, 10, device=self.fc1.weight.device)

    def get_deduplicated_keys(self) -> Set[str]:
        return {"fc3.weight"}

    def get_dynamic_shapes(self):
        return {0: Dim.DYNAMIC}


class ModuleWithWhere(ModuleForExport):
    def where(self, mask):
        return torch.where(mask)

    def forward(self, x):
        is_positive = x > 0
        output = torch.zeros_like(x)
        idx1, idx2 = self.where(is_positive)
        output[idx1, idx2] = x[idx1, idx2]
        return output

    def get_sample_input(self):
        return torch.randn(2, 10)

    def get_dynamic_shapes(self):
        return {0: Dim.DYNAMIC}

    def check_xfail(self, f_export, use_dynamic_shape, device) -> bool:
        return (
            use_dynamic_shape and f_export in [export, _torch_export_non_strict]
        ) or device == "meta"


class ModuleWithNonzero(ModuleWithWhere):
    def where(self, mask):
        return torch.nonzero(mask, as_tuple=True)

    def check_xfail(self, f_export, use_dynamic_shape, device):
        return device == "meta"


class ModuleWithRouting(ModuleForExport):
    def __init__(self):
        super().__init__()
        self.num_experts = 8
        self.seq_len = 32
        self.top_k = 2

    def forward(self, x):
        _, selected_experts = torch.topk(x, self.top_k, dim=-1)
        expert_mask = F.one_hot(selected_experts, self.num_experts).permute(2, 1, 0)
        out = torch.zeros_like(x)

        for idx_e in range(self.num_experts):
            _, top_x = torch.where(expert_mask[idx_e])

            out[top_x] += x[top_x]

            return out

    def get_sample_input(self):
        return torch.randn(self.seq_len, self.num_experts)

    def get_dynamic_shapes(self):
        return {0: Dim.DYNAMIC}

    def check_xfail(self, f_export, use_dynamic_shape, device) -> bool:
        return (
            use_dynamic_shape and f_export in [export, _torch_export_non_strict]
        ) or device == "meta"


class ModuleWithModuleList(ModuleForExport):
    def __init__(self, device: Optional[str] = None, num_active_layers: int = 2):
        super().__init__()
        self.fcs = nn.ModuleList([nn.Linear(10, 10) for _ in range(4)])
        self.num_active_layers = num_active_layers
        self.to(device)

    def forward(self, x):
        for fc in self.fcs[: self.num_active_layers]:
            x = fc(x)
        return x

    def get_sample_input(self):
        return torch.randn(2, 10, device=self.fcs[0].weight.device)

    def get_dynamic_shapes(self):
        return {0: Dim.DYNAMIC}

    def check_xfail(self, f_export, use_dynamic_shape, device) -> bool:
        # non-strict mode only works with our hack in torch_export_to_gm
        return f_export in [_torch_export_non_strict]


@pytest.mark.parametrize(
    "f_export",
    [torch.export.export, export, _torch_export_non_strict, torch_export_to_gm],
)
@pytest.mark.parametrize("use_dynamic_shape", [True, False])
@pytest.mark.parametrize("device", ["cpu", "cuda", "meta"])
@pytest.mark.parametrize(
    "mod_cls",
    [
        MLPForExport,
        MLPDuplicate,
        ModuleWithWhere,
        ModuleWithNonzero,
        ModuleWithRouting,
        ModuleWithModuleList,
    ],
)
def test_module_export(f_export, mod_cls, device, use_dynamic_shape):
    mod = mod_cls().to(device)
    if mod.check_xfail(f_export, use_dynamic_shape, device):
        pytest.xfail(
            f"{mod_cls.__name__} expected to fail with: {f_export.__name__}, "
            f"use_dynamic_shape={use_dynamic_shape}, device={device}"
        )

    x = mod.get_sample_input().to(device)

    # run export now
    if use_dynamic_shape:
        dynamic_shapes = mod.get_dynamic_shapes()
        ep = f_export(mod, (x,), dynamic_shapes=(dynamic_shapes,))
    else:
        ep = f_export(mod, (x,))

    if not isinstance(ep, GraphModule):
        ep = ep.module()

    y = mod(x)
    y_ep = ep(x)

    if device != "meta":
        assert all_close(y, y_ep)

    print(ep.graph)


@pytest.mark.parametrize("model_cls", [MLPDuplicate])
@pytest.mark.parametrize("device_export", ["cpu"])  # TODO: investigate meta device error
def test_deduplicate_during_export(model_cls: Type[nn.Module], device_export: str):
    model = model_cls(device=device_export)
    model.eval()
    t_in = model.get_sample_input()

    # run export now
    gm = torch_export_to_gm(model, (t_in,))

    # check that deduplicated keys are removed from state_dict
    assert gm.state_dict().keys() == model.state_dict().keys() - model.get_deduplicated_keys()

    def check_parameter_loading(param_to_pop: str, expected_param: str):
        """Helper function to check parameter loading behavior.

        Args:
            param_to_pop: The parameter to remove from state dict before loading
            expected_param: The parameter whose value should be loaded into fc1.weight
        """
        sd_og = model.state_dict()
        sd_og.pop(param_to_pop)
        gm.load_state_dict(sd_og)

        if device_export != "meta":
            assert torch.equal(gm.fc1.weight, sd_og[expected_param])

    # fc3.weight is aliased to fc1.weight
    # Test loading fc3.weight into gm.fc1.weight. State dict does not contain fc1.weight
    check_parameter_loading("fc1.weight", "fc3.weight")

    # Test loading fc1.weight into gm.fc1.weight. State dict does not contain fc3.weight
    check_parameter_loading("fc3.weight", "fc1.weight")


# ---------------------------------------------------------------------------
# MOE export with reduced experts
# ---------------------------------------------------------------------------

# Ensure the auto_deploy::torch_moe custom op is registered
import tensorrt_llm._torch.auto_deploy.custom_ops  # noqa: F401, E402


class SimpleMoEForExport(nn.Module):
    """A minimal MOE model using the ``auto_deploy::torch_moe`` custom op.

    The *expert_attr_name* parameter controls the attribute under which the
    expert ``nn.ModuleList`` is stored.  This lets tests verify that the
    probe-based expert discovery does **not** rely on the name ``"experts"``.
    """

    def __init__(
        self,
        num_experts: int = 8,
        hidden_dim: int = 16,
        inter_dim: int = 32,
        expert_attr_name: str = "experts",
    ):
        super().__init__()
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        self.expert_attr_name = expert_attr_name
        expert_list = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "gate_proj": nn.Linear(hidden_dim, inter_dim, bias=False),
                        "down_proj": nn.Linear(inter_dim, hidden_dim, bias=False),
                        "up_proj": nn.Linear(hidden_dim, inter_dim, bias=False),
                    }
                )
                for _ in range(num_experts)
            ]
        )
        # Store under a configurable attribute name so tests can verify
        # that the probe does NOT rely on the name being "experts".
        setattr(self, expert_attr_name, expert_list)
        self.gate = nn.Linear(hidden_dim, num_experts, bias=False)

    @property
    def _expert_list(self) -> nn.ModuleList:
        return getattr(self, self.expert_attr_name)

    def forward(self, x):
        experts = self._expert_list
        router_logits = self.gate(x)
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, 2, dim=-1)
        routing_weights = routing_weights.to(x.dtype)
        return torch.ops.auto_deploy.torch_moe(
            x,
            selected_experts,
            routing_weights,
            w1_weight=[e["gate_proj"].weight for e in experts],
            w2_weight=[e["down_proj"].weight for e in experts],
            w3_weight=[e["up_proj"].weight for e in experts],
        )


@pytest.mark.parametrize("expert_attr_name", ["experts", "mlp_bank"])
@pytest.mark.parametrize("num_experts", [4, 8])
@pytest.mark.parametrize("num_moe_experts_for_export", [1, 2])
@pytest.mark.parametrize("device", ["cuda"])
def test_moe_export_with_reduced_experts(
    num_experts, num_moe_experts_for_export, device, expert_attr_name
):
    """Export with fewer experts then expand — result must match full export."""
    mod = SimpleMoEForExport(num_experts=num_experts, expert_attr_name=expert_attr_name).to(device)
    mod.eval()
    x = torch.randn(4, mod.hidden_dim, device=device)

    # Full export (baseline)
    gm_full = torch_export_to_gm(mod, (x,))

    # Export with reduced experts
    gm_reduced = torch_export_to_gm(
        mod,
        (x,),
        num_moe_experts_for_export=num_moe_experts_for_export,
    )

    # --- structural check: both graphs must have the right expert count ---
    def _count_moe_experts(gm):
        for node in gm.graph.nodes:
            if node.op == "call_function" and "torch_moe" in str(node.target):
                return len(node.args[3])  # w1_weight list length
        return 0

    assert _count_moe_experts(gm_full) == num_experts
    assert _count_moe_experts(gm_reduced) == num_experts

    # --- numerical check: outputs must match ---
    y_full = gm_full(x)
    y_reduced = gm_reduced(x)
    assert all_close(y_full, y_reduced), "Reduced-expert export output differs from full export"

    # --- state-dict round-trip: loading the original weights must work ---
    sd = mod.state_dict()
    gm_reduced.load_state_dict(sd, strict=False)
    y_loaded = gm_reduced(x)
    assert all_close(y_loaded, y_full), "Output after state-dict reload differs"

    # --- verify source model is unmodified ---
    assert len(mod._expert_list) == num_experts, "Source model experts were not restored"


# ---------------------------------------------------------------------------
# Real-model MOE export: GLM4 MoE Lite
# ---------------------------------------------------------------------------

from tensorrt_llm._torch.auto_deploy.models.custom.modeling_glm4_moe_lite import (  # noqa: E402
    Glm4MoeLiteConfig,
    Glm4MoeLiteForCausalLM,
)


def _make_tiny_glm4_config(n_routed_experts: int = 8) -> Glm4MoeLiteConfig:
    """Create a minimal ``Glm4MoeLiteConfig`` suitable for unit tests."""
    return Glm4MoeLiteConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        q_lora_rank=32,
        kv_lora_rank=32,
        qk_nope_head_dim=12,
        qk_rope_head_dim=4,
        v_head_dim=16,
        n_routed_experts=n_routed_experts,
        n_shared_experts=1,
        num_experts_per_tok=2,
        moe_intermediate_size=64,
        n_group=1,
        topk_group=1,
        routed_scaling_factor=1.0,
        norm_topk_prob=True,
        first_k_dense_replace=1,  # layer 0 = dense MLP, layer 1 = MoE
        max_position_embeddings=128,
        rope_scaling=None,
        pad_token_id=0,
    )


def _count_moe_experts_in_graph(gm: GraphModule) -> int:
    """Return the number of experts in the first ``torch_moe`` call in *gm*."""
    for node in gm.graph.nodes:
        if node.op == "call_function" and "torch_moe" in str(node.target):
            return len(node.args[3])  # w1_weight list length
    return 0


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="GLM4 MoE Lite requires CUDA (uses noaux_tc_op)"
)
@pytest.mark.parametrize("n_routed_experts", [8, 16])
@pytest.mark.parametrize("num_moe_experts_for_export", [2])
def test_glm4_moe_lite_export_with_reduced_experts(n_routed_experts, num_moe_experts_for_export):
    """Export a tiny ``Glm4MoeLiteForCausalLM`` with reduced experts and verify
    that the expanded graph has the correct structure and accepts the original
    state dict.
    """
    # GLM4 MoE Lite uses noaux_tc_op which is CUDA-only, so we must use CUDA device
    device = "cuda"
    config = _make_tiny_glm4_config(n_routed_experts=n_routed_experts)
    model = Glm4MoeLiteForCausalLM(config).to(device)
    model.eval()

    input_ids = torch.randint(0, config.vocab_size, (1, 8), device=device)
    position_ids = torch.arange(8, device=device).unsqueeze(0)
    sample_kwargs = {"input_ids": input_ids, "position_ids": position_ids}

    # --- full export (baseline) ---
    gm_full = torch_export_to_gm(model, kwargs=sample_kwargs)

    # --- export with reduced experts ---
    gm_reduced = torch_export_to_gm(
        model,
        kwargs=sample_kwargs,
        num_moe_experts_for_export=num_moe_experts_for_export,
    )

    # Structural: both graphs must expose all experts
    assert _count_moe_experts_in_graph(gm_full) == n_routed_experts
    assert _count_moe_experts_in_graph(gm_reduced) == n_routed_experts

    # State-dict keys must match between full and reduced exports
    full_keys = set(gm_full.state_dict().keys())
    reduced_keys = set(gm_reduced.state_dict().keys())
    assert full_keys == reduced_keys, (
        f"State-dict key mismatch.\n"
        f"  Only in full: {full_keys - reduced_keys}\n"
        f"  Only in reduced: {reduced_keys - full_keys}"
    )

    # Load the original model weights into the reduced export graph
    gm_reduced.load_state_dict(model.state_dict(), strict=False)

    # Source model must be fully restored
    for name, mod in model.named_modules():
        if hasattr(mod, "experts") and isinstance(mod.experts, nn.ModuleList):
            assert len(mod.experts) == n_routed_experts, (
                f"Expert list in '{name}' was not restored to {n_routed_experts}"
            )
