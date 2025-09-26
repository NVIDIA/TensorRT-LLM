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
        return {0: Dim("batch_size", max=100)}


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
        return {0: Dim("batch_size", max=100)}


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
        return {0: Dim("batch_size", max=100)}

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
        return {0: Dim("seq_len", max=100)}

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
        return {0: Dim("batch_size", max=100)}

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
