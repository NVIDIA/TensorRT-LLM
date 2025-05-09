import enum
from typing import Dict, List

import torch
from torch.nn.parameter import Parameter

from ..utils import Fp4QuantizedTensor

E2M1_MAX = 6.0


class TensorParallelMode(str, enum.Enum):
    COLUMN = 'column'
    ROW = 'row'

    @classmethod
    def split_dim(cls, mode):
        return 1 if mode == cls.ROW else 0


class LinearBaseQuant:

    def __init__(self, quant_config, device):
        self.quant_config = quant_config
        self.device = device

    def __call__(self):
        raise "__call__ is not implemented."

    def _load_weight_for_name(self, weights: List[Dict], tensor_name):
        scale = None
        for w in weights:
            if tensor_name in w:
                if scale is None:
                    scale = w[tensor_name][...]
                else:
                    assert scale == w[tensor_name][
                        ...], "The scale should be same for all the weights"
        return scale

    def load_weight(self, weights, tensor_names):
        raise "load_weight is not implemented."

    def _copy(dst: Parameter, src: torch.Tensor):
        # TODO check that is it a reasonable change or not
        if dst.dtype != src.dtype:
            src = src.to(dst.dtype)
        assert dst.dtype == src.dtype, f"Incompatible dtype. dst: {dst.dtype}, src: {src.dtype}"
        dst.data.copy_(src)


class LinearQDQ(LinearBaseQuant):

    def __init__(self, quant_config, device):
        super().__init__(quant_config, device)
        assert quant_config.layer_quant_mode.has_fp8_qdq(
        ), "LinearQDQ only support fp8"
        self.scale = Parameter(torch.tensor(1.,
                                            dtype=torch.float32,
                                            device=device),
                               requires_grad=False)
        self.inv_scale = Parameter(torch.tensor(1.,
                                                dtype=torch.float32,
                                                device=device),
                                   requires_grad=False)

    def __call__(self):
        if input.dtype != torch.float8_e4m3fn:
            qinput, _ = torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor(
                input, self.scale)
        else:
            qinput = input
        return qinput, self.scale

    def load_weight(self, weights, tensor_name):
        scale = self._load_weight_for_name(weights, tensor_name)
        self._copy(self.scale, scale[0])
        self.inv_scale.data = 1.0 / self.scale


class LinearBlockScalesQuant(LinearBaseQuant):

    def __init__(self, quant_config, device):
        super().__init__(quant_config, device)
        self.scale = Parameter(torch.tensor(1.,
                                            dtype=torch.float32,
                                            device=device),
                               requires_grad=False)
        self.inv_scale = Parameter(torch.tensor(1.,
                                                dtype=torch.float32,
                                                device=device),
                                   requires_grad=False)

    def __call__(self, input):
        if input.dtype == torch.float8_e4m3fn:
            input = input.to(torch.bfloat16) * self.scale
        assert input.dtype == torch.bfloat16

        return torch.ops.trtllm.fp8_quantize_1x128(input)

    def load_weight(self, weights, tensor_name):
        scale = self._load_weight_for_name(weights, tensor_name)
        self._copy(self.scale, scale[0])
        self.inv_scale.data = 1.0 / self.scale


class LinearNVFP4(LinearBaseQuant):

    def __init__(self, quant_config, device):
        super().__init__(quant_config, device)
        self.scaling_vector_size = 16

        # FP32 per-tensor global scaling factor = 448*6/amax_input
        self.scale = Parameter(torch.empty([1],
                                           dtype=torch.float32,
                                           device=device),
                               requires_grad=False)
        self.inv_scale = Parameter(torch.empty([1],
                                               dtype=torch.float32,
                                               device=device),
                                   requires_grad=False)

    def __call__(self):
        if isinstance(input, Fp4QuantizedTensor):
            return input.fp4_tensor, input.scaling_factor
        else:
            return torch.ops.trtllm.fp4_quantize(input, self.scale,
                                                 self.scaling_vector_size,
                                                 False)

    def load_weight(self, weights, tensor_name):
        scale = self._load_weight_for_name(weights, tensor_name)
        scale = 1.0 / scale
        self._copy(self.scale, scale)
        self.inv_scale.data = self.scale / E2M1_MAX


class LinearQuant(LinearBaseQuant):

    def __init__(self, quant_config, device):
        super().__init__()
        self.quant_config = quant_config
        quant_mode = self.quant_config.quant_mode if self.quant_config else None
        # TODO: need to make src/target dtype to be parameters.
        if quant_mode:
            if quant_mode.has_fp8_qdq():
                self._quant = LinearQDQ(self.quant_config, device)
            elif quant_mode.has_nvfp4():
                self._quant = LinearNVFP4(self.quant_config, device)
            elif quant_mode.has_fp8_block_scales():
                self._quant = LinearBlockScalesQuant(self.quant_config, device)

    def __call__(self, input):
        return self._quant(input)

    def load_weight(self, weights):
        self._quant.load_weight(weights)
