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


class BaseQuant:

    def __init__(self, quant_config, device=torch.device('cpu')):
        super().__init__()
        self.quant_config = quant_config
        self.device = device

    def __call__(self, input):
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

    def load_weights(self,
                     weights,
                     tensor_name,
                     device: torch.device = torch.device('cpu')):
        raise "load_weights is not implemented."

    def load_weights_customized(self,
                                 weights,
                                 loader,
                                 device: torch.device = torch.device('cpu'),
                                 **kwargs):
        raise "load_weights_customized is not implemented."

    def _copy(self, dst: Parameter, src: torch.Tensor):
        # TODO check that is it a reasonable change or not
        if dst.dtype != src.dtype:
            src = src.to(dst.dtype)
        assert dst.dtype == src.dtype, f"Incompatible dtype. dst: {dst.dtype}, src: {src.dtype}"
        dst.data.copy_(src)


class NoopQuant(BaseQuant):

    def __init__(self, quant_config):
        super().__init__(quant_config)
        self.scale = Parameter(torch.tensor(1., dtype=torch.float32),
                               requires_grad=False)

    def __call__(self, input):
        return input, self.scale

    def load_weights(self,
                     weights,
                     tensor_name,
                     device: torch.device = torch.device('cpu')):
        scale = self._load_weight_for_name(weights, tensor_name)
        if scale is None:
            self.scale = None
            return
        self._copy(self.scale, scale)
        self.inv_scale.data = 1.0 / self.scale
        self.scale = self.scale.to(device)
        self.inv_scale = self.inv_scale.to(device)

    def load_weights_customized(self,
                                 weights,
                                 loader,
                                 device: torch.device = torch.device('cpu'),
                                 **kwargs):
        scale = loader(weights)
        self._copy(self.scale, scale)
        self.inv_scale.data = 1.0 / self.scale
        self.scale = self.scale.to(device)
        self.inv_scale = self.inv_scale.to(device)


class QDQ(BaseQuant):

    def __init__(self, quant_config):
        super().__init__(quant_config)
        assert quant_config.layer_quant_mode.has_fp8_qdq(
        ), "QDQ only support fp8"
        self.scale = Parameter(torch.tensor(1., dtype=torch.float32),
                               requires_grad=False)
        self.inv_scale = Parameter(torch.tensor(1., dtype=torch.float32),
                                   requires_grad=False)

    def __call__(self, input):
        if input.dtype != torch.float8_e4m3fn:
            if self.input_scale is not None:
                qinput, _ = torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor(
                    input, self.scale)
            else:
                # Dynamic quantization
                qinput, cur_input_scale = torch.ops.tensorrt_llm.quantize_e4m3_per_tensor(
                    input)
                self.scale = cur_input_scale.to(torch.float32)
        else:
            qinput = input
        return qinput, self.scale

    def load_weights(self,
                     weights,
                     tensor_name,
                     device: torch.device = torch.device('cpu')):
        scale = self._load_weight_for_name(weights, tensor_name)
        if scale is None:
            self.scale = None
            return
        self._copy(self.scale, scale)
        self.inv_scale.data = 1.0 / self.scale
        self.scale = self.scale.to(device)
        self.inv_scale = self.inv_scale.to(device)

    def load_weights_customized(self,
                                 weights,
                                 loader,
                                 device: torch.device = torch.device('cpu'),
                                 **kwargs):
        self.scale, self.inv_scale = loader(weights)
        self.scale = self.scale.to(device)
        self.inv_scale = self.inv_scale.to(device)


class BlockScalesQuant(BaseQuant):

    def __init__(self, quant_config):
        super().__init__(quant_config)
        self.scale = Parameter(torch.tensor(1., dtype=torch.float32),
                               requires_grad=False)
        self.inv_scale = Parameter(torch.tensor(1., dtype=torch.float32),
                                   requires_grad=False)

    def __call__(self, input):
        if self.scale is None:
            return input, self.scale
        if input.dtype == torch.float8_e4m3fn:
            input = input.to(torch.bfloat16) * self.scale
        assert input.dtype == torch.bfloat16

        return torch.ops.trtllm.fp8_quantize_1x128(input)

    def load_weights(self,
                     weights,
                     tensor_name,
                     device: torch.device = torch.device('cpu')):
        scale = self._load_weight_for_name(weights, tensor_name)
        if scale is None:
            self.scale = None
            return
        self._copy(self.scale, scale)
        self.inv_scale.data = 1.0 / self.scale
        self.scale = self.scale.to(device)
        self.inv_scale = self.inv_scale.to(device)

    def load_weights_customized(self,
                                 weights,
                                 loader,
                                 device: torch.device = torch.device('cpu'),
                                 **kwargs):
        self.scale, self.inv_scale = loader(weights)
        self.scale = self.scale.to(device)
        self.inv_scale = self.inv_scale.to(device)


class NVFP4(BaseQuant):

    def __init__(self, quant_config, **kwargs):
        super().__init__(quant_config)
        self.scaling_vector_size = 16
        self.scale = Parameter(torch.tensor(1, dtype=torch.float32),
                               requires_grad=False)
        self.inv_scale = Parameter(torch.tensor(1, dtype=torch.float32),
                                   requires_grad=False)

        self.scale_factor_use_ue8m0 = False
        self.is_scale_factor_swizzled = kwargs[
            "is_scale_factor_swizzled"] if "is_scale_factor_swizzled" in kwargs else None

    def __call__(self, input):
        if self.scale is None:
            return input, self.scale
        if isinstance(input, Fp4QuantizedTensor):
            return input.fp4_tensor, input.scaling_factor
        else:
            return torch.ops.trtllm.fp4_quantize(input, self.scale,
                                                 self.scaling_vector_size,
                                                 self.scale_factor_use_ue8m0,
                                                 True)

    def load_weights(self,
                     weights,
                     tensor_name,
                     device: torch.device = torch.device('cpu')):
        scale = self._load_weight_for_name(weights, tensor_name)
        if scale is None:
            self.scale = None
            return
        scale = 1.0 / scale
        self._copy(self.scale, scale)
        self.inv_scale.data = self.scale / E2M1_MAX
        self.scale = self.scale.to(device)
        self.inv_scale = self.inv_scale.to(device)

    def load_weights_customized(self,
                                 weights,
                                 loader,
                                 device: torch.device = torch.device('cpu'),
                                 **kwargs):
        self.scale, self.inv_scale = loader(self, weights, **kwargs)
        self.scale = self.scale.to(device)
        self.inv_scale = self.inv_scale.to(device)


class LinearQuantCreator:

    @staticmethod
    def create_quantizer(quant_config):
        quant_mode = quant_config.layer_quant_mode if quant_config else None
        quant = None
        # TODO: need to make src/target dtype to be parameters.
        if quant_mode:
            if quant_mode.has_fp8_qdq():
                quant = QDQ(quant_config)
            elif quant_mode.has_nvfp4():
                quant = NVFP4(quant_config)
            elif quant_mode.has_fp8_block_scales():
                quant = BlockScalesQuant(quant_config)

        return quant


class MoeQuantCreator:

    @staticmethod
    def create_quantizer(quant_config, backend):
        quant_mode = quant_config.layer_quant_mode if quant_config else None
        quant = None
        # TODO: need to make src/target dtype to be parameters.
        if quant_mode:
            if quant_mode.has_fp8_qdq():
                quant = QDQ(quant_config)
            elif quant_mode.has_nvfp4():
                quant = NVFP4(quant_config, is_scale_factor_swizzled=False)
            elif quant_mode.has_fp8_block_scales():
                if backend == "trtllm_gen":
                    quant = BlockScalesQuant(quant_config)
                else:
                    quant = NoopQuant(quant_config)
            elif quant_mode.is_int4_weight_only_per_group():
                assert backend != "trtllm_gen"
                quant = NoopQuant(quant_config)
            else:
                raise f"Quant mode {quant_mode} is not supported."
        return quant
