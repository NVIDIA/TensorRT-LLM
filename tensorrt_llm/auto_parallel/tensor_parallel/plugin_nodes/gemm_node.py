import tensorrt as trt

from tensorrt_llm._utils import trt_dtype_to_str

from ..matmul_node import MatrixMultiply
from ..plugin_node import PluginNode


class GemmPlugin(MatrixMultiply, PluginNode):

    def __init__(self, layer):
        PluginNode.__init__(self, layer)
        batch_dims = [i for i in range(len(self.get_output(0).shape))][:-2]
        self._generate_bcast_dims(batch_dims, self.get_output(0).shape)
        pfc_as_list = self.plugin_info.pfc_as_list
        self.op0_transpose = (pfc_as_list['transa'][0] == 1)
        self.op1_transpose = (pfc_as_list['transb'][0] == 1)
        self.num_out_dims = len(self.get_output(0).shape)
        self.dtype = trt_dtype_to_str(trt.DataType(pfc_as_list['type_id'][0]))

    def _collect_strategies(self, device_mesh):
        strategies_vector = MatrixMultiply._collect_strategies(
            self, device_mesh)
        return strategies_vector

    def _get_math_time(self, strategy, device_mesh):
        return MatrixMultiply._get_math_time(self, strategy, device_mesh)
