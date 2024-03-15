import tensorrt as trt

from ..gather_node import Gather
from ..plugin_node import PluginNode


class LookupPlugin(Gather, PluginNode):

    def __init__(self, layer):
        PluginNode.__init__(self, layer)
        self.mode = trt.GatherMode.DEFAULT
        self.axis = 0
        self.num_elementwise_dims = 0
        self.input_id = 1
        self.indice_id = 0
        self.support_vocab_tp = True

    def _collect_strategies(self, device_mesh):
        return Gather._collect_strategies(self, device_mesh)
