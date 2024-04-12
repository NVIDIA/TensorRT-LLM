from ..normalization_node import Normalization
from ..plugin_node import PluginNode


class LayernormPlugin(Normalization, PluginNode):

    def __init__(self, layer):
        PluginNode.__init__(self, layer)
        # the is only true for llm model, because layer norm is only effect on hidden dim
        hidden_dim = len(self.op_data['input0'].shape) - 1
        self.axes = 1 << hidden_dim
        self.weight_bias_dim_base = hidden_dim

    def _collect_strategies(self, device_mesh):
        return Normalization._collect_strategies(self, device_mesh)


class RMSnormPlugin(Normalization, PluginNode):

    def __init__(self, layer):
        PluginNode.__init__(self, layer)
        # the is only true for llm model, because rms norm is only effect on hidden dim
        hidden_dim = len(self.op_data['input0'].shape) - 1
        self.axes = 1 << hidden_dim
        self.weight_bias_dim_base = hidden_dim

    def _collect_strategies(self, device_mesh):
        return Normalization._collect_strategies(self, device_mesh)
