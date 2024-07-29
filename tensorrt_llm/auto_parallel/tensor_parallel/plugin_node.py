from tensorrt_llm.network import PluginInfo, get_plugin_info

from .node import Node
from .sharding_strategy import StrategiesVector


class PluginNode(Node):

    def __init__(self, layer):
        super().__init__(layer)
        layer.to_subclass()
        self.plugin = layer.as_trt().plugin
        self.plugin_type: str = self.plugin.plugin_type
        self.plugin_info: PluginInfo = get_plugin_info(layer.graph.as_trt(),
                                                       layer.name)
        layer.to_base_class()

    def _collect_strategies(self, device_mesh):
        raise NotImplementedError(
            f"Auto parallel does not support {self.plugin_type} plugin right now."
        )

    def _default_strategy(self, device_mesh):
        strategies_vector = StrategiesVector(self)
        dim_partition_dict_mapping = {}
        for idx in range(self.num_inputs):
            dim_partition_dict_mapping[f'input{idx}'] = {}
        for idx in range(self.num_outputs):
            dim_partition_dict_mapping[f'output{idx}'] = {}
        sharding_spec_mapping = self._to_sharding_spec_mapping(
            dim_partition_dict_mapping, device_mesh)
        if 0 == len(sharding_spec_mapping):
            return strategies_vector
        name = '{}_all_replicate'.format(self.plugin_type)
        sharding_strategy = self._get_sharding_strategy(
            name=name,
            sharding_spec_mapping=sharding_spec_mapping,
            communication_action_mapping={})
        strategies_vector.append(sharding_strategy)
        return strategies_vector
