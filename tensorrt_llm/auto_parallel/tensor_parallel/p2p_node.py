import copy
from enum import Enum

from .comm_spec import CommSpec
from .identity_node import Identity
from .sharding_strategy import StrategiesVector


class P2PType(Enum):
    CROSS_DEVICE = 0
    CROSS_HOST = 1


class P2PNode(Identity):

    def __init__(self, layer):
        super().__init__(layer)
        self.p2p_type = layer.attrs["p2p_type"]
        self.is_fake = True

    def _collect_strategies(self, device_mesh):
        # one input for softmax node
        predecessor = self.predecessor_nodes[0]
        strategies_vector = StrategiesVector(self)
        for idx, strategy in enumerate(predecessor.strategies_vector):
            # current node's local name input0 -> global name xxx
            global_input_name = self.op_data['input0'].name
            # global name xxx -> pre node local output name
            prenode_local_name = predecessor.global_to_local_op_name[
                global_input_name]
            dim_partition_dict = copy.deepcopy(
                strategy.sharding_specs[prenode_local_name].dim_partition_dict)
            in0_partition_dict = dim_partition_dict
            out_partition_dict = copy.deepcopy(dim_partition_dict)
            dim_partition_dict_mapping = {
                "input0": in0_partition_dict,
                "output0": out_partition_dict,
            }
            sharding_spec_mapping = self._to_sharding_spec_mapping(
                dim_partition_dict_mapping, device_mesh)
            if 0 == len(sharding_spec_mapping):
                continue

            logical_process_axis = [
                ['p2p_cross_device']
            ] if self.p2p_type == P2PType.CROSS_DEVICE else [['p2p_cross_host']]
            # get communication action mapping
            communication_action_mapping = {}
            output0_comm_action = CommSpec(
                comm_pattern='peer_to_peer',
                sharding_spec=sharding_spec_mapping['output0'],
                logical_process_axis=logical_process_axis,
            )
            communication_action_mapping['output0'] = output0_comm_action

            name = '{} = <P2P op> {}'.format(
                sharding_spec_mapping['output0'].sharding_sequence,
                sharding_spec_mapping['input0'].sharding_sequence)
            sharding_strategy = self._get_sharding_strategy(
                name=name,
                sharding_spec_mapping=sharding_spec_mapping,
                communication_action_mapping=communication_action_mapping)
            strategies_vector.append(sharding_strategy)
        return strategies_vector

    def _profile_sharding_cost(self, strategy, device_mesh):
        return 0.0
