class ShardingStrategy(object):

    def __init__(self,
                 name=None,
                 sharding_specs=None,
                 communication_actions=None):
        self.name = name or ""
        self.sharding_specs = sharding_specs or {}
        self.communication_actions = communication_actions
        self.sharding_cost = 0
        self.communication_cost = 0
        self.resharding_costs = {}
        self.best_resharding_cost = {}
        self.node_names = {}

        self.comm_buff_memory_footprint = 0
        self.inout_memory_footprint = 0
        self.const_memory_footprint = 0
        self.peak_memory_footprint = 0
        self.computation_macs = 0
        self.alpha_beta_cost = 0

    def print_strategy(self, best_resharding_cost_only=False, file=None):

        def print_resharding_costs(resharding_cost):
            for prenode_node_name, rcosts in resharding_cost.items():
                if isinstance(prenode_node_name, int):
                    idx = prenode_node_name
                    prenode_node_name = self.node_names[idx]
                    print(f'    pre_node = {idx} {prenode_node_name}',
                          file=file)
                else:
                    print(f'    pre_node = {prenode_node_name}', file=file)
                for idx, rcost in enumerate(rcosts):
                    transpaths, commspecs, cost = rcost
                    print(f'    {idx}: ', end=' ', file=file)
                    device_mesh.shape_consistency_manager.print_shape_consistency_result(
                        transpaths, commspecs, cost, file)

        print(f'name = {self.name}', file=file)
        print(f'sharding_cost = {self.sharding_cost}', file=file)
        print(
            f'communication_buffer_memory_footprint = {self.comm_buff_memory_footprint}, communication_cost = {self.communication_cost}',
            file=file)
        print(f'inout_memory_footprint = {self.inout_memory_footprint}',
              file=file)
        print(f'peak_memory_footprint = {self.peak_memory_footprint}',
              file=file)
        print(f'const_memory_footprint = {self.const_memory_footprint}',
              file=file)
        print('sharding_specs:', file=file)
        device_mesh = None
        for specname, spec in self.sharding_specs.items():
            print(specname + ', ', end=' ', file=file)
            spec.print_spec(file)
            device_mesh = spec.device_mesh

        if best_resharding_cost_only and self.best_resharding_cost:
            print('best_resharding_costs:', file=file)
            print_resharding_costs(self.best_resharding_cost)
        else:
            print('resharding costs:', file=file)
            print_resharding_costs(self.resharding_costs)


class StrategiesVector(list):
    '''
    Each node in fx graph will have a corresponding StrategiesVector, to store all the possible
    strategies of the node.

    Argument:
        node (Node): node for which the list of sharding strategies are generated.
    '''

    def __init__(self, node):
        super().__init__()
        self.node = node
