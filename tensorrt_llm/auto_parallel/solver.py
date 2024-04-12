"""This code is adapted from Alpa https://github.com/alpa-projects/alpa/ with some changes.
"""
import multiprocessing
import time
import warnings
from collections import defaultdict

import numpy as np
import pulp
from pulp import LpMinimize, LpProblem, LpVariable, lpDot, lpSum

from ..logger import logger


class Solution:

    def __init__(self, leaf_strategies, s_val, e_val, edge_pairs,
                 node_index_dict, total_cost):
        self.leaf_strategies = leaf_strategies
        self.nodes = [
            strategies_vector.node for strategies_vector in self.leaf_strategies
        ]
        self.s_val = s_val
        self.e_val = e_val
        self.total_cost = total_cost
        self.edge_pairs = list(np.reshape(edge_pairs, (-1, 2)))
        self.node_index_dict = node_index_dict
        self.index_node_dict = {}
        for node, index in self.node_index_dict.items():
            self.index_node_dict[index] = node
        self.node_best_strategy = {}
        self._annotate_strategy()

    def _annotate_strategy(self):
        self.node_best_strategy = {}
        for index, node in enumerate(self.nodes):
            best_strategy_id = self.s_val[index]
            best_strategy = self.leaf_strategies[index][best_strategy_id]
            self.node_best_strategy[node.node_name] = best_strategy

        for edge_idx, edge_pair in enumerate(self.edge_pairs):
            src_node = self.index_node_dict[edge_pair[0]]
            dst_node = self.index_node_dict[edge_pair[1]]
            src_node_index = self.node_index_dict[src_node]
            for dst_pre_node in dst_node.predecessor_nodes:
                if dst_pre_node is None:
                    continue
                if src_node.node_name == dst_pre_node.node_name:
                    self.node_best_strategy[
                        dst_node.node_name].best_resharding_cost[
                            src_node.node_name] = [
                                self.node_best_strategy[dst_node.node_name].
                                resharding_costs[src_node.node_name][
                                    self.s_val[src_node_index]]
                            ]

    def print_solution(self):
        for index, node in enumerate(self.nodes):
            best_strategy = self.node_best_strategy[node.node_name]
            print(f'\n[{index}]: node_name = {node.node_name}')
            best_strategy.print_strategy(best_resharding_cost_only=True)
        print(f'solution total cost = {self.total_cost}')


class CostGraph:
    '''
    A graph data structure to simplify the edge cost graph. It has two main functions:
    1. To feed the quadratic resharding costs into solver, we need to linearize it. We build edge_cost in
    CostGraph, and it stored every combinations of strategies for a src-dst node pair in an 1D list.
    2. To reduce the searching space, we merge computationally-trivial operators, such as
    element-wise operators, transpose, and reduction, into their following nodes. The merging information will
    be given by the StrategiesVector depending on the type of target node and following nodes.

    Argument:
        leaf_strategies(List[StrategiesVector]): It stores StrategiesVector of every nodes on the graph.
        simplify(bool, optional): The generated cost graph will be simplified if it is true. (default to True)
    '''

    def __init__(self, leaf_strategies):
        self.leaf_strategies = leaf_strategies
        self.nodes = [
            strategies_vector.node for strategies_vector in leaf_strategies
        ]
        # stores number of strategies in each node
        self.node_strategies_vector = {}
        for node, strategies_vector in zip(self.nodes, self.leaf_strategies):
            self.node_strategies_vector[node] = strategies_vector
        # extra_node_costs will store the extra costs introduced by merging nodes
        self.extra_node_costs = {}
        self.following_dict = {}
        self._build_cost_graph()

    def _remove_invalid_node(self, node, attr_name):
        remove_list = []
        target_node_list = getattr(node, attr_name, [])
        for target_node in target_node_list:
            if target_node not in self.nodes:
                remove_list.append(target_node)
        for element in remove_list:
            target_node_list.remove(element)

    def _build_cost_graph(self):
        '''
        This method will generate edge_cost for adjacent node pair. Additionally, 'parents' and 'children' attribute will be
        set to node.
        '''
        self.edge_costs = {}
        for dst_node, strategies_vector in zip(self.nodes,
                                               self.leaf_strategies):
            # build edge_cost
            for src_node in dst_node.predecessor_nodes:
                if src_node is None:
                    continue
                if src_node not in self.nodes:
                    continue
                node_pair = (src_node, dst_node)
                edge_cost = {}
                for i in range(len(strategies_vector)):
                    for j in range(len(self.node_strategies_vector[src_node])):
                        resharding_cost = strategies_vector[i].resharding_costs[
                            src_node.node_name][j][-1]
                        edge_cost[(j, i)] = resharding_cost
                self.edge_costs[node_pair] = edge_cost

    def get_edge_cost(self, src_node, dst_node):
        return self.edge_costs[(src_node, dst_node)]


class Solver:
    INFINITY_COST = 1e13

    def __init__(self,
                 cost_graph: CostGraph,
                 memory_budget: float = -1.0,
                 solution_numbers: int = 1,
                 memory_increasing_coefficient: float = 1.3,
                 verbose=False):
        '''
        Solver class will integrate information provided by the components and use ILP solver to find a possible optimal strategies combination for target computing graph.
        Argument:
            graph: The computing graph to be optimized.
            strategies_constructor: It will provide all the possible strategies for each node in the computing graph.
            cost_graph: A graph data structure to simplify the edge cost graph.
            graph_analyser: graph_analyser will analyse the graph to obtain the variable liveness information, which will be used to generate memory constraints.
            memory_budget: Memory constraint for the solution.
            solution_numbers: If solution_numbers is larger than one, solver will us a serious of solutions based on different memory budget.
            memory_increasing_coefficient: If solution_numbers is larger than one, we will use this coefficient to generate new memory budget.
        '''
        self.cost_graph = cost_graph
        self.leaf_strategies = cost_graph.leaf_strategies
        self.nodes = cost_graph.nodes
        self.memory_budget = memory_budget
        self.solution_numbers = solution_numbers
        if self.solution_numbers > 1:
            self.memory_increasing_coefficient = memory_increasing_coefficient
        else:
            self.memory_increasing_coefficient = 1
        # temporarily we use all nodes as liveness list, we count the backward memory cost together with
        # forward memory cost into the node memory cost, and no activation checkpoint is used in this phase.
        # self.liveness_list = self.graph_analyser.liveness_analysis()
        self.liveness_list = self.nodes
        self.node_index_dict = self._generate_node_index_dict()
        # The last solution vector of auto sharding.
        self.last_s_val = None
        # The last objective value of the best ILP solution.
        self.last_objective = None
        self.verbose = verbose

    def _generate_node_index_dict(self):
        node_index_dict = {}
        for index, node in enumerate(self.nodes):
            node_index_dict[node] = index
        return node_index_dict

    def _prepare_data_for_solver(self):
        '''
        Extract information from components for solver.
        '''
        node_nums = len(self.leaf_strategies)
        memory_budget = self.memory_budget

        # prepare strategies_len
        strategies_len = []
        for node in self.nodes:
            strategies_len.append(
                len(self.cost_graph.node_strategies_vector[node]))
        strategies_len = np.array(strategies_len)

        # prepare edge_pairs and resharding costs
        edge_pairs = []
        resharding_costs = []
        edge_cost_level = []
        edge_resharding_weights = []
        for pairs, edge_cost in self.cost_graph.edge_costs.items():
            src_node = pairs[0]
            dst_node = pairs[1]
            src_node_index = self.node_index_dict[src_node]
            dst_node_index = self.node_index_dict[dst_node]
            edge_pairs.append(src_node_index)
            edge_pairs.append(dst_node_index)
            edge_cost_level.append(
                (dst_node.building_block_id, dst_node.cost_level))
            for i in range(strategies_len[src_node_index]):
                for j in range(strategies_len[dst_node_index]):
                    resharding_costs.append(edge_cost[(i, j)])
            edge_resharding_weights.append(dst_node.resharding_weight +
                                           dst_node.pipeline_weight)
        edge_pairs = np.array(edge_pairs)
        resharding_costs = np.array(resharding_costs)
        edge_resharding_weights = np.array(edge_resharding_weights)
        # prepare compute_costs, communication_costs and memory_costs
        compute_costs = []
        communication_costs = []
        memory_costs = []
        peak_act_memory_costs, constant_memory_costs = [], []
        node_sharding_weights = []
        for node, strategies_vector in zip(self.nodes, self.leaf_strategies):
            for index, strategy in enumerate(strategies_vector):
                compute_cost = strategy.sharding_cost
                origin_communication_cost = strategy.communication_cost
                memory_cost = strategy.const_memory_footprint * node.sharding_weight
                peak_act_memory = strategy.peak_memory_footprint
                # extract the memory cost in float from MemoryCost item and sum them up
                compute_costs.append(compute_cost)
                # node in extra_node_costs means it has some extra communication
                # cost from node merging, so we need to add those extra communication
                # cost into

                communication_costs.append(origin_communication_cost)
                peak_act_memory_costs.append(peak_act_memory)
                constant_memory_costs.append(memory_cost)
            node_sharding_weights.append(node.sharding_weight +
                                         node.pipeline_weight)

        compute_costs = np.array(compute_costs)
        communication_costs = np.array(communication_costs)
        memory_costs = np.array([constant_memory_costs, peak_act_memory_costs])
        node_sharding_weights = np.array(node_sharding_weights)
        same_spec_nodes_dict = defaultdict(list)
        node_cost_level = []
        for idx, node in enumerate(self.nodes):
            if node.same_spec_id >= 0:
                same_spec_nodes_dict[node.same_spec_id].append(idx)
            node_cost_level.append((node.building_block_id, node.cost_level))
        # omit initial value for nodes
        s_init_np = None
        following_nodes = [-1 for i in range(node_nums)]
        liveness_set = self.nodes
        alias_set = []
        alias_convert_costs = None
        return node_nums, memory_budget, strategies_len, following_nodes, edge_pairs, alias_set, liveness_set, compute_costs, communication_costs, memory_costs, resharding_costs, node_sharding_weights, edge_resharding_weights, same_spec_nodes_dict, node_cost_level, edge_cost_level, alias_convert_costs, s_init_np, self.verbose

    def _call_solver_serialized_args(self,
                                     node_nums,
                                     memory_budget,
                                     strategies_len,
                                     following_nodes,
                                     edge_pairs,
                                     alias_set,
                                     liveness_set,
                                     compute_costs,
                                     communication_costs,
                                     memory_costs,
                                     resharding_costs,
                                     node_sharding_weights,
                                     edge_resharding_weights,
                                     same_spec_nodes_dict,
                                     node_cost_level,
                                     edge_cost_level,
                                     alias_convert_costs,
                                     s_init_np=None,
                                     verbose=True):
        """
        Call the solver with serialized arguments.
        """

        time.time()

        for x in [
                strategies_len, edge_pairs, compute_costs, communication_costs,
                memory_costs, resharding_costs, node_sharding_weights,
                edge_resharding_weights
        ]:
            assert isinstance(x, np.ndarray)
        assert len(strategies_len) == node_nums, "strategies_len"

        def get_non_zero_index(binary_vector):
            """
            Get the index of non-zero item in a vector.
            """
            ct = 0
            ret = None
            for i, elem in enumerate(binary_vector):
                if pulp.value(elem):
                    ret = i
                    ct += 1

            assert ct == 1
            return ret

        # 0. Unpack flatten numpy arrays
        s_follow = following_nodes
        s_alias = alias_set

        E = edge_pairs.reshape((-1, 2))  # noqa
        r = []
        pt = 0
        edge_set = set()
        for (i, j) in E:
            prod_length = strategies_len[i] * strategies_len[j]

            if (i, j) in edge_set:
                raise ValueError(f"Duplicated edges: {(i, j)}")

            edge_set.add((i, j))
            r.append(resharding_costs[pt:pt + prod_length])
            pt += prod_length
        assert pt == len(resharding_costs)

        ######################
        # omit alias set now #
        ######################

        # A = alias_set.reshape((-1, 2))  # noqa
        # for (i, j) in A:
        #     prod_length = strategies_len[i] * strategies_len[j]
        #     v.append(alias_convert_costs[pt:pt + prod_length])
        #     pt += prod_length
        # assert pt == len(alias_convert_costs)

        # L = []  # noqa
        # pt = node_nums
        # for i in range(node_nums):
        #     length = liveness_set[i]
        #     L.append(liveness_set[pt:pt + length])
        #     pt += length
        # assert pt == len(liveness_set)
        pt = 0

        c = []
        d = []
        m = []
        peak_m = []
        pt = 0
        for i in range(node_nums):
            length = strategies_len[i]
            c.append(compute_costs[pt:pt + length])
            d.append(communication_costs[pt:pt + length])
            m.append(memory_costs[0][pt:pt + length])
            peak_m.append(memory_costs[1][pt:pt + length])
            pt += length
        assert pt == len(compute_costs), f"{pt} == {len(compute_costs)}"
        assert pt == len(
            communication_costs), f"{pt} == {len(communication_costs)}"
        assert pt == len(memory_costs[0]), f"{pt} == {len(memory_costs[0])}"

        # 1. Create variables

        #############################
        # create variables for node #
        #############################
        s = []
        num_nodes = 0
        reverse_follow_backpatch = []
        for i in range(node_nums):
            if s_follow[i] < 0:
                if strategies_len[i] == 1:
                    s.append([1])
                else:
                    if i not in s_alias:
                        num_nodes += 1
                        s.append(
                            LpVariable.matrix(f"s[{i}]",
                                              (range(strategies_len[i]), ),
                                              cat="Binary"))
                    else:
                        s.append(s[s_alias[i]])
            else:
                if s_follow[i] < len(s):
                    s.append(s[s_follow[i]])
                else:
                    s.append(None)
                    reverse_follow_backpatch.append(i)

        for i in reverse_follow_backpatch:
            s[i] = s[s_follow[i]]

        #############################
        # create variables for edge #
        #############################
        e = []
        num_edges = 0
        map_edge_to_idx = {}
        for (idx, (i, j)) in enumerate(E):
            if len(s[i]) == 1:
                e.append(s[j])
            elif len(s[j]) == 1:
                e.append(s[i])
            else:
                if i in s_alias and j in s_alias and (
                        s_alias[i], s_alias[j]) in map_edge_to_idx:
                    e.append(e[map_edge_to_idx[(s_alias[i], s_alias[j])]])
                else:
                    num_edges += 1
                    e.append(
                        LpVariable.matrix(f"e[{i},{j}]",
                                          (range(len(s[i]) * len(s[j])), ),
                                          cat="Binary"))
            assert len(e[idx]) == len(r[idx])
            map_edge_to_idx[(i, j)] = idx
        for element in s:
            assert len(element) > 0
        # 2. Set initial value
        ######################################
        # set a initial value for warm start #
        ######################################
        if s_init_np is not None:
            s_init = s_init_np.reshape((-1, 3))
            for (idx, value, fix) in s_init:
                for i in range(len(s[idx])):
                    s[idx][i].setInitialValue(i == value)
                    if fix:
                        s[idx][i].fixValue()

        # 3. Objective
        prob = LpProblem("myProblem", LpMinimize)
        ###################################################################
        # computing the node cost(computing cost and communication cost)  #
        ###################################################################
        obj = 0
        block_cost_level_dict = {}
        for i in range(node_nums):
            assert len(s[i]) == len(c[i])
            assert len(s[i]) == len(d[i])
            obj += (lpDot(s[i], c[i]) +
                    lpDot(s[i], d[i])) * node_sharding_weights[i]
            cost_level = node_cost_level[i]
            if -1 != cost_level[1]:
                if cost_level in block_cost_level_dict:
                    block_cost_level_dict[cost_level] += lpDot(
                        s[i], c[i]) + lpDot(s[i], d[i])
                else:
                    block_cost_level_dict[cost_level] = lpDot(
                        s[i], c[i]) + lpDot(s[i], d[i])

        #############################################
        # computing the edge cost(resharding cost)  #
        #############################################

        for i in range(len(E)):
            assert len(e[i]) == len(r[i])
            obj += lpDot(e[i], r[i]) * edge_resharding_weights[i]
            cost_level = edge_cost_level[i]
            if -1 != cost_level[1]:
                if cost_level in block_cost_level_dict:
                    block_cost_level_dict[cost_level] += lpDot(e[i], r[i])
                else:
                    block_cost_level_dict[cost_level] = lpDot(e[i], r[i])
        prob += obj
        if len(block_cost_level_dict) >= 2:
            block_cost_levels = [key for key in block_cost_level_dict.keys()]
            for i in range(len(block_cost_levels)):
                for j in range(i + 1, len(block_cost_levels)):
                    if block_cost_levels[i][1] > block_cost_levels[j][1]:
                        prob += block_cost_level_dict[
                            block_cost_levels[i]] >= block_cost_level_dict[
                                block_cost_levels[j]] + 1e-6
                    elif block_cost_levels[i][1] < block_cost_levels[j][1]:
                        prob += block_cost_level_dict[
                            block_cost_levels[j]] >= block_cost_level_dict[
                                block_cost_levels[i]] + 1e-6
        # 4. Constraints
        # (a). specified by `cat="Binary"`

        # (b)
        #################################################
        # make sure each node only choose one strategy  #
        #################################################
        for i in range(node_nums):
            if s_follow[i] < 0:
                prob += lpSum(s[i]) == 1

        # (c)
        #################################################
        # force to constrain some nodes have the same sharding specs  #
        #################################################
        for spec_id, same_spec_nodes_id in same_spec_nodes_dict.items():
            num_same_spec_nodes = len(same_spec_nodes_id)
            if num_same_spec_nodes >= 2:
                src_node_s = s[same_spec_nodes_id[0]]
                num_specs = len(src_node_s)
                for i in range(1, num_same_spec_nodes):
                    dst_node_s = s[same_spec_nodes_id[i]]
                    assert len(
                        dst_node_s
                    ) == num_specs, f'unmatched num_specs when force node {same_spec_nodes_id[0]} and {same_spec_nodes_id[i]} the same specs'
                    for j in range(num_specs):
                        prob += (src_node_s[j] == dst_node_s[j])

        # (c)
        #################################################
        # compute memory consumption with liveness set  #
        #################################################
        if memory_budget > 0:
            # calculate the constant memory
            mem = 0
            for node in liveness_set:
                if node not in self.node_index_dict:
                    continue
                node_index = self.node_index_dict[node]
                mem += lpSum(s[node_index][j] * m[node_index][j]
                             for j in range(len(s[node_index])))
            # calculate the peak activation memory
            for node in liveness_set:
                if node not in self.node_index_dict:
                    continue
                node_index = self.node_index_dict[node]
                cur_peak_mem = lpSum(s[node_index][j] * peak_m[node_index][j]
                                     for j in range(len(s[node_index])))
                total_mem = mem + cur_peak_mem
                prob += total_mem <= memory_budget

        # (d). specified by `cat="Binary"`

        for (idx, (i, j)) in enumerate(E):
            if strategies_len[i] == 1 or strategies_len[j] == 1:
                continue

            # (e)
            prob += lpSum(e[idx]) == 1

            # (f)
            for row in range(len(s[i])):
                C = len(s[j])  # noqa
                prob += lpSum(e[idx][row * C + col]
                              for col in range(0, C)) <= s[i][row]

            # (g)
            for col in range(len(s[j])):
                R = len(s[i])  # noqa
                C = len(s[j])  # noqa
                prob += lpSum(e[idx][row * C + col]
                              for row in range(0, R)) <= s[j][col]

        if prob.objective.isNumericalConstant():
            objective = float(pulp.value(prob.objective))
            status = pulp.LpStatusOptimal
        else:
            msg = verbose
            time_limit = 600
            solver = pulp.PULP_CBC_CMD(
                mip=True,
                msg=msg,
                timeLimit=time_limit,
                threads=multiprocessing.cpu_count(),
            )
            prob.solve(solver)

            status = prob.status
            objective = pulp.value(prob.objective)
            objective = float(
                objective) if objective is not None else self.INFINITY_COST

            if prob.status in [pulp.LpStatusInfeasible]:
                objective = self.INFINITY_COST

        # Get and check results
        s_val = np.full((node_nums, ), -1, dtype=np.int32)
        for i in range(node_nums):
            s_val[i] = get_non_zero_index(s[i])

        e_val = np.full((len(E), ), -1, dtype=np.int32)
        for (idx, (i, j)) in enumerate(E):
            e_val[idx] = get_non_zero_index(e[idx])
            i_spec_index = e_val[idx] // len(s[j])
            j_spec_index = e_val[idx] % len(s[j])
            assert i_spec_index == s_val[i], f"e_val[{i}][{j}]"
            assert j_spec_index == s_val[j], f"e_val[{i}][{j}]"
            if verbose and r[idx][e_val[idx]] > 0:
                print(f"Edge cost {(i, j)} : {r[idx][e_val[idx]]}")

        self.last_s_val = list(s_val)
        # self._recover_merged_node_strategy()
        self.last_objective = objective

        if objective >= self.INFINITY_COST:
            warnings.warn(
                f"Cannot find an optimized solution given memory budget {self.memory_budget}, Please consider\n" + \
                f"1. increase memory budget if possible\n" + \
                f"2. enlarge mesh shape if possible\n" + \
                f"3. decrease the maximum parameters(i.e., max_batch_size, max_seq_len, etc.) in building config")
        if memory_budget > 0:
            # calculate the constant memory
            mem = 0
            for node in liveness_set:
                if node not in self.node_index_dict:
                    continue
                node_index = self.node_index_dict[node]
                j = self.last_s_val[node_index]
                mem += m[node_index][j]
            max_peak_mem = 0
            for node in liveness_set:
                if node not in self.node_index_dict:
                    continue
                node_index = self.node_index_dict[node]
                j = self.last_s_val[node_index]
                cur_peak_mem = peak_m[node_index][j]
                max_peak_mem = max(max_peak_mem, cur_peak_mem)
            logger.debug(
                f'constant_mem = {mem}, peak_mem = {max_peak_mem}, memory_budget = {memory_budget}'
            )

        solution = Solution(self.leaf_strategies, self.last_s_val, e_val,
                            edge_pairs, self.node_index_dict,
                            self.last_objective)
        return status, solution

    def find_solution(self):
        """
        Call the solver with serialized arguments and handle python errors. Additionally,
        we could give a serious of solutions with different memory budget.
        """
        if self.solution_numbers == 1:
            args = self._prepare_data_for_solver()
            ret = self._call_solver_serialized_args(*args)

            return ret

        origin_memory_budget = self.memory_budget
        memory_budget_list = [
            origin_memory_budget * self.memory_increasing_coefficient**i
            for i in range(self.solution_numbers)
        ]
        ret_list = []
        for memory_budget in memory_budget_list:
            self.memory_budget = memory_budget
            args = self._prepare_data_for_solver()
            ret = self._call_solver_serialized_args(*args)
            ret_list.append(ret)

        return ret_list
