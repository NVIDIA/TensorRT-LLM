# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from collections import OrderedDict
from typing import List, Optional
from weakref import WeakSet

import torch
import torch._inductor.config as inductor_config
from torch._functorch.aot_autograd import aot_module_simplified
from torch._inductor.compile_fx import compile_fx_inner, select_decomp_table
from torch._inductor.pattern_matcher import PatternMatcherPass
from torch._subclasses import FakeTensor
from torch.fx import GraphModule

import tensorrt_llm
from tensorrt_llm import logger
from tensorrt_llm.mapping import Mapping

from .multi_stream.auto_multi_stream import multi_stream_schedule
from .patterns import MATCHER_SUBSYSTEM
from .patterns.ar_residual_norm import register_ar_fusions
from .patterns.residual_add_norm import (register_add_norm,
                                         register_add_norm_quant)
from .piecewise_optimizer import PiecewiseRunner, piecewise_optimizer
from .recover_pass import recover_pass
from .remove_copy_pass import remove_copy_for_mutates_args


class Backend:

    _graph_pool_handle: tuple[int, int] = None

    # Following classes are used to let weakref ref the stream and eventlist objects.
    class Streams(list):
        pass

    class Events(list):
        pass

    def __init__(
        self,
        enable_inductor=True,
        enable_userbuffers=False,
        enable_piecewise_cuda_graph: bool = False,
        capture_num_tokens: Optional[List[int]] = None,
        max_num_streams: int = 1,
        mapping=None,
    ) -> None:
        super().__init__()
        self.elapsed_time = 0
        self.module_inference_event = []
        self.module_inference_time = 0
        self.call_count = 0
        self.mapping = mapping
        self.custom_passes = Backend.build_custom_passes(
            enable_userbuffers, mapping)
        self.rank = tensorrt_llm.mpi_rank()
        self.enable_inductor = enable_inductor
        self.capture_num_tokens = sorted(capture_num_tokens or [])
        self.piecewise_cuda_graph = enable_piecewise_cuda_graph
        self._piecewise_runners: WeakSet[PiecewiseRunner] = WeakSet()
        self.no_optimization = False
        self.num_streams = max_num_streams
        self.events = Backend.Events()
        inductor_config.enable_auto_functionalized_v2 = False

        if Backend._graph_pool_handle is None:
            Backend._graph_pool_handle = torch.cuda.graph_pool_handle()

        self.match_count = []
        self.match_count_by_pass = OrderedDict()

    @classmethod
    def build_custom_passes(cls, enable_userbuffers, mapping: Mapping):
        world_size = tensorrt_llm.mpi_world_size()
        # Really naive pass manager here
        custom_passes = [PatternMatcherPass("add_norm", MATCHER_SUBSYSTEM)]
        if world_size > 1:
            # Currently torch compile cannot work properly with lamport fusion kernel
            # TO-DO: Fix this issue
            os.environ["DISABLE_LAMPORT_REDUCE_NORM_FUSION"] = "1"
            ub_enabled = enable_userbuffers and tensorrt_llm.bindings.internal.userbuffers.ub_supported(
            )
            custom_passes[-1] = PatternMatcherPass("ar_residual_norm",
                                                   MATCHER_SUBSYSTEM)
            register_ar_fusions(custom_passes, mapping, ub_enabled)
            # Fallback: fuse remaining add+rmsnorm not preceded by allreduce
            custom_passes.append(
                PatternMatcherPass("add_norm_fallback", MATCHER_SUBSYSTEM))
            register_add_norm(custom_passes[-1])
        else:
            # Add fp8 quant pattern before fp16/bf16 pattern
            custom_passes[-1] = PatternMatcherPass("add_norm_quant",
                                                   MATCHER_SUBSYSTEM)
            register_add_norm_quant(custom_passes[-1])
            custom_passes.append(
                PatternMatcherPass("add_norm_fallback", MATCHER_SUBSYSTEM))
            register_add_norm(custom_passes[-1])
        return custom_passes

    def bypass_optimization(self):
        self.no_optimization = True

    def enable_optimization(self):
        self.no_optimization = False

    def generate_events(self, num_events: int):
        if num_events > len(self.events):
            self.events += [
                torch.cuda.Event() for _ in range(num_events - len(self.events))
            ]

    def clear_piecewise_cuda_graphs(self) -> None:
        runners = list(self._piecewise_runners)
        for runner in runners:
            runner.clear_cuda_graphs()

        # CUDACachingAllocator does not allow a private pool handle to be
        # reused after its last graph is reset. Preserve the sharing between
        # runners from the same compiled graph while rotating dead handles.
        replacement_pools: dict[tuple[int, int], tuple[int, int]] = {}
        for runner in runners:
            old_pool = runner.graph_pool_handle
            if old_pool not in replacement_pools:
                replacement_pools[old_pool] = torch.cuda.graph_pool_handle()
            runner.graph_pool_handle = replacement_pools[old_pool]

    def optimize(
        self,
        gm: GraphModule,
        example_inputs: List[torch.Tensor],
    ):
        graph = gm.graph
        self.match_count = []
        self.match_count_by_pass = OrderedDict()
        for custom_pass in self.custom_passes:
            total_match_count = 0
            match_count = custom_pass.apply(graph)
            self.match_count.append(match_count)
            total_match_count += match_count
            while match_count:
                match_count = custom_pass.apply(graph)
                self.match_count.append(match_count)
                total_match_count += match_count
            pass_name = custom_pass.pass_name or (
                f"unnamed_pass_{len(self.match_count_by_pass)}")
            self.match_count_by_pass[pass_name] = total_match_count
        graph.eliminate_dead_code()
        # After this pass, cannot run any dce!!!
        remove_copy_for_mutates_args(graph)

        # Do not apply multi-stream if enable piecewise cuda graph or inductor
        # For piecewise cuda graph, we will apply the multi-stream optimization in piecewise_optimizer
        # For inductor, we do not control the passes inside inductor.
        if self.num_streams > 1 and not self.piecewise_cuda_graph and not self.enable_inductor:
            num_events = multi_stream_schedule(gm, self.num_streams)
            self.generate_events(num_events)

        gm.recompile()

        if self.piecewise_cuda_graph:
            gm, num_events, runners = piecewise_optimizer(
                gm,
                example_inputs,
                self.enable_inductor,
                self.input_num_tokens,
                self.capture_num_tokens,
                self._graph_pool_handle,
                self.num_streams,
            )
            self._piecewise_runners.update(runners)
            self.generate_events(num_events)
            return gm
        elif self.enable_inductor:
            return compile_fx_inner(gm, example_inputs)
        else:
            return gm

    def __call__(self, gm: GraphModule,
                 example_inputs: List[torch.Tensor]) -> callable:

        if self.no_optimization:
            logger.warning(
                "Bypassing torch.compile optimization and fallback to eager execution!"
            )
            return gm

        self.input_num_tokens = None
        # On multimodal wrappers (e.g. Qwen2/3-VL) the LM forward is
        # invoked with `input_ids=None` and `inputs_embeds=<tensor>`,
        # so dynamo eliminates the `input_ids` placeholder; the
        # `inputs_embeds` placeholder carries the (num_tokens, H)
        # tensor whose leading dim is the same num_tokens.
        for node in gm.graph.nodes:
            if node.op == "placeholder":
                if node.name in [
                        "l_input_ids_",
                        "l_kwargs_input_ids_",
                        "l_inputs_embeds_",
                        "l_kwargs_inputs_embeds_",
                ]:
                    example_value = node.meta["example_value"]
                    assert isinstance(example_value, FakeTensor)
                    self.input_num_tokens = example_value.shape[0]
                    break

        if self.piecewise_cuda_graph:
            assert (
                self.input_num_tokens is not None
            ), "Cannot detect input_num_tokens. Cannot use piecewise CUDA graph. What is the name of `input_ids`?"

        gm = recover_pass(gm)

        return aot_module_simplified(
            gm,
            example_inputs,
            fw_compiler=self.optimize,
            decompositions=select_decomp_table(),
        )
