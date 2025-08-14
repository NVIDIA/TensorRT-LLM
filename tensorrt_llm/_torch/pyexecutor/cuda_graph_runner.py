import bisect
import contextlib
import threading
import weakref
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Tuple

import torch

from ..expert_statistic import ExpertStatistic
from ..utils import make_weak_ref, set_piecewise_cuda_graph_flag
from .resource_manager import ResourceManager, ResourceManagerType
from .scheduler import ScheduledRequests

if TYPE_CHECKING:
    from .model_engine import PyTorchModelEngine

# A large prime number used for dummy request IDs to avoid collisions
CUDA_GRAPH_DUMMY_REQUEST_ID = (1 << 64) - 1


class graph_capturing_local(threading.local):

    def __init__(self):
        self.is_graph_capturing = False


_local = graph_capturing_local()


def set_graph_capturing(enable: bool):
    _local.is_graph_capturing = enable


def is_graph_capturing() -> bool:
    return _local.is_graph_capturing


@contextlib.contextmanager
def capturing_cuda_graph_context():
    """A context manager to safely set and unset graph capturing flags."""
    set_graph_capturing(True)
    set_piecewise_cuda_graph_flag(False)
    try:
        yield
    finally:
        set_graph_capturing(False)
        set_piecewise_cuda_graph_flag(True)


class CUDAGraphRunner:
    """
    Manages the lifecycle and execution of CUDA graphs for the model engine.

    This unified class handles high-level orchestration (padding, eligibility)
    and low-level execution (capturing, resource management, replaying) for
    multiple graphs, keyed by (batch size, draft_len).
    """
    WARMUP_STEPS = 2

    def __init__(self, engine: "PyTorchModelEngine"):
        self.engine_ref = weakref.ref(engine)

        # High-level configuration
        config = engine.pytorch_backend_config
        self.enabled = config.use_cuda_graph
        self.padding_enabled = config.cuda_graph_padding_enabled
        self.supported_batch_sizes = engine._cuda_graph_batch_sizes
        self.max_supported_batch_size = engine._max_cuda_graph_batch_size
        self.max_beam_width = engine.max_beam_width
        self.spec_config = engine.spec_config

        self.graphs: Dict[Tuple[int, int], torch.cuda.CUDAGraph] = {}
        self.static_inputs: Dict[Tuple[int, int], Dict[str, torch.Tensor]] = {}
        self.graph_outputs: Dict[Tuple[int, int],
                                 Callable[[], Optional[torch.Tensor]]] = {}
        self.graph_metadata: Dict[Tuple[int, int], Dict[str, Any]] = {}
        self.memory_pool = engine._cuda_graph_mem_pool
        self.padding_dummy_request: Optional["Request"] = None

    @property
    def enable_spec_decode(self):
        return self._get_engine().is_spec_decode

    @property
    def draft_len(self):
        return self.spec_config.max_draft_len if self.enable_spec_decode else 0

    @property
    def spec_metadata(self):
        return self._get_engine().spec_metadata

    @property
    def draft_tokens_cuda(self):
        return self._get_engine().draft_tokens_cuda

    @property
    def attn_metadata(self):
        return self._get_engine().attn_metadata

    def __del__(self):
        self.clear()

    def _get_engine(self) -> "PyTorchModelEngine":
        """Safely dereferences the weak reference to the engine."""
        engine = self.engine_ref()
        if engine is None:
            raise RuntimeError(
                "The parent PyTorchModelEngine has been garbage collected.")
        return engine

    def maybe_get_cuda_graph(self, batch: ScheduledRequests):
        if not self._can_run_graph(batch):
            return False, None, None

        batch_size = len(batch.generation_requests)

        key = (batch_size, self.draft_len)
        if key in self.graphs:
            return True, self.graph_metadata[key][
                "attn_metadata"], self.graph_metadata[key]["spec_metadata"]

        if batch_size not in self.supported_batch_sizes:
            return False, None, None

        num_sequences_in_batch = batch_size * self.max_beam_width
        attn_metadata = self.attn_metadata.create_cuda_graph_metadata(
            num_sequences_in_batch, False, self.draft_len)
        assert attn_metadata.is_cuda_graph

        if self.enable_spec_decode:
            spec_metadata = self.spec_metadata.create_cuda_graph_metadata(
                num_sequences_in_batch)
            spec_metadata.draft_tokens = self.draft_tokens_cuda
        else:
            spec_metadata = None

        return True, attn_metadata, spec_metadata

    def execute(self, batch: ScheduledRequests, inputs: Dict[str, Any],
                forward_fn: Callable) -> Optional[torch.Tensor]:
        """
        Runs the model via a CUDA graph or captures it if needed.

        Returns the model output tensor if a graph was run, otherwise None.
        """
        if not self._can_run_graph(batch):
            return None

        batch_size = len(batch.generation_requests)

        if batch_size not in self.graphs:
            if batch_size in self.supported_batch_sizes:
                self._capture_graph(batch_size, forward_fn, inputs)
            else:
                return None

        return self._run_graph(batch_size, inputs)

    def _capture_graph(self, batch_size: int, forward_fn: Callable,
                       initial_inputs: Dict[str, Any]):
        """Captures the forward pass for a given batch size."""
        engine = self._get_engine()
        key = (batch_size, self.draft_len)

        max_tokens_per_req = 1
        if engine.is_spec_decode:
            max_tokens_per_req += engine.spec_config.max_draft_len

        spec_metadata = initial_inputs.get("spec_metadata", None)
        # [CUDA graph spec decode padding]
        # We pad input IDs/position IDs to the maximum draft length (token per request).
        # We're forced to do this because we cannot reallocate inputs over many graph runs.
        token_per_request = spec_metadata.max_draft_len + 1 if spec_metadata is not None else 1

        static_tensors = {
            "input_ids":
            torch.ones((batch_size * self.max_beam_width * token_per_request, ),
                       device="cuda",
                       dtype=torch.int32),
            "position_ids":
            torch.zeros((
                1,
                batch_size * self.max_beam_width * token_per_request,
            ),
                        device="cuda",
                        dtype=torch.int32),
        }
        if engine.use_mrope:
            static_tensors["mrope_position_deltas"] = torch.zeros(
                (batch_size, 1), device="cuda", dtype=torch.int32)
        self.static_inputs[key] = static_tensors

        capture_inputs = initial_inputs.copy()
        capture_inputs.update(static_tensors)

        self.graph_metadata[key] = {
            "attn_metadata": initial_inputs["attn_metadata"],
            "spec_metadata": spec_metadata,
        }

        graph = torch.cuda.CUDAGraph()
        with capturing_cuda_graph_context():
            for _ in range(self.WARMUP_STEPS):
                forward_fn(capture_inputs)

            with torch.cuda.graph(graph, pool=self.memory_pool):
                output = forward_fn(capture_inputs)

        self.graphs[key] = graph
        self.graph_outputs[key] = make_weak_ref(output)
        self.memory_pool = graph.pool()

    def _run_graph(self, batch_size: int,
                   current_inputs: Dict[str, Any]) -> Optional[torch.Tensor]:
        """Replays a previously captured graph."""
        key = (batch_size, self.draft_len)
        stored_meta = self.graph_metadata[key]
        assert current_inputs["attn_metadata"] is stored_meta["attn_metadata"]
        if stored_meta["spec_metadata"] is not None:
            assert current_inputs.get(
                "spec_metadata") is stored_meta["spec_metadata"]

        static_tensors = self.static_inputs[key]

        input_ids = current_inputs["input_ids"]
        seqlen = input_ids.shape[0]
        static_tensors["input_ids"][:seqlen].copy_(input_ids)

        position_ids = current_inputs["position_ids"]
        static_tensors["position_ids"][:, :seqlen].copy_(position_ids)

        if "mrope_position_deltas" in static_tensors:
            static_tensors["mrope_position_deltas"].copy_(
                current_inputs["mrope_position_deltas"])

        self.graphs[key].replay()
        output_ref = self.graph_outputs[key]

        return output_ref

    def _can_run_graph(self, batch: ScheduledRequests) -> bool:
        """Checks if the current batch is eligible for CUDA graph execution."""
        engine = self._get_engine()
        if not self.enabled or not batch.can_run_cuda_graph:
            return False

        if hasattr(engine, 'iter_counter') and ExpertStatistic.set_iter(
                engine.iter_counter):
            return False

        if engine.enable_attention_dp and engine.mapping.tp_size > 1:
            batch_size = len(batch.generation_requests)
            all_rank_info = engine.dist.tp_allgather(
                [batch.can_run_cuda_graph, batch_size])

            is_all_gen_only = all(info[0] for info in all_rank_info)
            is_all_bs_equal = all(info[1] == all_rank_info[0][1]
                                  for info in all_rank_info)

            if not is_all_gen_only or not is_all_bs_equal:
                return False

        return True

    def _round_up_batch_size(self, batch_size: int) -> int:
        """Finds the smallest supported graph batch size >= the given size."""
        if not self.supported_batch_sizes:
            return 0
        idx = bisect.bisect_left(self.supported_batch_sizes, batch_size)
        if idx == len(self.supported_batch_sizes):
            return 0
        return self.supported_batch_sizes[idx]

    @contextlib.contextmanager
    def pad_batch(self, scheduled_requests: ScheduledRequests,
                  resource_manager: ResourceManager):
        """Context manager to pad a batch to a graph-compatible size."""
        engine = self._get_engine()
        kv_cache_manager = resource_manager.get_resource_manager(
            engine.kv_cache_manager_key)
        padding_size = 0
        if self.padding_enabled and self._can_run_graph(scheduled_requests):
            current_batch_size = len(scheduled_requests.generation_requests)

            if current_batch_size >= self.max_supported_batch_size:
                # Already at or beyond max size, no padding up
                padded_batch_size = current_batch_size
            else:
                padded_batch_size = self._round_up_batch_size(
                    current_batch_size)

            if padded_batch_size > 0 and padded_batch_size != current_batch_size:
                padding_size = padded_batch_size - current_batch_size

                if current_batch_size + padding_size > engine.batch_size:
                    padding_size = 0

                if padding_size > 0:
                    if self.padding_dummy_request is None:
                        if kv_cache_manager.get_num_free_blocks() > 0:
                            self.padding_dummy_request = kv_cache_manager.add_dummy_requests(
                                [CUDA_GRAPH_DUMMY_REQUEST_ID],
                                is_gen=True,
                                max_num_draft_tokens=engine.max_draft_len,
                                use_mrope=engine.use_mrope,
                                max_beam_width=engine.max_beam_width)[0]
                            self.padding_dummy_request.is_cuda_graph_dummy = True
                            spec_res_mgr = resource_manager.get_resource_manager(
                                ResourceManagerType.SPEC_RESOURCE_MANAGER)
                            if spec_res_mgr:
                                spec_res_mgr.add_dummy_requests(
                                    [CUDA_GRAPH_DUMMY_REQUEST_ID])
                        else:
                            padding_size = 0

                    if self.padding_dummy_request:
                        scheduled_requests.generation_requests.extend(
                            [self.padding_dummy_request] * padding_size)
                    else:
                        padding_size = 0

        try:
            yield scheduled_requests
        finally:
            if padding_size > 0:
                scheduled_requests.generation_requests = scheduled_requests.generation_requests[:
                                                                                                -padding_size]

    def clear(self):
        """Releases all captured graphs and the associated memory pool."""
        for graph in self.graphs.values():
            graph.reset()
        self.graphs.clear()
        self.static_inputs.clear()
        self.graph_outputs.clear()
        self.graph_metadata.clear()
        del self.memory_pool
        self.memory_pool = None
        torch.cuda.empty_cache()
