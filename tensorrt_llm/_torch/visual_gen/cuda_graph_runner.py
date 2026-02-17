import functools
import gc
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple, TypeAlias

import torch

from tensorrt_llm.logger import logger

from ..utils import make_weak_ref

KeyType: TypeAlias = Tuple[Tuple[int, ...] | Tuple[str, Tuple[int, ...]], ...]


@dataclass
class CUDAGraphRunnerConfig:
    """Configuration for the CUDAGraphRunner."""

    use_cuda_graph: bool
    """
    Master switch. Same three-path semantics as the LLM runner:

    1. False  → runner is dormant, maybe_get_cuda_graph always returns None
    2. True, ineligible → fallback to eager (e.g., first call with new shapes)
    3. True, eligible   → capture (if new) or replay
    """
    cuda_graph_mem_pool: Any = None


class CUDAGraphRunner:
    """
    Manages CUDA graph lifecycle for visual generation.

    Mirrors the LLM CUDAGraphRunner API:
      get_graph_key → maybe_get_cuda_graph → needs_capture → capture / replay

    Key differences from the LLM runner:
    - Keys are derived from tensor shapes (automatic), not batch metadata
    - Static buffers are per-key (allocated at capture time), not shared/pre-allocated
    - No batch padding, speculative decoding, or cross-rank coordination
    - Outputs returned via make_weak_ref (zero-copy, same as LLM runner)
    """

    WARMUP_STEPS = 2

    def __init__(self, config: CUDAGraphRunnerConfig):
        self.config = config
        self.enabled = config.use_cuda_graph

        self.graphs: Dict[KeyType, torch.cuda.CUDAGraph] = {}
        self.graph_outputs: Dict[KeyType, Any] = {}  # weak refs
        self.static_inputs: Dict[KeyType, Tuple[List[Any], Dict[str, Any]]] = {}
        self.memory_pool = config.cuda_graph_mem_pool

    def get_graph_key(self, *args, **kwargs) -> KeyType:
        input_shapes = tuple(
            list(tuple(arg.shape) for arg in args if isinstance(arg, torch.Tensor))
            + list(
                (k, tuple(kwargs[k].shape))
                for k in sorted(kwargs.keys())
                if isinstance(kwargs[k], torch.Tensor)
            )
        )
        return input_shapes

    def capture(
        self, key: KeyType, fn: Callable, args: Tuple[Any, ...], kwargs: Dict[str, Any]
    ) -> None:
        logger.info(f"Capturing graph for shapes: {key}")

        # One time clone of inputs
        static_args = [arg.clone() if isinstance(arg, torch.Tensor) else arg for arg in args]
        static_kwargs = {
            k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()
        }

        graph = torch.cuda.CUDAGraph()
        for _ in range(self.WARMUP_STEPS):
            fn(*static_args, **static_kwargs)
            torch.cuda.synchronize()
            gc.collect()
            torch.cuda.empty_cache()

        with torch.cuda.graph(graph, pool=self.memory_pool):
            output = fn(*static_args, **static_kwargs)

        self.graphs[key] = graph
        self.static_inputs[key] = (static_args, static_kwargs)
        self.graph_outputs[key] = make_weak_ref(output)
        self.memory_pool = graph.pool()

    def replay(
        self,
        key: KeyType,
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
    ) -> Any:
        static_args, static_kwargs = self.static_inputs[key]

        for i, arg in enumerate(args):
            if isinstance(arg, torch.Tensor):
                static_args[i].copy_(arg)
            else:
                static_args[i] = arg

        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                static_kwargs[k].copy_(v)
            else:
                static_kwargs[k] = v

        self.graphs[key].replay()
        return self.graph_outputs[key]

    def wrap(self, fn):
        """Wrap a callable with CUDA graph capture/replay.

        Returns a drop-in replacement that:
        - On first call with new tensor shapes: captures a graph
        - On subsequent calls with same shapes: replays the graph
        - Falls back to eager if called with positional args or if disabled
        """

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            if not self.enabled:
                return fn(*args, **kwargs)

            key = self.get_graph_key(*args, **kwargs)

            if key not in self.graphs:
                self.capture(key, fn, args, kwargs)
                return self.replay(key, args, kwargs)
            else:
                return self.replay(key, args, kwargs)

        return wrapper

    def __del__(self):
        self.clear()

    def clear(self):
        """Releases all captured graphs and the associated memory pool."""
        for graph in self.graphs.values():
            graph.reset()
        self.graphs.clear()
        self.graph_outputs.clear()
        self.static_inputs.clear()
        del self.memory_pool
        self.memory_pool = None
        torch.cuda.empty_cache()
