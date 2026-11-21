# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import ast
import concurrent.futures
import contextlib
import copy
import dis
import enum
import errno
import fcntl
import hashlib
import importlib.metadata
import importlib.util
import inspect
import itertools
import json
import os
import struct
import sys
import time
import types
import weakref
import zlib
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache, partial
from pathlib import Path
from typing import (Any, Callable, Dict, List, NamedTuple, Optional, Set, Tuple,
                    Union)

import torch
from cuda.bindings import driver

import tensorrt_llm
from tensorrt_llm import _jit_share as jit_share
from tensorrt_llm._torch.distributed import Distributed
from tensorrt_llm._torch.modules.multi_stream_utils import no_eager_forks
from tensorrt_llm._torch.nccl_window_graph import (
    abandon_nccl_window_graph_owner, nccl_window_allocation_suppressed,
    nccl_window_graph_capture, release_nccl_window_graph_owner)
from tensorrt_llm._torch.nccl_window_tensor_scope import \
    nccl_window_tensor_scope
from tensorrt_llm._utils import (confidential_compute_enabled, local_mpi_size,
                                 mpi_disabled, nvtx_range)
from tensorrt_llm.bindings.internal.runtime import (delay_kernel,
                                                    record_global_timer)
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping

# Unique tag to avoid collisions with other comms
PP_COMM_TAG_AUTOTUNING = 30000


class DistributedTuningStrategy(enum.Enum):
    """
    Strategy for distributed tuning.
    Args:
        BROADCAST: One rank (rank 0) tunes and broadcasts results to others
        INDEPENDENT: Each rank tunes independently (default for non-comm ops)
        MERGE: All ranks participate in tuning and reach merge
        PARALLEL: All ranks participate in tuning with partial tactics
    """
    BROADCAST = "broadcast"
    INDEPENDENT = "independent"
    MERGE = "merge"
    PARALLEL = "parallel"


@dataclass(slots=True, unsafe_hash=True)
class DynamicTensorSpec:
    """
    A specification for a dynamic tensor dimension.
    Args:
        input_idx: The index of the input tensor.
        dim_idx: The index of the dimension to tune.
        gen_tuning_buckets: A tuple of values to try or a function generating values.
        map_to_tuning_buckets: A function to map dimensions to tuning buckets during inference.
            A negative result indicates the input is outside the supported tuning range.
            Such inputs miss the cache. In skip-sweep mode, a negative mapping
            for any required dimension suppresses profiling of the entire operation.
    """
    input_idx: int
    dim_idx: int
    gen_tuning_buckets: Union[Tuple[int], Callable] = ()
    map_to_tuning_buckets: Callable = lambda x: x


@dataclass(slots=True, unsafe_hash=True)
class ConstraintSpec:
    """
    A specification for a constraint on a tensor dimension.
    Args:
        input_idx: The index of the input tensor.
        dim_idx: The index of the dimension to constrain.
        infer_shape: A function to infer the shape of the dimension.
    """
    input_idx: int
    dim_idx: int
    infer_shape: Callable


@dataclass(kw_only=True)
class TuningConfig:
    """Configuration for autotuning.

    This class specifies all the tuning configurations for a single tuning process.
    Args:
        dynamic_tensor_specs (Tuple[DynamicTensorSpec]): Specifications for how different tensor dimensions
            should be tuned to optimize performance. Each spec defines:
            - Which input tensor dimension is dynamic
            - How to generate tuning values
            - How to map dimensions to tuning values during inference

            Example:
                >>> config = TuningConfig(
                ...     dynamic_tensor_specs=(
                ...         DynamicTensorSpec(
                ...             input_idx=0,
                ...             dim_idx=1,
                ...             gen_tuning_buckets=(32, 64, 128),
                ...             map_to_tuning_buckets=lambda x: ((x + 31) // 32) * 32
                ...         ),
                ...     )
                ... )
        constraint_specs (Tuple[ConstraintSpec]): Specifications for constraints on tensor dimensions.
            Each spec defines:
            - Which input tensor dimension is constrained
            - How to infer the shape of the dimension based on other dimensions

            Example:
                >>> config = TuningConfig(
                ...     constraint_specs=(
                ...         ConstraintSpec(
                ...             input_idx=1,
                ...             dim_idx=2,
                ...             infer_shape=lambda shapes: shapes[0][0] * 2
                ...         ),
                ...     )
                ... )
        tune_max_num_tokens (int): The maximum saturation number of tokens to be tuned.
            During the inference, the input tensor will be saturated with the same value. Or if
            any value is provided to the choose_one function, the input tensor will be saturated
            with the provided value.
            If not provided, the autotuner will not consider the max num tokens.
        inputs_pre_hook (Callable): A function that takes a list of input tensors, returns a list of modified input tensors.
            It is called before the input tensors are prepared for the tuning process to match the real data distribution.
        use_cold_l2_cache (bool): Whether to use cold L2 cache.
            This flag is to create circular buffer of input tensors to avoid L2 cache hits to simulate cold L2 cache.
            Notice that not all tuning processes can benefit from this feature.
        use_cuda_graph (bool): Whether to use CUDA graph for the tuning process.
        distributed_tuning_strategy (DistributedTuningStrategy): Strategy for distributed tuning.
        exclude_from_cache (bool): Skip disk persistence of tuning results for this op.
            Use for JIT-driven ops where caching the tactic across processes would skip
            the JIT warmup. In-process cache still works.
        parallel_jit_warmup (int): Before profiling, run every not-yet-tuned profile once
            on this many threads. For runners whose first call per shape JIT-compiles a
            kernel and releases the GIL while compiling (DeepGEMM), those compiles then
            overlap, and the profiling loop that follows finds them built. Opted-in
            runners must be thread-safe and must not use NCCL windows. 0 disables; -1 uses
            the CPUs this rank may run on, divided by the ranks on the node unless the
            rank is pinned to a subset, up to 32: neighbouring sweep points often select
            the same kernel and compile it twice, so more threads keep paying off.
            TLLM_AUTOTUNER_PARALLEL_JIT_WARMUP overrides the thread count.
        share_jit_builds (bool): For runners whose kernels are DeepGEMM builds or CuTe DSL
            kernels kept in their class's ``kernel_cache``. When the processes of the launch
            share their JIT builds (TLLM_JIT_SHARE_DIR, see tensorrt_llm._jit_share), each
            kernel is built by one process and loaded by the others. They split the profiles
            not yet tuned through claims in that directory: for a single tactic, each rank
            runs the profiles it claims first, setting aside one whose kernel another rank is
            building, then the profiles other ranks claimed once those have run; for several
            tactics under INDEPENDENT tuning, a build pass first runs each (profile, runner,
            tactic) once on the rank that claims it, and the profiling that follows finds the
            kernels built. The claims then take the place of the ``parallel_jit_warmup``
            thread pool, which does not run.
        cold_l2_flush_bytes (int): When > 0, every timed repeat first reads a buffer of this many
            bytes (evicting the operands, weights included, from L2) and each repeat is timed alone
            with %globaltimer stamps inside one CUDA graph; the median repeat is the tactic's time.
            For decode-sized GEMMs whose weights never stay in L2 between uses.
    """
    dynamic_tensor_specs: Tuple[DynamicTensorSpec, ...] = ()
    constraint_specs: Tuple[ConstraintSpec, ...] = ()
    tune_max_num_tokens: int = None
    inputs_pre_hook: Callable = None
    use_cold_l2_cache: bool = False
    use_cuda_graph: bool = True
    distributed_tuning_strategy: DistributedTuningStrategy = DistributedTuningStrategy.INDEPENDENT
    exclude_from_cache: bool = False
    parallel_jit_warmup: int = 0
    share_jit_builds: bool = False
    cold_l2_flush_bytes: int = 0


@dataclass(frozen=True)
class NvMMHConfig:
    """nvMatmulHeuristics (nvMMH) guidance for CuTe DSL GEMM/BMM runners.

    When installed on the AutoTuner, nvMMH-aware runners build their tactic
    list from the top ``max_tactics`` kernel configurations nvMMH ranks for
    the problem instead of enumerating their full tactic space. Every field
    nvMMH models (CTA tile, cluster shape, split-K, rasterization order,
    swizzle) is taken as returned; only knobs nvMMH does not model are swept.
    """
    max_tactics: int = 5

    def __post_init__(self):
        if (isinstance(self.max_tactics, bool)
                or not isinstance(self.max_tactics, int)
                or self.max_tactics < 1):
            raise ValueError(
                f"NvMMHConfig.max_tactics must be a positive integer, got "
                f"{self.max_tactics!r}")


@dataclass(unsafe_hash=True)
class StaticDim:
    val: int

    def _opt(self):
        return self.val


@dataclass(unsafe_hash=True)
class DynamicDim:
    '''Range of one dimension'''
    min: int
    opt: int
    max: int

    def _opt(self):
        return self.opt


Dim = Union[DynamicDim, StaticDim]


@dataclass
class OptimizationProfile:
    '''Ranges of all tensors, all dimension
    '''
    shapes: List[List[Dim]] = field(default_factory=lambda: [[]])

    def get_hash_key(self):
        return self.get_opt_shapes()

    def get_opt_shapes(self):
        '''Only the opt shapes are considered as hash key
        '''
        # TODO: remove duplicate shape generation
        opt_shapes = []
        for t in self.shapes:
            opt_shapes.append(tuple([d._opt() for d in t]))
        return tuple(opt_shapes)


#TODO: can/shall we use the torch builtin FakeTensor class?
@dataclass
class FakeTensor:
    dtype: torch.dtype
    device: torch.device
    shape: List[Dim]


class TunableRunner(ABC):

    @abstractmethod
    def get_valid_tactics(self, inputs: List[torch.Tensor],
                          profile: OptimizationProfile, **kwargs) -> List[Any]:
        """One tactic corresponding to one cuda kernel normally, but how to interpret the meaning
        of tactic is pure internal details of the runner.

        The autotuner will just pass the tactic value to the forward w/o. any knowledge on what the tactic
        means. User can choose to implement their own types of tactic for flexibility, such as using a dict-typed
        to represent a collection of named configs.

        The type of the tactic is arbitrary. But serialization/deserialization of the cache requires that the type is compatible with json.dumps/json.loads.
        To evaluate if a type of tactic is compatible with current workflow, try the following code:
            *  assert YOUR_TACTIC_OBJECT == eval(repr(YOUR_TACTIC_OBJECT))

        tactic==-1 has special meaning, means the fallback kernel which should be able to implement any shapes
        This fallback tactic is needed for 2 reasons:
            * when the autotuner cannot find a valid tactic in its cache.
            * in eager mode, w/o autotunning the custom op should have at least one kernel, which makes the autotuning
              process an optional process, such that user can opt out.

        We choose not to have a standalone can_implement function, the tactics returned by get_valid_tactics should return
        valid kernel for these given input tensors.
        """
        return [-1]

    def __call__(self, inputs, **kwargs):
        return self.forward(inputs, **kwargs)

    @abstractmethod
    def forward(
            self,
            /,  # tensors are position only
            inputs: List[torch.Tensor],
            *,  # all others are keyword args only
            tactic: Any = -1,
            do_preparation: bool = False,
            **kwargs) -> Any:
        """Forward pass for tunable runners.

        Args:
            inputs: List of input tensors (position-only argument)
            tactic: A arbitrary type that represents a specific kernel config.
                    For instance, it can be an integer number that specifies the unique ID of the implementation tactic to use.
                    -1 (default) represents the fallback tactic that must be implemented
                    to handle any input shapes when autotuning is disabled.
            do_preparation: When True, allows one-time setup operations to be performed
                          before tactic evaluation begins. These operations are excluded
                          from the performance measurements during autotuning. Notice that
                          anything prepared in this phase should be persistent in the forward
                          and can be accessed by the following forward calls.

        Returns:
            Any: Output of the forward pass.

        """
        raise NotImplementedError

    def unique_id(self):
        """
        Returns a tuple of the unique id of the runner. The unique id will be converted to a string for the cache key.
        A common practice is to return a tuple of the runner's attributes, for example:
            return (self.output_dtype, self.attribute_1, ...)

        Returns:
            Any: The unique id of the runner, which can be converted to a string for the cache key.
        """
        return tuple(self.__dict__.values())

    def tactic_space_key(self) -> Optional[Tuple]:
        """Discriminates cache entries profiled over a non-default tactic space.

        Runners whose candidate tactics depend on a process-wide search policy
        (e.g. nvMatmulHeuristics guidance) return a literal tuple describing
        that policy, so entries tuned under one policy are not reused under
        another. None keeps the default cache key.
        """
        return None


_TRACKED_PACKAGE = "tensorrt_llm"


def _tracked(module: str) -> bool:
    """Whether code of `module` counts toward a runner's fingerprint. This
    module is left out: a change to the tuner is not a change to any kernel."""
    return module.split(".")[0] == _TRACKED_PACKAGE and module != __name__


_CONSTANT_TYPES = (int, float, complex, str, bytes, bool, type(None), enum.Enum)
_DEFINITIONS = (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
_OPERATOR_TYPES = (torch._ops.OpOverload, torch._ops.OpOverloadPacket)
_SCRIPT_CLASS = getattr(torch._C, "ScriptClass", ())
# Registers Python implementations of ops through Library.define/impl.
_FAST_CUSTOM_OP_MODULE = "tensorrt_llm._torch.custom_ops.fast_custom_op"
# torch.ops / torch.classes namespaces that torch itself registers.
_TORCH_NAMESPACES = ("aten", "prim", "prims")
# When this module was imported: a tracked source file changed after it may
# not hold the code this process runs.
_EPOCH_NS = time.time_ns()
# Marks the fingerprint of a runner whose sources changed after _EPOCH_NS.
# Entries under it serve this process only: save_cache leaves them out, and
# no cache file holds an entry this process could hit with it.
_EDITED_PREFIX = "edited-"
# TLLM_AUTOTUNER_CACHE_KEY: what a cache key's fingerprint covers.
# "build" (default): the TensorRT-LLM build and installed kernel packages, the
# same on every rank of an installation. "code": also the Python code each
# runner reaches (_code_fingerprint), best effort. "none": no fingerprint, the
# key as before fingerprints.
_CACHE_KEY_MODES = ("build", "code", "none")
_CACHE_KEY_MODE = os.environ.get("TLLM_AUTOTUNER_CACHE_KEY", "build").lower()
if _CACHE_KEY_MODE not in _CACHE_KEY_MODES:
    logger.warning(f"[Autotuner] TLLM_AUTOTUNER_CACHE_KEY={_CACHE_KEY_MODE!r} "
                   f"is not one of {_CACHE_KEY_MODES}; using 'build'.")
    _CACHE_KEY_MODE = "build"
# Packages whose kernels runners launch; their versions are part of a build.
_KERNEL_PACKAGES = ("cuda", "cutlass", "deep_gemm", "flashinfer", "triton")


class _SourceIndex(NamedTuple):
    """Digests of a file and of each class, function and name assignment in
    it, by qualified name (several definitions of one name share a digest),
    with the names each definition loads, as `a.b.c` chains."""
    path: str
    mtime_ns: int
    size: int
    digest: str
    # qualname -> digest of its definition(s).
    definitions: Dict[str, str]
    # Names evaluated when the definition runs: decorators, defaults, bases,
    # class bodies and assigned values.
    eager: Dict[str, Tuple[str, ...]]
    # Every name a definition loads, its body included.
    loads: Dict[str, Tuple[str, ...]]
    # name -> (level, module, attr) of each module-level `from` import.
    imports: Dict[str, Tuple[int, str, str]]
    # name -> (digest, loads) of the statements that change, rather than
    # bind, the value of a name when the file runs: `TABLE[k] = v`,
    # `Cls.attr = v`, `X += v`, `del TABLE[k]`, `TABLE.update(...)`.
    mutations: Dict[str, Tuple[str, Tuple[str, ...]]]


class _Alias(NamedTuple):
    """The module a global name refers to, when code reaches past the name
    into the module's attributes."""
    module: str


class _External(NamedTuple):
    """A module outside the tracked code, named by an import that is not run."""
    module: str


_SOURCE_INDEXES: Dict[str, _SourceIndex] = {}


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _dotted_loads(nodes) -> List[str]:
    """The names, as `a.b.c` chains, that AST nodes load."""
    names: List[str] = []

    def walk(node):
        attrs = []
        base = node
        while isinstance(base, ast.Attribute):
            attrs.append(base.attr)
            base = base.value
        if isinstance(base, ast.Name):
            if isinstance(base.ctx, ast.Load):
                names.append(".".join([base.id, *reversed(attrs)]))
            return
        for child in ast.iter_child_nodes(base):
            walk(child)

    for node in nodes:
        if node is not None:
            walk(node)
    return list(dict.fromkeys(names))


def _evaluated_at_definition(node) -> List[ast.AST]:
    """The expressions a `def` evaluates when it runs."""
    return [
        *node.decorator_list, *node.args.defaults,
        *(d for d in node.args.kw_defaults if d is not None)
    ]


def _root_name(node) -> Optional[str]:
    """The name a chain of attributes and subscripts starts from, or None."""
    while isinstance(node, (ast.Attribute, ast.Subscript)):
        node = node.value
    return node.id if isinstance(node, ast.Name) else None


def _mutated_names(node) -> List[str]:
    """The names a statement changes the value of without binding them."""
    if isinstance(node, (ast.Assign, ast.AnnAssign)):
        targets = node.targets if isinstance(node,
                                             ast.Assign) else [node.target]
        targets = [t for t in targets if not isinstance(t, ast.Name)]
    elif isinstance(node, ast.AugAssign):
        targets = [node.target]
    elif isinstance(node, ast.Delete):
        targets = node.targets
    elif (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
          and isinstance(node.value.func, ast.Attribute)):
        targets = [node.value.func.value]  # TABLE.update(...)
    else:
        return []
    return [n for n in map(_root_name, targets) if n is not None]


_TRY_STAR = getattr(ast, "TryStar", ())


def _blocks(node):
    """(header, header expressions, statements) for each block of a compound
    statement, or None for a simple one. The header names what decides
    whether, and how often, the block's statements run (`for atom in
    range(9, 17)`, `if SM == 100`, `not if SM == 100` for an else)."""
    unparse = ast.unparse
    if isinstance(node, (ast.For, ast.AsyncFor)):
        header = f"for {unparse(node.target)} in {unparse(node.iter)}"
        return [(header, [node.iter], node.body),
                (f"{header} else", [node.iter], node.orelse)]
    if isinstance(node, ast.While):
        header = f"while {unparse(node.test)}"
        return [(header, [node.test], node.body),
                (f"{header} else", [node.test], node.orelse)]
    if isinstance(node, ast.If):
        header = f"if {unparse(node.test)}"
        return [(header, [node.test], node.body),
                (f"not {header}", [node.test], node.orelse)]
    if isinstance(node, (ast.With, ast.AsyncWith)):
        return [("with " + ", ".join(map(unparse, node.items)),
                 [item.context_expr for item in node.items], node.body)]
    if isinstance(node, (ast.Try, *((_TRY_STAR, ) if _TRY_STAR else ()))):
        return [("try", [], node.body),
                *((f"except {unparse(h.type) if h.type else ''}",
                   [h.type] if h.type else [], h.body) for h in node.handlers),
                ("try else", [], node.orelse), ("finally", [], node.finalbody)]
    if isinstance(node, ast.Match):
        return [(f"match {unparse(node.subject)} case {unparse(c.pattern)}" +
                 (f" if {unparse(c.guard)}" if c.guard else ""),
                 [node.subject, *([c.guard] if c.guard else [])], c.body)
                for c in node.cases]
    return None


def _source_index(path: str) -> _SourceIndex:
    """The index of a file, rebuilt whenever its size or mtime changes."""
    st = os.stat(path)
    index = _SOURCE_INDEXES.get(path)
    if (index is not None and index.mtime_ns == st.st_mtime_ns
            and index.size == st.st_size):
        return index
    with open(path, encoding="utf-8") as f:
        text = f.read()
    lines = text.splitlines(keepends=True)
    spans: Dict[str, List[str]] = {}
    eager: Dict[str, List[str]] = {}
    loads: Dict[str, List[str]] = {}
    imports: Dict[str, Tuple[int, str, str]] = {}
    mutations: Dict[str, List[Tuple[ast.AST, Tuple]]] = {}

    def text_of(node, headers):
        """A statement's source, after the headers of the blocks that
        decide whether and how often it runs."""
        start = min([node.lineno] +
                    [d.lineno for d in getattr(node, "decorator_list", ())])
        return "".join(h + "\n" for h, _ in headers) + "".join(
            lines[start - 1:node.end_lineno])

    def header_loads(headers):
        return [n for _, nodes in headers for n in nodes]

    def add(name, node, eager_nodes, load_nodes, headers):
        spans.setdefault(name, []).append(text_of(node, headers))
        guards = header_loads(headers)
        eager.setdefault(name, []).extend(_dotted_loads([*eager_nodes,
                                                         *guards]))
        loads.setdefault(name, []).extend(_dotted_loads([*load_nodes, *guards]))

    def visit(nodes, prefix, headers):
        for child in nodes:
            name = (prefix +
                    child.name if isinstance(child, _DEFINITIONS) else None)
            if isinstance(child, ast.ClassDef):
                nested = [s for s in child.body if isinstance(s, _DEFINITIONS)]
                evaluated = [
                    *child.decorator_list, *child.bases,
                    *(k.value for k in child.keywords),
                    *(s for s in child.body if not isinstance(s, _DEFINITIONS)),
                    *(e for s in nested if not isinstance(s, ast.ClassDef)
                      for e in _evaluated_at_definition(s))
                ]
                add(name, child, evaluated, evaluated, headers)
                visit(ast.iter_child_nodes(child), name + ".", ())
                continue
            if isinstance(child, _DEFINITIONS):
                add(name, child, _evaluated_at_definition(child), [child],
                    headers)
                visit(ast.iter_child_nodes(child), name + ".<locals>.", ())
                continue
            for mutated in _mutated_names(child):
                mutations.setdefault(prefix + mutated, []).append(
                    (child, headers))
            if isinstance(child, (ast.Assign, ast.AnnAssign)):
                targets = (child.targets
                           if isinstance(child, ast.Assign) else [child.target])
                value = [child.value]
                for target in targets:
                    if isinstance(target, ast.Name):
                        add(prefix + target.id, child, value, value, headers)
            elif isinstance(child, ast.ImportFrom) and not prefix:
                for alias in child.names:
                    imports[alias.asname
                            or alias.name] = (child.level, child.module
                                              or "", alias.name)
            blocks = _blocks(child)
            if blocks is None:
                visit(ast.iter_child_nodes(child), prefix, headers)
            else:
                for header, header_nodes, body in blocks:
                    visit(body, prefix,
                          (*headers, (header, tuple(header_nodes))))

    visit(ast.parse(text).body, "", ())
    index = _SourceIndex(path, st.st_mtime_ns, st.st_size, _digest(text), {
        k: _digest("".join(v))
        for k, v in spans.items()
    }, {
        k: tuple(dict.fromkeys(v))
        for k, v in eager.items()
    }, {
        k: tuple(dict.fromkeys(v))
        for k, v in loads.items()
    }, imports, {
        k: (_digest("".join(text_of(n, h) for n, h in v)),
            tuple(
                _dotted_loads([n for n, _ in v] +
                              [g for _, h in v for g in header_loads(h)])))
        for k, v in mutations.items()
    })
    _SOURCE_INDEXES[path] = index
    return index


def _index_of(obj) -> Optional[_SourceIndex]:
    """The index of the file that defines `obj`, or None without one."""
    try:
        path = inspect.getsourcefile(obj)
    except (TypeError,
            OSError):  # A builtin, or a class of a fileless __main__.
        return None
    if not path:
        return None
    try:
        return _source_index(path)
    except (OSError, SyntaxError, ValueError) as e:
        logger.warning_once(
            f"[Autotuner] Could not read {path} for an autotuner cache key "
            f"({e!r}); its code does not count toward the keys that reach it.",
            key=("autotuner_unreadable_source", path))
        return None


def _definition_digest(obj, index: _SourceIndex) -> str:
    """The digest of a class's or function's definition (its whole file when
    the definition cannot be found in it)."""
    digest = index.definitions.get(obj.__qualname__)
    if digest is None:
        try:
            digest = _digest(inspect.getsource(obj))
        except (OSError, TypeError):
            digest = index.digest
    return digest


def _is_extension(obj) -> bool:
    """Whether a module, or the module of a class, is native code."""
    name = obj.__name__ if inspect.ismodule(obj) else obj.__module__
    modules = [obj] if inspect.ismodule(obj) else []
    while name:
        modules.append(sys.modules.get(name))
        name = name.rpartition(".")[0]
    for module in modules:
        path = getattr(module, "__file__", None)
        if isinstance(path, str):
            return path.endswith((".so", ".pyd"))
    return False


_MISSING = object()


def _attribute(obj, name: str, default=None):
    """getattr, or `default` for an attribute that is missing or cannot load
    (a lazy stand-in for an optional package raises ImportError or
    RuntimeError)."""
    try:
        return getattr(obj, name)
    except (AttributeError, ImportError, RuntimeError):
        return default


def _torch_namespace(module_name: str) -> Optional[str]:
    """The namespace a torch.ops or torch.classes namespace module stands for
    (torch names the latter `torch.classes<namespace>`), or None."""
    for prefix in ("torch.ops.", "torch.classes"):
        if module_name.startswith(prefix) and len(module_name) > len(prefix):
            return module_name[len(prefix):].lstrip(".").partition(".")[0]
    return None


def _narrowable(module, attr: str) -> bool:
    """Whether to look `attr` up on `module`. Only tracked modules, and the
    torch.ops and torch.classes chains that lead to native code: a module
    elsewhere may import on attribute access (PEP 562), and counts by its
    package anyway."""
    name = getattr(module, "__name__", "")
    if not isinstance(name, str):
        return False
    return (_tracked(name) or (name == "torch" and attr in ("ops", "classes"))
            or name in ("torch.ops", "torch.classes")
            or _torch_namespace(name) is not None)


def _narrow(name: str, obj, attrs):
    """(name, object) for a chain of attributes: module.attr chains name the
    attribute, not the whole module. A module reached under another name
    (`import a as first`) also yields that binding, as an _Alias."""
    for attr in attrs:
        if not inspect.ismodule(obj) or not _narrowable(obj, attr):
            break
        value = _attribute(obj, attr, _MISSING)
        if value is _MISSING:
            break
        if name != obj.__name__:
            yield name, _Alias(obj.__name__)
        name, obj = f"{obj.__name__}.{attr}", value
    yield name, obj


def _resolve_loads(names, globals_: Dict[str, Any]):
    """(name, object) for `a.b.c` chains looked up in a module's globals."""
    module_name = globals_.get("__name__", "")
    for dotted in names:
        head, *attrs = dotted.split(".")
        if head in globals_:
            yield from _narrow(f"{module_name}.{head}", globals_[head], attrs)


def _load_module(name: str):
    """Tracked module `name`, imported if it is not yet, or None if it cannot
    load. The import may sit in a branch this process never runs (kernels for
    another architecture), so its failure must not fail the cache lookup."""
    module = sys.modules.get(name)
    if module is None:
        try:
            module = importlib.import_module(name)
        except Exception as e:  # noqa: BLE001 - an import runs the module's code
            # What cannot load is not run either (a kernel for another arch).
            logger.debug(f"[Autotuner] {name} does not import ({e!r}); "
                         "it does not count toward autotuner cache keys.")
            return None
    return module


class _ImportScan(NamedTuple):
    """An import statement in a code object, as names. `binds` holds
    (local, attr) for each variable it stores to; attr is None when it binds
    a module itself (`import a.b`, or with `aliased`, `import a.b as c`)."""
    name: str
    level: int
    fromlist: Any
    aliased: bool
    binds: Tuple[Tuple[Optional[str], Optional[str]], ...]


class _CodeScan(NamedTuple):
    """What a code object, and the code nested in it, loads by name. It holds
    names only, never the objects they refer to, so caching it keeps nothing
    alive: a global can hold a layer's weights."""
    globals: Tuple[Tuple[str, Tuple[str, ...]], ...]  # (global, attributes)
    imports: Tuple[_ImportScan, ...]
    locals: Tuple[Tuple[str, Tuple[str, ...]], ...]  # (variable, attributes)
    nested: Tuple["_CodeScan", ...]


_CODE_SCANS: "weakref.WeakKeyDictionary[types.CodeType, _CodeScan]" = (
    weakref.WeakKeyDictionary())


def _loaded_locals(ins) -> Tuple[str, ...]:
    """The local or cell variables an instruction loads, in order."""
    if ins.opname == "STORE_FAST_LOAD_FAST":
        return (ins.argval[1], )
    if ins.opname.startswith(("LOAD_FAST", "LOAD_DEREF")):
        return ins.argval if isinstance(ins.argval, tuple) else (ins.argval, )
    return ()


def _scan_import(instructions, i: int) -> _ImportScan:
    """The import statement whose IMPORT_NAME is instruction `i`."""
    level = instructions[i - 2].argval if i >= 2 else 0
    fromlist = instructions[i - 1].argval if i >= 1 else None
    aliased, binds, attr = False, [], None
    for ins in instructions[i + 1:]:
        if ins.opname == "IMPORT_FROM":
            aliased, attr = True, ins.argval
        elif ins.opname.startswith("STORE"):
            if fromlist is None:
                binds.append((ins.argval, None))
                break
            if attr is not None:
                binds.append((ins.argval, attr))
            attr = None
        elif fromlist is not None or ins.opname not in ("SWAP", "POP_TOP"):
            break
    if fromlist is None and not binds:
        binds.append((None, None))
    return _ImportScan(instructions[i].argval,
                       level if isinstance(level, int) else 0, fromlist,
                       aliased, tuple(binds))


def _scan_code(code: types.CodeType) -> _CodeScan:
    """What `code` loads by name (cached per code object, which every
    function a factory builds shares)."""
    scan = _CODE_SCANS.get(code)
    if scan is not None:
        return scan
    instructions = list(dis.get_instructions(code))

    def attrs_after(i):
        attrs = []
        for nxt in instructions[i + 1:]:
            if nxt.opname not in ("LOAD_ATTR", "LOAD_METHOD"):
                break
            attrs.append(nxt.argval)
        return tuple(attrs)

    imports = tuple(
        _scan_import(instructions, i) for i, ins in enumerate(instructions)
        if ins.opname == "IMPORT_NAME")
    imported = {local for imp in imports for local, _ in imp.binds if local}
    globals_, locals_ = [], []
    for i, ins in enumerate(instructions):
        if ins.opname in ("LOAD_GLOBAL", "LOAD_NAME"):
            globals_.append((ins.argval, attrs_after(i)))
            continue
        loaded = _loaded_locals(ins)
        for k, local in enumerate(loaded):
            # A variable an import here binds, or one of an enclosing code.
            if local in imported or ins.opname.startswith("LOAD_DEREF"):
                # Only the last variable an instruction loads can be followed
                # by its attributes; any other use counts the whole module.
                locals_.append(
                    (local, attrs_after(i) if k == len(loaded) - 1 else ()))
    scan = _CodeScan(
        tuple(dict.fromkeys(globals_)), imports, tuple(dict.fromkeys(locals_)),
        tuple(
            _scan_code(const) for const in code.co_consts
            if isinstance(const, types.CodeType)))
    _CODE_SCANS[code] = scan
    return scan


def _import_refs(imp: _ImportScan, package: str):
    """(local, name, object) for what an import statement binds (local is the
    variable it stores to, if any). A module outside the tracked code counts
    by its package: it is neither imported nor looked into, except that
    `torch` and its ops and classes namespaces bind their local, which a load
    follows to native code as it does through a global `torch`."""
    try:
        target = (importlib.util.resolve_name(
            "." * imp.level + imp.name, package) if imp.level else imp.name)
    except (ImportError, ValueError):
        return
    if not _tracked(target):
        yield None, target, _External(target)
        if target == "torch":
            for local, attr in imp.binds:
                if local is not None and imp.fromlist is None:
                    yield local, "torch", torch
                elif local is not None and attr in ("ops", "classes"):
                    yield local, f"torch.{attr}", getattr(torch, attr)
        return
    module = _load_module(target)
    if module is None:
        return
    if imp.fromlist is None:
        [(local, _)] = imp.binds
        if local is not None and not imp.aliased:
            top = target.partition(".")[0]  # `import a.b` binds `a`.
            yield local, top, sys.modules.get(top, module)
        else:
            yield local, target, module
        return
    for local, attr in imp.binds:
        value = _attribute(module, attr, _MISSING)
        if value is _MISSING:
            value = _load_module(f"{target}.{attr}")
            if value is None:
                continue
        yield local, f"{target}.{attr}", value


def _code_refs(scan: _CodeScan, globals_: Dict[str, Any], outer=None):
    """(name, object) for what a scanned code object, and the code nested in
    it, loads by name, looked up now: globals, attributes of global modules,
    imports, and attributes of the modules it imports into local variables
    (or that `outer`, the enclosing code's, holds)."""
    package = globals_.get("__package__") or ""
    module_name = globals_.get("__name__", "")
    # local -> every (name, module) an import in this code may bind it to.
    modules: Dict[str, List[Tuple[str, Any]]] = {
        local: list(bound)
        for local, bound in (outer or {}).items()
    }
    for imp in scan.imports:
        for local, name, obj in _import_refs(imp, package):
            if local is not None and inspect.ismodule(obj):
                # Counted where it is used, through the attributes used.
                bound = modules.setdefault(local, [])
                if (name, obj) not in bound:
                    bound.append((name, obj))
            else:
                yield name, obj
    for head, attrs in scan.globals:
        if head in globals_:
            yield from _narrow(f"{module_name}.{head}", globals_[head], attrs)
    for local, attrs in scan.locals:
        for name, module in modules.get(local, ()):
            yield from _narrow(name, module, attrs)
    for nested in scan.nested:
        yield from _code_refs(nested, globals_, modules)


def _fast_custom_op_impl(qualname: str) -> Optional[Callable]:
    """The Python implementation fast_custom_op registered for `qualname`, or
    None. Ops it registers exist only once its module is loaded."""
    lookup = _attribute(sys.modules.get(_FAST_CUSTOM_OP_MODULE),
                        "registered_python_impl")
    return lookup(qualname) if callable(lookup) else None


def _closure_functions(fn) -> List[Callable]:
    """The functions a (possibly wrapped) function holds in its closure."""
    try:
        fn = inspect.unwrap(fn)
    except ValueError:
        return []
    functions = []
    for cell in getattr(fn, "__closure__", None) or ():
        try:
            value = cell.cell_contents
        except ValueError:  # An empty cell.
            continue
        if inspect.isfunction(value):
            functions.append(value)
    return functions


def _captured(fn):
    """(variable, value) for each cell of `fn`'s closure: what a function a
    factory builds captures from it. The `__class__` cell of a method that
    calls super() is its own class, and is left out."""
    code = getattr(fn, "__code__", None)
    for var, cell in zip(code.co_freevars if code else (),
                         getattr(fn, "__closure__", None) or ()):
        if var == "__class__":
            continue
        try:
            yield var, cell.cell_contents
        except ValueError:  # An empty cell.
            continue


def _captured_definitions(fn):
    """(name, object) for the definitions `fn` captures (functions, classes,
    modules, and kernel objects that name their own definition), which a
    factory can vary from one built function to the next. Captured plain
    values are not counted, as a class attribute's value is not."""
    for var, value in _captured(fn):
        if (inspect.isfunction(value) or inspect.isclass(value)
                or inspect.ismodule(value) or isinstance(value, partial)):
            yield f"{fn.__module__}.{fn.__qualname__}.<closure>.{var}", value
        elif callable(value):
            own_module = _attribute(value, "__module__")
            own_name = _attribute(value, "__qualname__")
            if isinstance(own_module, str) and isinstance(own_name, str):
                yield f"{own_module}.{own_name}", value


def _function_identity(fn, depth: int = 0) -> Tuple:
    """A function's code and the identities of the definitions it captures:
    what tells apart functions one factory builds."""
    try:
        fn = inspect.unwrap(fn)
    except ValueError:
        pass
    captured = []
    if depth < 4:
        for var, value in _captured(fn):
            if inspect.isclass(value):
                captured.append((var, _class_identity(value)))
            elif inspect.ismodule(value):
                captured.append((var, value.__name__))
            elif callable(value):
                identity = _function_identity(value, depth + 1)
                if identity[0] is None:  # A kernel object: its own name.
                    identity = (_attribute(value, "__module__"),
                                _attribute(value, "__qualname__"))
                captured.append((var, identity))
    return getattr(fn, "__code__", None), tuple(captured)


def _function_refs(fn) -> Tuple[Tuple[str, Any], ...]:
    """What `fn` loads by name, looked up in its globals now."""
    return tuple(_code_refs(_scan_code(fn.__code__), fn.__globals__))


def _is_literal(obj, depth: int = 0) -> bool:
    if isinstance(obj, _CONSTANT_TYPES):
        return True
    if depth > 4:
        return False
    if isinstance(obj, (tuple, list, set, frozenset)):
        return all(_is_literal(v, depth + 1) for v in obj)
    if isinstance(obj, dict):
        return all(
            _is_literal(k, depth + 1) and _is_literal(v, depth + 1)
            for k, v in obj.items())
    return False


def _stable_repr(obj) -> str:
    """repr of a literal, with sets in a fixed order (their iteration order
    can change from one process to the next)."""
    if isinstance(obj, (set, frozenset)):
        return "{" + ", ".join(sorted(map(_stable_repr, obj))) + "}"
    if isinstance(obj, dict):
        return "{" + ", ".join(f"{_stable_repr(k)}: {_stable_repr(v)}"
                               for k, v in obj.items()) + "}"
    if isinstance(obj, (tuple, list)):
        return (type(obj).__name__ + "(" + ", ".join(map(_stable_repr, obj)) +
                ")")
    return repr(obj)


@lru_cache(maxsize=None)
def _packages_distributions() -> Dict[str, List[str]]:
    return importlib.metadata.packages_distributions()


@lru_cache(maxsize=None)
def _package_versions(top: str) -> str:
    """The installed versions of the distributions that provide top-level
    package `top` ('' for the standard library and unknown packages)."""
    versions = []
    # A dist-info without METADATA (an interrupted install) maps to None.
    dists = {d for d in _packages_distributions().get(top, ()) if d}
    for dist in sorted(dists):
        try:
            versions.append(f"{dist}=={importlib.metadata.version(dist)}")
        except (importlib.metadata.PackageNotFoundError, ValueError):
            continue
    return ",".join(versions)


def _elf_build_id(path: Path) -> Optional[str]:
    """The GNU build ID of an ELF file, or None."""
    with open(path, "rb") as f:
        header = f.read(64)
        if header[:4] != b"\x7fELF":
            return None
        wide = header[4] == 2
        order = "<" if header[5] == 1 else ">"
        if wide:
            (table_offset, ) = struct.unpack_from(order + "Q", header, 0x28)
            entry_size, count = struct.unpack_from(order + "HH", header, 0x3A)
        else:
            (table_offset, ) = struct.unpack_from(order + "I", header, 0x20)
            entry_size, count = struct.unpack_from(order + "HH", header, 0x2E)
        if not entry_size or not count:
            return None
        f.seek(table_offset)
        table = f.read(entry_size * count)
        for entry in range(0, entry_size * count, entry_size):
            (kind, ) = struct.unpack_from(order + "I", table, entry + 4)
            if kind != 7:  # SHT_NOTE
                continue
            offset, size = struct.unpack_from(order + ("QQ" if wide else "II"),
                                              table,
                                              entry + (0x18 if wide else 0x10))
            f.seek(offset)
            notes = f.read(size)
            pos = 0
            while pos + 12 <= len(notes):
                name_size, desc_size, note = struct.unpack_from(
                    order + "III", notes, pos)
                pos += 12
                owner = notes[pos:pos + name_size]
                pos += (name_size + 3) & ~3
                if note == 3 and owner.rstrip(b"\0") == b"GNU":
                    return notes[pos:pos + desc_size].hex()  # NT_GNU_BUILD_ID
                pos += (desc_size + 3) & ~3
    return None


def _library_id(path: Path) -> str:
    """A shared library's build ID, else the digest of its content."""
    try:
        build_id = _elf_build_id(path)
        if build_id:
            return build_id
        digest = hashlib.sha256()
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(1 << 24), b""):
                digest.update(block)
        return digest.hexdigest()
    except (OSError, struct.error) as e:
        logger.warning_once(
            f"[Autotuner] Could not identify {path} ({e!r}); a rebuild of it "
            "does not change autotuner cache keys.",
            key=("autotuner_unreadable_library", str(path)))
        return ""


@lru_cache(maxsize=None)
def _native_build_id() -> str:
    """IDs of TensorRT-LLM's native libraries, which hold its C++ kernels."""
    root = Path(tensorrt_llm.__file__).parent
    libraries = sorted([*root.glob("*.so"), *(root / "libs").glob("*.so")])
    return "\n".join(f"{path.name}:{_library_id(path)}" for path in libraries)


class _FingerprintWalk:
    """Collects the digests of what a runner class's tactics depend on."""

    def __init__(self, runner_cls: type):
        owners = [
            k for k in runner_cls.__mro__
            if issubclass(k, TunableRunner) and k is not TunableRunner
        ]
        self.homes = {k.__module__ for k in owners}
        self.todo: List[Tuple[str,
                              Any]] = [(f"{k.__module__}.{k.__qualname__}", k)
                                       for k in owners]
        self.sources: Dict[str, str] = {}
        self.packages: Set[str] = set()
        self.native = False
        self.edited = False
        self.seen: Set[int] = set()
        self.seen_names: Set[str] = set()

    def digest(self) -> str:
        while self.todo:
            self.visit(*self.todo.pop())
        for top in self.packages:
            versions = _package_versions(top)
            if versions:
                self.sources[f"<package>:{top}"] = versions
        if self.native:
            self.sources["<native>"] = _native_build_id()
        digest = hashlib.sha256()
        for label in sorted(self.sources):
            digest.update(f"{label}\0{self.sources[label]}\0".encode())
        return (_EDITED_PREFIX if self.edited else "") + digest.hexdigest()[:16]

    def counts(self, module: str) -> bool:
        return _tracked(module) or module in self.homes

    def add_package(self, module: str) -> None:
        top = module.partition(".")[0]
        if top != _TRACKED_PACKAGE:
            self.packages.add(top)

    def index(self, obj) -> Optional[_SourceIndex]:
        index = _index_of(obj)
        if index is not None and index.mtime_ns > _EPOCH_NS:
            # The disk may not hold the code this process runs: tactics tuned
            # here must neither be saved under that code's key nor be served
            # from a cache file written for it.
            self.edited = True
            logger.warning_once(
                f"[Autotuner] {index.path} changed after this process started; "
                "autotuner cache entries of the runners that use it are kept in "
                "this process only.",
                key=("autotuner_source_edited", index.path))
        return index

    def visit(self, name: str, obj) -> None:
        if isinstance(obj, _External):
            self.add_package(obj.module)
            return
        if isinstance(obj, _Alias):
            self.bind(name, obj.module)
            return
        if isinstance(obj, _SCRIPT_CLASS):
            self.native = True  # A torch.classes class, registered in C++.
            return
        if isinstance(obj, (staticmethod, classmethod)):
            obj = obj.__func__
        if isinstance(obj, property):
            self.todo.extend(
                (name, f) for f in (obj.fget, obj.fset, obj.fdel) if f)
            return
        if isinstance(obj, _OPERATOR_TYPES):
            self.visit_operator(getattr(obj, "overloadpacket", obj))
            return
        if isinstance(obj, partial):
            self.todo.append((f"{name}[func]", obj.func))
            self.todo.extend(
                (f"{name}[{i}]", v)
                for i, v in enumerate((*obj.args, *obj.keywords.values())))
        elif callable(obj) and not inspect.isclass(obj):
            try:
                obj = inspect.unwrap(obj)
            except ValueError:
                pass  # A cycle of __wrapped__: keep the outermost callable.
        if inspect.ismodule(obj):
            self.bind(name, obj.__name__)
            namespace = _torch_namespace(obj.__name__)
            if namespace is None:
                self.visit_definition(obj)
            elif namespace in _TORCH_NAMESPACES:
                self.packages.add("torch")
            else:
                # A namespace of native operators or classes, reached when
                # the operator or class itself is not (yet) registered.
                self.native = True
        elif inspect.isclass(obj) or inspect.isfunction(obj):
            if isinstance(obj.__module__, str):
                self.bind(name, f"{obj.__module__}.{obj.__qualname__}")
            self.visit_definition(obj)
        else:
            self.visit_value(name, obj)

    def bind(self, name: str, target: str) -> None:
        """Record that global `name` refers to `target`, so swapping which
        definition two names refer to changes the digest even though every
        definition's source stays the same."""
        if target != name and name.rpartition(".")[0] in sys.modules:
            self.sources[f"<ref>:{name}"] = target

    def visit_operator(self, packet) -> None:
        """A torch.ops operator: its Python implementations if it has them
        (from torch.library.custom_op or fast_custom_op), else the native
        library that registers it."""
        qualname = getattr(packet, "_qualified_op_name", "")
        if qualname in self.seen_names:
            return
        self.seen_names.add(qualname)
        custom_ops = _attribute(_attribute(torch, "_library"), "custom_ops")
        definition = (_attribute(custom_ops, "OPDEFS") or {}).get(qualname)
        if definition is not None:
            # The default implementation and each device's registered kernel;
            # torch keeps a kernel in the closure of a wrapper.
            implementation = _attribute(definition, "_init_fn")
            if callable(implementation):
                self.todo.append((qualname, implementation))
            for wrapper in (_attribute(definition, "_backend_fns")
                            or {}).values():
                self.todo.extend(
                    (qualname, f) for f in _closure_functions(wrapper))
        elif callable(fast_impl := _fast_custom_op_impl(qualname)):
            self.todo.append((qualname, fast_impl))
        elif qualname.partition("::")[0] in _TORCH_NAMESPACES:
            self.packages.add("torch")
        else:
            self.native = True

    def visit_definition(self, obj) -> None:
        if id(obj) in self.seen:
            return
        self.seen.add(id(obj))
        module = obj.__name__ if inspect.ismodule(obj) else obj.__module__
        if not isinstance(module, str):
            return
        if not self.counts(module):
            self.add_package(module)
            return
        index = self.index(obj)
        if index is None:
            self.native = self.native or _is_extension(obj)
            return
        if inspect.ismodule(obj):
            # Counted whole: with every definition of its own and every name
            # it imports with `from` (a caller can reach both through the
            # module), and what those use.
            self.sources[f"{module}:<module>"] = index.digest
            self.todo.extend((f"{module}.{k}", v) for k, v in vars(obj).items()
                             if k in index.definitions or k in index.imports)
            return
        self.sources[f"{module}:{obj.__qualname__}"] = _definition_digest(
            obj, index)
        if module in sys.modules:
            self.add_mutations(module, obj.__qualname__, index)
        globals_ = (obj.__globals__ if inspect.isfunction(obj) else getattr(
            sys.modules.get(module), "__dict__", {}))
        self.todo.extend(
            _resolve_loads(index.eager.get(obj.__qualname__, ()), globals_))
        if inspect.isfunction(obj):
            self.todo.extend(_function_refs(obj))
            self.todo.extend(_captured_definitions(obj))
            return
        self.todo.extend((f"{module}.{obj.__qualname__}.{k}", v)
                         for k, v in vars(obj).items())
        self.todo.extend(
            (f"{b.__module__}.{b.__qualname__}", b) for b in obj.__mro__[1:])

    def locate(self, module: str, attr: str):
        """(module, attr, index) of the definition of global `attr` of
        `module`, following `from` imports to the module that defines it.
        Module-level statements that change the value, in each module on the
        way, count with it."""
        for _ in range(16):
            if module not in sys.modules or not self.counts(module):
                return module, attr, None
            index = self.index(sys.modules[module])
            if index is not None:
                self.add_mutations(module, attr, index)
            if (index is None or attr in index.definitions
                    or attr not in index.imports):
                return module, attr, index
            level, source, attr = index.imports[attr]
            try:
                module = (importlib.util.resolve_name(
                    "." * level + source, sys.modules[module].__package__)
                          if level else source)
            except (ImportError, ValueError):
                return module, attr, None
        return module, attr, None

    def add_mutations(self, module: str, name: str,
                      index: _SourceIndex) -> None:
        """Count the module-level statements of `module` that change global
        `name` after it is bound, and the names they load."""
        mutation = index.mutations.get(name)
        if mutation is not None:
            digest, loads = mutation
            self.sources[f"{module}:{name}<changed>"] = digest
            self.todo.extend(_resolve_loads(loads, vars(sys.modules[module])))

    def visit_value(self, name: str, obj) -> None:
        """A value reached by name: it counts by the source that defines it
        and by the names that source loads."""
        module, _, attr = name.rpartition(".")
        if module not in sys.modules:
            # A class attribute: the class's source and the names its body
            # loads cover it.
            return
        if name in self.seen_names:
            return
        self.seen_names.add(name)
        module, attr, index = self.locate(module, attr)
        self.bind(name, f"{module}.{attr}")
        if not self.counts(module):
            self.add_package(module)
            return
        if index is not None and attr in index.definitions:
            self.sources[f"{module}:{attr}"] = index.definitions[attr]
            self.todo.extend(
                _resolve_loads(index.loads[attr], vars(sys.modules[module])))
            return
        if index is None and module in sys.modules and _is_extension(
                sys.modules[module]):
            self.native = True
            return
        # No definition in the source (a value set some other way): count
        # the value itself.
        label = f"{module}.{attr}"
        if _is_literal(obj):
            self.sources[label] = _stable_repr(obj)
        elif isinstance(obj, (tuple, list)):
            self.todo.extend((f"{label}[{i}]", v) for i, v in enumerate(obj))
        elif isinstance(obj, dict):
            self.todo.extend(
                (f"{label}[{i}]", v)
                for i, v in enumerate(itertools.chain(*obj.items())))
        else:
            own_module = _attribute(obj, "__module__")
            own_name = _attribute(obj, "__qualname__")
            if (isinstance(own_module, str) and isinstance(own_name, str)
                    and f"{own_module}.{own_name}" != label):
                self.todo.append((f"{own_module}.{own_name}", obj))
            elif self.counts(type(obj).__module__):
                self.todo.append(
                    (f"{type(obj).__module__}.{type(obj).__qualname__}",
                     type(obj)))


# (module, qualname) -> code fingerprint of that runner class.
_FINGERPRINTS: Dict[Tuple, str] = {}
# Identities of a set of candidate runner classes -> its fingerprint.
_CANDIDATE_FINGERPRINTS: Dict[Tuple, str] = {}
_LOCAL_CLASS_IDENTITIES: "weakref.WeakKeyDictionary[type, Tuple]" = (
    weakref.WeakKeyDictionary())
_CLASS_IDENTITIES_IN_PROGRESS: "weakref.WeakSet[type]" = weakref.WeakSet()
# (module, qualname) -> a runner class's fingerprint under "build".
_BUILD_RUNNER_FINGERPRINTS: Dict[Tuple[str, str], str] = {}


def _class_name(cls: type) -> Tuple[str, str]:
    return cls.__module__, cls.__qualname__


def _class_identity(cls: type) -> Tuple:
    """What tells runner classes apart for the fingerprint caches. A class of
    a module counts by its name. A class a function builds (`<locals>` in
    its name) can be built again, alike or not: it also counts by what the
    walk follows in it, its bases and the classes and code of its namespace
    with what that code captures, so classes one factory builds alike share
    a fingerprint (walked once), and classes it builds on different bases or
    around different captured implementations do not."""
    if "<locals>" not in cls.__qualname__:
        return _class_name(cls)
    identity = _LOCAL_CLASS_IDENTITIES.get(cls)
    if identity is None and cls in _CLASS_IDENTITIES_IN_PROGRESS:
        # A method that names its own class (`Runner.tactics`) captures it.
        return _class_name(cls)
    if identity is None:
        _CLASS_IDENTITIES_IN_PROGRESS.add(cls)
        try:
            identity = _local_class_identity(cls)
        finally:
            _CLASS_IDENTITIES_IN_PROGRESS.discard(cls)
    return identity


def _local_class_identity(cls: type) -> Tuple:
    """_class_identity of a class a function builds, computed once."""
    identity = _LOCAL_CLASS_IDENTITIES.get(cls)
    if identity is None:
        members = []
        for key, value in sorted(vars(cls).items()):
            if isinstance(value, (staticmethod, classmethod)):
                value = value.__func__
            values = ((value.fget, value.fset,
                       value.fdel) if isinstance(value, property) else
                      (value, ))
            for value in values:
                if inspect.isclass(value):
                    members.append((key, _class_identity(value)))
                elif callable(value):
                    member = _function_identity(value)
                    if member[0] is not None:
                        members.append((key, member))
        identity = _LOCAL_CLASS_IDENTITIES[cls] = (*_class_name(cls),
                                                   tuple(
                                                       map(
                                                           _class_identity,
                                                           cls.__bases__)),
                                                   tuple(members))
    return identity


def _coarse_fingerprint(runner_cls: type) -> str:
    """A fingerprint that does not walk the code: the runner's name, the
    TensorRT-LLM and torch versions and the native build. It is the same on
    every rank."""
    text = "\0".join(
        (*_class_name(runner_cls), tensorrt_llm.__version__, torch.__version__,
         _native_build_id()))
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _code_fingerprint(runner_cls: type) -> str:
    """A digest of the code a runner's tactics depend on, in its cache keys
    under TLLM_AUTOTUNER_CACHE_KEY=code. It is best effort: what Python code
    reaches only at run time (a callable a runner instance holds, getattr) is
    not followed.

    It starts from the runner and its TunableRunner bases and follows, one
    definition at a time, what they reach by name in tensorrt_llm code
    (TensorRT-LLM itself, and the modules that define the runner):

    * a function counts by its source, the globals, module attributes and
      imports its code loads, and the names its decorators and defaults
      evaluate;
    * a class by its source, its bases, its attributes, and the names its body
      evaluates;
    * any other value (a constant, table, kernel object or instance) by the
      source that defines it, following `from` imports to the module that
      does, and by the names that source loads; a value without a source
      definition counts by its own repr when it is a literal;
    * a value, class or function also by the module-level statements that
      change it after it is bound (`TABLE[k] = v`, `Cls.attr = v`,
      `TABLE.update(...)`), and the names those load;
    * a torch.ops operator by its Python implementation, else by the build IDs
      of TensorRT-LLM's native libraries, as does a native module or class;
    * code outside tensorrt_llm by the installed version of its package; such
      modules are neither imported nor looked into for a key;
    * each global name used, by the definition it refers to.

    A TRT-LLM change thus re-tunes only the runners whose code it touches; a
    module that cannot be narrowed counts whole. Sources are read from disk,
    so every process running the same files computes the same fingerprint.
    It is cached by the class's identity (_class_identity), not by the class
    object, so a runner class a function builds alike on every call is walked
    once and not kept alive. A source changed after this module was imported
    may not be the code this process runs: the fingerprint is then marked as
    this process's (_EDITED_PREFIX), and entries under it are not saved. If
    the walk fails, the runner is keyed by the TensorRT-LLM build instead.
    """
    identity = _class_identity(runner_cls)
    fingerprint = _FINGERPRINTS.get(identity)
    if fingerprint is None:
        name = _class_name(runner_cls)
        try:
            fingerprint = _FingerprintWalk(runner_cls).digest()
        except Exception as e:  # noqa: BLE001 - must not fail the op it keys
            logger.warning_once(
                f"[Autotuner] Could not fingerprint the code of "
                f"{'.'.join(name)} ({e!r}); its cache entries are keyed by "
                "the TensorRT-LLM build instead.",
                key=("autotuner_fingerprint_failed", name))
            fingerprint = _coarse_fingerprint(runner_cls)
        _FINGERPRINTS[identity] = fingerprint
    return fingerprint


@lru_cache(maxsize=None)
def _build_fingerprint() -> str:
    """A digest of the TensorRT-LLM build: its version, its native libraries'
    build IDs, and the installed versions of torch and the kernel packages.
    It is the same on every rank of an installation, and changes with a
    rebuild (even under the same version string) or a package upgrade."""
    parts = [
        f"tensorrt_llm=={tensorrt_llm.__version__}",
        f"torch=={torch.__version__}",
        _native_build_id(),
        *map(_package_versions, _KERNEL_PACKAGES),
    ]
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()[:16]


def _runner_fingerprint(runner_cls: type) -> str:
    """A runner class's own fingerprint: under TLLM_AUTOTUNER_CACHE_KEY=code
    its code fingerprint; else the build fingerprint and its module and
    qualified name."""
    if _CACHE_KEY_MODE == "code":
        return _code_fingerprint(runner_cls)
    name = _class_name(runner_cls)
    fingerprint = _BUILD_RUNNER_FINGERPRINTS.get(name)
    if fingerprint is None:
        fingerprint = _BUILD_RUNNER_FINGERPRINTS[name] = hashlib.sha256(
            "\0".join((_build_fingerprint(), *name)).encode()).hexdigest()[:16]
    return fingerprint


def _candidates_fingerprint(runner_classes) -> Optional[str]:
    """The fingerprint in the key of an entry tuned among the runners of
    `runner_classes`: the runner's own when it was the only candidate, else a
    digest over every candidate's, so that a change to a runner that lost
    also re-opens the comparison. None when TLLM_AUTOTUNER_CACHE_KEY=none
    leaves fingerprints out of keys."""
    if _CACHE_KEY_MODE == "none":
        return None
    class_key = _class_identity if _CACHE_KEY_MODE == "code" else _class_name
    identities = (_CACHE_KEY_MODE,
                  *dict.fromkeys(map(class_key, runner_classes)))
    fingerprint = _CANDIDATE_FINGERPRINTS.get(identities)
    if fingerprint is None:
        fingerprints = sorted({_runner_fingerprint(k) for k in runner_classes})
        fingerprint = fingerprints[0]
        if len(fingerprints) > 1:
            edited = any(f.startswith(_EDITED_PREFIX) for f in fingerprints)
            fingerprint = (_EDITED_PREFIX if edited else "") + hashlib.sha256(
                "\0".join(fingerprints).encode()).hexdigest()[:16]
        _CANDIDATE_FINGERPRINTS[identities] = fingerprint
    return fingerprint


def _tactic_spaces(candidates) -> str:
    """For several candidates of which any has a tactic space key, a digest
    of every candidate's; else ''. A runner that lost and then widens its
    tactic space re-opens the comparison."""
    if len(candidates) < 2:
        return ""
    spaces = [r.tactic_space_key() for r in candidates]
    if all(space is None for space in spaces):
        return ""
    text = repr(
        sorted((_class_name(type(r)), repr(space))
               for r, space in zip(candidates, spaces)))
    return hashlib.sha256(text.encode()).hexdigest()[:8]


def _key_fingerprint(runner_cls: type,
                     set_fingerprint: Optional[str],
                     tactic_spaces: str = "") -> Optional[str]:
    """The fingerprint in the key of `runner_cls`'s entry, tuned among runners
    whose set has `set_fingerprint`: that alone for a runner tuned alone,
    else the set's, the runner's own and the candidates' tactic spaces
    (_tactic_spaces). Two candidates of one set can share a class name and
    unique_id; their own fingerprints keep their entries apart where they
    differ (always under "code"; under "build", when the module or qualified
    name does)."""
    if set_fingerprint is None:
        return None
    own = _runner_fingerprint(runner_cls)
    if own == set_fingerprint:
        return own
    return f"{set_fingerprint}/{own}" + (f"/{tactic_spaces}"
                                         if tactic_spaces else "")


def _edited_key(key: Tuple) -> bool:
    """Whether a cache key carries a fingerprint marked _EDITED_PREFIX."""
    return (len(key) > 4 and isinstance(key[4], str)
            and key[4].startswith(_EDITED_PREFIX))


def _clear_fingerprints() -> None:
    _build_fingerprint.cache_clear()
    _FINGERPRINTS.clear()
    _CANDIDATE_FINGERPRINTS.clear()
    _BUILD_RUNNER_FINGERPRINTS.clear()


# What lru_cache offered, for tests that model a new process.
_code_fingerprint.cache_clear = _FINGERPRINTS.clear
_candidates_fingerprint.cache_clear = _clear_fingerprints


@contextlib.contextmanager
def autotune(tune_mode: bool = True,
             cache_path: str = None,
             skip_dynamic_tuning_buckets: bool = False,
             post_tune_merge_dist: Optional[Distributed] = None):
    """Context manager for autotuning with distributed support.

    Args:
        tune_mode: Whether to enable tuning mode
        cache_path: Path to save/load cache files
        skip_dynamic_tuning_buckets: When True, suppress bucket generation in
            _optimization_profiles() and profile the encountered shapes and
            their mapped buckets, subject to tune_max_num_tokens. Inputs with
            no mapped bucket use the fallback without profiling. Useful for
            workloads (e.g. diffusion) where the LLM-oriented M-bucket sweep
            is unnecessary.
        post_tune_merge_dist: Optional ``Distributed``. When set, tactics are
            tuned per-rank independently and merged across ranks in one
            collective at context exit (before the cache is saved) — for
            pipelines whose warmup is not SPMD and so cannot use the per-op
            distributed strategies.
    """
    autotuner = AutoTuner.get()
    if post_tune_merge_dist is not None:
        # Save the singleton's prior state: the full-world dist attached for the
        # merge is temporary and must not leak into later sessions. Attach the
        # real (world-sized) mapping now for a correct rank, but with no dist
        # yet: _is_distributed() stays False so per-op sync is skipped while a
        # non-SPMD pipeline tunes; the dist is attached at exit to merge.
        prev_mapping, prev_dist = autotuner.mapping, autotuner._dist
        autotuner.setup_distributed_state(post_tune_merge_dist.mapping, None)
    rank = autotuner.mapping.rank

    # if cache_path is provided, use the rank-specific file
    tune_required = tune_mode
    if cache_path is not None:
        # check if the rank-specific file exists
        # if the rank-specific file exists, load it
        file_exists = os.path.exists(cache_path)
        if file_exists:
            logger.info(f"[Autotuner] Loading cache from {cache_path}")
            autotuner.profiling_cache.load_cache(cache_path, rank)

    # record the old tuning mode
    old_mode = autotuner.is_tuning_mode
    old_skip = autotuner.skip_dynamic_tuning_buckets
    autotuner.is_tuning_mode = tune_required
    autotuner.skip_dynamic_tuning_buckets = skip_dynamic_tuning_buckets
    # Avoid making compiled AllReduce paths initialize or inspect AutoTuner.
    from tensorrt_llm._torch.distributed.ops import \
        set_allreduce_autotuner_tuning_mode
    set_allreduce_autotuner_tuning_mode(tune_required)
    autotune_enabled = tune_required and not old_mode

    if autotune_enabled:
        logger.info("[Autotuner] Autotuning process starts ...")

    # Fine-grained sync kernels deadlock when profiled, so force them off for the duration via the
    # C++ override rather than mutating os.environ. TODO: remove once autotuning handles them.
    if autotune_enabled:
        torch.ops.trtllm.set_fine_grained_sync_disabled_override(True)

    # While tuning, CuTe DSL kernels in shared runner caches go through the
    # directory of TLLM_JIT_SHARE_DIR (see tensorrt_llm._jit_share).
    kernel_sharing = (jit_share.sharing_kernels() if tune_required
                      and jit_share.enabled() else contextlib.nullcontext())
    eager_forks_context = (no_eager_forks()
                           if tune_required else contextlib.nullcontext())

    try:
        with kernel_sharing, eager_forks_context:
            yield
    finally:
        autotuner.is_tuning_mode = old_mode
        autotuner.skip_dynamic_tuning_buckets = old_skip
        set_allreduce_autotuner_tuning_mode(old_mode)
        if autotune_enabled:
            AutoTuner.L2_FLUSH_BUFFERS.clear()
            torch.ops.trtllm.set_fine_grained_sync_disabled_override(False)
            logger.info("[Autotuner] Autotuning process ends")

        try:
            # Merge tactics across ranks in one collective before the cache is
            # saved, so non-SPMD pipelines can tune independently and still agree.
            if post_tune_merge_dist is not None:
                autotuner.setup_distributed_state(post_tune_merge_dist.mapping,
                                                  post_tune_merge_dist)
                autotuner.post_tune_merge_tactics()

            # save cache
            if cache_path is not None:
                logger.info(f"[Autotuner] Saving cache to {cache_path}")
                autotuner.profiling_cache.save_cache(cache_path, rank)
        finally:
            # Restore the singleton's prior distributed state (see enter) so the
            # temporary full-world state does not persist past this context.
            if post_tune_merge_dist is not None:
                autotuner.mapping = prev_mapping
                autotuner._dist = prev_dist


@dataclass
class AutoTunerStatistics:
    """Statistics collected by the AutoTuner.

    Attributes:
        cache_misses (int): Number of cache misses requiring fallback
        cache_miss_config_collection (Dict[str, Set[OptimizationProfile]]): Collection of configs that caused cache misses
        failed_profiling_count (Dict[str, int]): Number of failed profiling attempts per operation
        tuned_op_profiled_configs (Dict[str, int]): Profiled configurations per operation
        tuned_op_time_cost (Dict[str, float]): Time cost per operation
    """
    cache_misses: int = 0
    cache_miss_config_collection: Dict[str,
                                       Set[tuple]] = field(default_factory=dict)
    failed_profiling_count: Dict[str, Set[Tuple[str, TunableRunner,
                                                OptimizationProfile]]] = field(
                                                    default_factory=dict)
    tuned_op_profiled_configs: Dict[str, int] = field(default_factory=dict)
    tuned_op_time_cost: Dict[str, float] = field(default_factory=dict)

    def __str__(self) -> str:
        """Return a string representation of collected statistics.
        """
        stats_str = ""
        stats_str += f"Cache misses: {self.cache_misses}\n"
        if self.cache_miss_config_collection:
            stats_str += "Cache miss config collection:\n"
            for op, profiles in sorted(
                    self.cache_miss_config_collection.items()):
                stats_str += f"  {op}:\n"
                for profile in sorted(profiles, key=str):
                    stats_str += f"    - Config: {profile}\n"

        if self.tuned_op_profiled_configs:
            stats_str += "Tuned operations:\n"
            for op in sorted(self.tuned_op_profiled_configs.keys()):
                successful = self.tuned_op_profiled_configs[op]
                failed = len(self.failed_profiling_count[op])
                stats_str += f"  {op}:\n"
                stats_str += f"    - Successful configs: {successful}\n"
                stats_str += f"    - Failed profiling count: {failed}\n"
                if failed > 0:
                    stats_str += f"    - Failed profiling combinations:\n"
                    for failed_key in self.failed_profiling_count[op]:
                        stats_str += f"      - {failed_key}\n"

        if self.tuned_op_time_cost:
            stats_str += "Tuned operations time cost:\n"
            for op in sorted(self.tuned_op_time_cost.keys()):
                stats_str += f"  {op}: {self.tuned_op_time_cost[op] * 1000:.4f} milliseconds\n"

        return stats_str


# POSIX ESTALE and the Linux NFS client's EBADHANDLE (521).
_STALE_HANDLE_ERRNOS = (errno.ESTALE, 521)


def _read_cache_text(file_path: Path, attempts: int = 5) -> str:
    """Read the cache file, retrying reads that hit a stale NFS handle."""
    for attempt in range(attempts):
        try:
            with open(file_path, "r") as f:
                return f.read()
        except OSError as e:
            if e.errno not in _STALE_HANDLE_ERRNOS or attempt == attempts - 1:
                raise
            logger.warning(
                f"[AutoTuner] Retrying cache read of {file_path} after {e!r}")
            time.sleep(0.2 * (attempt + 1))
    raise AssertionError("unreachable")


def _fastest_entries(per_rank: List[Dict[Tuple, Tuple]]) -> Dict[Tuple, Tuple]:
    """Per cache key, the fastest of the ranks' entries (ties: the lowest rank's)."""
    merged = {}
    for entries in per_rank:
        for key, value in entries.items():
            if value[-1] < merged.get(key, (float('inf'), ))[-1]:
                merged[key] = value
    return merged


def _majority_entries(per_rank: List[Dict[Tuple, Tuple]]) -> Dict[Tuple, Tuple]:
    """Per cache key, the entry whose tactic most ranks hold (ties: the lowest rank's)."""
    merged = {}
    for key in {key for entries in per_rank for key in entries}:
        held = [entries[key] for entries in per_rank if key in entries]
        votes = Counter(repr(value[1]) for value in held)
        top = max(votes.values())
        merged[key] = next(value for value in held
                           if votes[repr(value[1])] == top)
    return merged


@contextlib.contextmanager
def _exclusive_cache_lock(lock_path: Path):
    """Cross-node-correct exclusive lock used by save_cache/load_cache.

    POSIX byte-range lock (`fcntl.lockf`) held on a SEPARATE lockfile
    (not the data file), which avoids the truncate / rename complications
    of locking the file you're also rewriting. The lockfile is persistent;
    the kernel releases the lock when the holding fd is closed — including
    on process death, so there is no stale-lock cleanup path to maintain.

    Both readers and writers use LOCK_EX: read/write frequencies on the
    autotuner cache are similar and the simpler "always exclusive" model
    avoids reader/writer-priority pitfalls on networked filesystems.
    """
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.lockf(fd, fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.lockf(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


class AutoTunerProfilingCache:
    """AutoTunerCache for caching profiling results.

    The profiling cache can be serialized to disk for persistence across sessions:
        - Use save_cache() to save the cache after tuning
        - Use load_cache() to restore cached results before inference
        - JSON format provides human-readable output and cross-platform compatibility

    Cache organization:
        - Ops with INDEPENDENT strategy are stored per-rank (rank_0, rank_1, ...)
        - Ops with non-INDEPENDENT strategy (BROADCAST, MERGE, PARALLEL) are stored
          in a shared dict since all ranks share the same tuning results
    """

    # Key for shared cache entries (non-INDEPENDENT ops)
    SHARED_CACHE_KEY = "shared"

    def __init__(self):
        self.cache: Dict[Tuple, Tuple] = dict()

        # Track which ops use which distributed strategy
        # Maps custom_op name -> DistributedTuningStrategy
        self.independent_op: Set[str] = set()

        # Ops whose tuning results must NOT be persisted to disk (see the
        # `exclude_from_cache` field on TuningConfig). Populated lazily
        # from choose_one when an excluded op is first seen.
        self.excluded_op: Set[str] = set()

        # Cache metadata for local storage and validation
        self.lib_version = tensorrt_llm.__version__
        self.creation_timestamp = time.time()
        # gpu_platform
        self.device_name = torch.cuda.get_device_name()
        self.device_capability = torch.cuda.get_device_capability()

    def __setitem__(self, cache_key: Tuple, value: Tuple) -> None:
        self.cache[cache_key] = value

    def __getitem__(self, cache_key: Tuple) -> Tuple:
        return self.cache[cache_key]

    def __len__(self) -> int:
        return len(self.cache)

    def clear(self) -> None:
        self.cache.clear()
        self.independent_op.clear()
        self.excluded_op.clear()

    def fallback_entry(self) -> Tuple:
        # runner_id = 0, tactic = -1
        return 0, -1, float('inf')

    def search_cache(
        self,
        custom_op: str,
        runners: List[TunableRunner],
        input_shapes: Tuple[torch.Size],
        tuning_config: TuningConfig,
        apply_map_to_tuning_buckets: bool = True,
    ) -> Tuple[bool, int, int, Dict[str, Any], OptimizationProfile]:
        """Search for cached profiling results matching the current configuration.

        Args:
            custom_op (str): The name of the custom operation to be tuned
            runners (List[TunableRunner]): List of candidate implementations to profile
            profile (OptimizationProfile): Optimization profile
            apply_map_to_tuning_buckets: If True, apply map_to_tuning_buckets for runtime cache lookups.
                If False, use raw bucket values for tuning cache storage.

        Returns:
            A tuple containing:
            [is_cache_hit, runner_id, tactic, stored_profile]
            runner_id is the index in the current runners list

        An entry hits only when it was tuned among the same runner classes:
        their code fingerprints are part of its key.
        """
        # Under PARALLEL each rank of the tuning group times a share of the
        # (runner, tactic) pairs and records its own fastest, so after the
        # merge a shape can hold one entry per runner: the winner is the
        # fastest of them, not the first runner in the list.
        parallel = (tuning_config.distributed_tuning_strategy ==
                    DistributedTuningStrategy.PARALLEL)
        found = None
        set_fingerprint = _candidates_fingerprint(
            tuple(type(r) for r in runners))
        tactic_spaces = _tactic_spaces(runners)
        for idx, r in enumerate(runners):
            if (cache_key := self._cache_key(
                    custom_op, r, input_shapes, tuning_config,
                    apply_map_to_tuning_buckets,
                    _key_fingerprint(type(r), set_fingerprint,
                                     tactic_spaces))) in self.cache:
                # Return the current index in runners list, not the cached runner_id
                cached_runner_id, tactic, min_time = self.cache[cache_key]
                if not parallel:
                    return True, idx, tactic, min_time
                if found is None or min_time < found[3]:
                    found = (True, idx, tactic, min_time)
        if found is not None:
            return found

        return False, *self.fallback_entry()

    def get_cache_key(
        self,
        custom_op: str,
        runner: TunableRunner,
        input_shapes: Tuple[torch.Size],
        tuning_config: TuningConfig,
        apply_map_to_tuning_buckets: bool = True,
        candidates: Optional[List[TunableRunner]] = None,
    ) -> Tuple:
        """The key of `runner`'s entry for `input_shapes`:
        `(custom_op, runner class name, unique_id, shape profile,
        fingerprint[, tactic space key])`. `candidates` are the runners it was
        tuned among (only `runner` when omitted); the fingerprint covers them
        and `runner` (_key_fingerprint) under TLLM_AUTOTUNER_CACHE_KEY's mode,
        and is left out under "none"."""
        candidates = candidates or [runner]
        set_fingerprint = _candidates_fingerprint(
            tuple(type(r) for r in candidates))
        return self._cache_key(
            custom_op, runner, input_shapes, tuning_config,
            apply_map_to_tuning_buckets,
            _key_fingerprint(type(runner), set_fingerprint,
                             _tactic_spaces(candidates)))

    def _cache_key(self, custom_op: str, runner: TunableRunner,
                   input_shapes: Tuple[torch.Size], tuning_config: TuningConfig,
                   apply_map_to_tuning_buckets: bool,
                   fingerprint: Optional[str]) -> Tuple:
        cache_key = (
            custom_op,
            runner.__class__.__name__,
            str(runner.unique_id()),
            AutoTuner.get()._find_nearest_profile(
                input_shapes,
                tuning_config.dynamic_tensor_specs,
                tuning_config.constraint_specs,
                tuning_config.tune_max_num_tokens,
                apply_map_to_tuning_buckets,
            ),
        )
        if fingerprint is not None:
            cache_key += (fingerprint, )
        tactic_space_key = runner.tactic_space_key()
        if tactic_space_key is not None:
            cache_key += (tactic_space_key, )
        return cache_key

    def entries_of(self, custom_op: str, runner_cls: type):
        """(unique_id, profile, entry) for each entry of `custom_op` won by
        `runner_cls` tuned as the only candidate, under its current code
        (under TLLM_AUTOTUNER_CACHE_KEY=none, each entry of the key layout
        without a fingerprint).
        Entries of an older key layout or code fingerprint (loaded from an
        older cache file) are left out: their tactics may index another
        candidate list."""
        fingerprint = _candidates_fingerprint((runner_cls, ))
        for key, entry in self.cache.items():
            if key[:2] != (custom_op, runner_cls.__name__):
                continue
            if (len(key) == 4 if fingerprint is None else len(key) > 4
                    and key[4] == fingerprint):
                yield key[2], key[3], entry

    def merge_cache_data(self, cache_data: Dict[Tuple, Tuple]):
        self.cache.update(cache_data)

    def get_specific_custom_op(self, custom_op: str) -> Dict[Tuple, Tuple]:
        return {k: v for k, v in self.cache.items() if k[0] == custom_op}

    @staticmethod
    def shareable(entries: Dict[Tuple, Tuple]) -> Dict[Tuple, Tuple]:
        """`entries` other processes may use: without those keyed under an
        edited source's fingerprint, which serve their own process only."""
        return {k: v for k, v in entries.items() if not _edited_key(k)}

    def add_independent_op(self, custom_op: str,
                           strategy: DistributedTuningStrategy) -> None:
        if strategy != DistributedTuningStrategy.INDEPENDENT:
            self.independent_op.add(custom_op)

    def add_excluded_op(self, custom_op: str) -> None:
        """Mark `custom_op` as excluded from disk persistence.

        Entries for excluded ops still live in the in-memory cache (so
        within one process the lookup-and-reuse path works normally),
        but `_partition_cache_by_strategy` will drop them at `save_cache`
        time so they are never written to disk. Combined with the
        force-retune branch in `AutoTuner.choose_one`, this guarantees
        the tuning process (and any JIT warmup it triggers as a side
        effect) re-runs on every process startup.
        """
        self.excluded_op.add(custom_op)

    def _partition_cache_by_strategy(
            self) -> Tuple[Dict[Tuple, Tuple], Dict[Tuple, Tuple]]:
        """Partition cache entries into shared and rank-specific caches.

        Returns:
            A tuple of (shared_cache, rank_cache) where:
            - shared_cache: entries for non-INDEPENDENT ops (BROADCAST, MERGE, PARALLEL)
            - rank_cache: entries for INDEPENDENT ops
            Entries belonging to `self.excluded_op` are dropped from both
            partitions so they never reach disk.
        """
        shared_cache = {}
        rank_cache = {}

        for key, value in self.cache.items():
            custom_op = key[0]  # First element of cache key is custom_op name
            if custom_op in self.excluded_op:
                continue
            if custom_op not in self.independent_op:
                rank_cache[key] = value
            else:
                shared_cache[key] = value

        return shared_cache, rank_cache

    def save_cache(self, file_path: Union[str, Path], rank: int) -> None:
        """Save the profiling cache to disk in JSON format.

        Cache entries are organized based on distributed strategy:
        - INDEPENDENT ops are saved per-rank (rank_0, rank_1, ...)
        - Non-INDEPENDENT ops (BROADCAST, MERGE, PARALLEL) are saved in a shared dict
        - Ops marked `exclude_from_cache=True` are dropped entirely.

        Args:
            file_path: Path where to save the cache
            rank: The rank of the current process

        Raises:
            IOError: If file cannot be written

        Note:
            The cache is saved in JSON format which provides human-readable output.
            Some type information may be lost for complex tactic objects.

            Concurrency model:
            * Cross-process mutual exclusion uses POSIX byte-range locks
              (`fcntl.lockf`) on a SEPARATE lockfile next to the data
              file. The lockfile is persistent (never unlinked); the
              kernel auto-releases the lock when the holding fd is
              closed, including on crash, so no stale-lock cleanup is
              needed. Both readers and writers take LOCK_EX — read/write
              frequencies are similar here and the simpler logic is
              worth more than read parallelism.
            * The data file itself is replaced atomically via
              (write-to-tmp + fsync + rename). A writer killed at any
              point leaves a discardable .tmp file; `file_path` is never
              partially written, so concurrent readers always see a
              fully-formed JSON document.
            * The on-disk rank-specific dict is MERGED (not replaced)
              with this writer's contribution so a concurrent writer
              that loaded earlier does not get its entries clobbered.
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = self._lock_path_for(file_path)

        try:
            # Partition cache into shared (non-INDEPENDENT) and rank-specific (INDEPENDENT)
            shared_cache, rank_cache = self._partition_cache_by_strategy()

            serialized_shared_cache = self._serialize_cache_data(shared_cache)
            serialized_rank_cache = self._serialize_cache_data(rank_cache)

            with _exclusive_cache_lock(lock_path):
                current_cache = self._read_existing_cache(file_path)

                # Merge shared cache entries (non-INDEPENDENT ops)
                if self.SHARED_CACHE_KEY not in current_cache:
                    current_cache[self.SHARED_CACHE_KEY] = {}
                current_cache[self.SHARED_CACHE_KEY].update(
                    serialized_shared_cache)

                # Merge rank-specific cache entries (INDEPENDENT ops).
                # MUST be a merge (not assignment): a concurrent writer
                # that committed its rank_{rank} contribution between
                # this writer's load_cache and save_cache would otherwise
                # be silently dropped by an `=` assignment that re-writes
                # the slot with only this writer's in-memory cache.
                rank_key = f"rank_{rank}"
                if rank_key not in current_cache:
                    current_cache[rank_key] = {}
                current_cache[rank_key].update(serialized_rank_cache)

                self._atomic_write_json(file_path, current_cache)
            logger.info(
                f"[AutoTuner] Successfully saved cache to {file_path} using JSON format"
            )
        except Exception as e:
            logger.error(f"[AutoTuner] Failed to save cache with JSON: {e}")
            raise

    @staticmethod
    def _lock_path_for(file_path: Path) -> Path:
        """Return the lockfile path that pairs with `file_path`."""
        return file_path.with_name(file_path.name + ".lock")

    def _read_existing_cache(self, file_path: Path) -> Dict[str, Any]:
        """Read the data file (best-effort) under the caller's lock."""
        if not file_path.exists():
            return {"metadata": self._serialize_metadata()}
        try:
            content = _read_cache_text(file_path)
            if not content.strip():
                return {"metadata": self._serialize_metadata()}
            return json.loads(content)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            # An earlier writer from a pre-fix version may have left a
            # corrupted file behind. Don't fail the current save — start
            # from a fresh dict and overwrite atomically.
            logger.warning(f"[AutoTuner] Could not parse existing cache at "
                           f"{file_path}: {e!r}; starting from a fresh dict.")
            return {"metadata": self._serialize_metadata()}

    @staticmethod
    def _atomic_write_json(file_path: Path, payload: Dict[str, Any]) -> None:
        """Write `payload` to `file_path` atomically (tmp + fsync + rename)."""
        tmp_path = file_path.with_name(
            f".{file_path.name}.tmp.{os.getpid()}.{time.time_ns()}")
        with open(tmp_path, "w") as f:
            json.dump(payload, f, indent=2, default=str)
            f.flush()
            os.fsync(f.fileno())
        os.rename(tmp_path, file_path)

    def load_cache(self, file_path: Union[str, Path], rank: int) -> None:
        """Load the profiling cache from disk in JSON format.

        Loads both shared cache entries (non-INDEPENDENT ops) and rank-specific
        entries (INDEPENDENT ops) and merges them into the current cache.

        Args:
            file_path: Path to the cache file
            rank: The rank of the current process

        Raises:
            FileNotFoundError: If cache file doesn't exist
            IOError: If file cannot be read

        Note:
            Loading will replace the current cache contents. The cache is loaded
            from JSON format.
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Cache file not found: {file_path}")

        lock_path = self._lock_path_for(file_path)
        try:
            with _exclusive_cache_lock(lock_path):
                current_cache_contents = json.loads(_read_cache_text(file_path))
            self._deserialize_metadata(
                current_cache_contents.get("metadata", {}))

            # Start with empty cache and independent ops set
            self.cache = {}
            self.independent_op = set()

            # Load shared cache entries (non-INDEPENDENT ops)
            if self.SHARED_CACHE_KEY in current_cache_contents:
                shared_cache = self._deserialize_cache_data(
                    current_cache_contents[self.SHARED_CACHE_KEY])
                self.cache.update(shared_cache)
                # add custom op in shared cache to independent ops set
                for key in shared_cache.keys():
                    self.independent_op.add(key[0])
                logger.debug(
                    f"[AutoTuner] Loaded {len(shared_cache)} shared cache entries"
                )

            # Load rank-specific cache entries (INDEPENDENT ops)
            rank_key = f"rank_{rank}"
            if rank_key in current_cache_contents:
                rank_cache = self._deserialize_cache_data(
                    current_cache_contents[rank_key])
                self.cache.update(rank_cache)
                logger.debug(
                    f"[AutoTuner] Loaded {len(rank_cache)} rank-specific cache entries for rank {rank}"
                )

            logger.info(
                f"[AutoTuner] Successfully loaded cache from {file_path} using JSON format (total {len(self.cache)} entries)"
            )

            logger.info(
                f"[AutoTuner] independent_op: {type(self.independent_op) if hasattr(self, 'independent_op') else 'not found'}"
            )
        except Exception as e:
            logger.error(f"[AutoTuner] Failed to load cache with JSON: {e}")
            raise

    def _serialize_metadata(self) -> Dict[str, Any]:
        return {
            "lib_version": self.lib_version,
            "creation_timestamp": self.creation_timestamp,
            "device_name": self.device_name,
            "device_capability": self.device_capability,
        }

    def _deserialize_metadata(self, metadata: Dict[str, Any]) -> None:
        self.lib_version = metadata["lib_version"]
        self.creation_timestamp = metadata["creation_timestamp"]
        self.device_name = metadata["device_name"]
        self.device_capability = metadata["device_capability"]

    def _serialize_cache_data(self,
                              cache: Optional[Dict[Tuple, Tuple]] = None
                              ) -> Dict[str, Any]:
        """Convert the profiling cache to a JSON-serializable format.

        Args:
            cache: Optional cache dict to serialize. If None, uses self.cache.

        Returns:
            Dictionary that can be serialized to JSON

        Note:
            This method handles the conversion of complex objects to JSON-compatible
            representations. Some type information may be lost in the conversion.
        """
        if cache is None:
            cache = self.cache

        serializable_cache = {}

        for key, value in cache.items():
            if _edited_key(key):
                continue  # Tuned against code the disk may no longer hold.
            # Convert any simple object to string for JSON compatibility
            key_str = str(key)
            runner_id, tactic, min_time = value
            # Enum tactics (e.g. Fp4QuantTactic) repr as "<Fp4QuantTactic.TRTLLM: -1>",
            # which ast.literal_eval can't parse back -> load_cache crash. Serialize
            # the underlying value (reload yields that value; an IntEnum compares
            # equal to it and kernels take the int).
            is_enum = isinstance(tactic, enum.Enum)
            tactic_check = tactic.value if is_enum else tactic
            tactic_str = repr(tactic_check)
            try:
                # Verify the serialized value round-trips (compare against the
                # value we store, not the enum member — else non-IntEnum fails).
                assert tactic_check == ast.literal_eval(
                    tactic_str
                ), f"Tactic is not compatible with json.dumps/json.loads"
            except Exception as e:
                logger.warning_once(
                    f"[AutoTuner] Could not serialize tactic: {tactic_str} for cache key {key_str} due to {e}. Deserialization may fail.",
                    key=tactic_str)

            serializable_cache[key_str] = {
                "runner_id": runner_id,
                "tactic": tactic_str,
                "min_time": min_time,
            }

        return serializable_cache

    def _deserialize_cache_data(
            self, cache_data: Dict[str, Any]) -> Dict[Tuple, Tuple]:
        """Convert JSON-serialized cache back to the original format.

        Args:
            serializable_cache: Dictionary loaded from JSON

        Returns:
            Profiling cache in the original format

        Note:
            This attempts to reconstruct the original data structures but may not
            perfectly preserve all type information, especially for complex tactic objects.
        """
        cache = {}

        for key_str, value in cache_data.items():
            # Reconstruct the tuple key safely
            try:
                key = ast.literal_eval(key_str)
            except (ValueError, SyntaxError):
                logger.warning(
                    f"[AutoTuner] Could not reconstruct cache key: {key_str}")
                continue
            try:
                tactic = ast.literal_eval(value["tactic"])
            except (ValueError, TypeError, SyntaxError):
                # Skip any tactic whose repr is not a Python literal so one bad
                # entry can't crash the whole load; the op just re-profiles.
                # (#16782 removes the current enum trigger at the serialize side.)
                # SyntaxError is caught too; `continue` is required, else the
                # entry reuses the previous tactic.
                logger.warning_once(
                    f"[AutoTuner] Could not deserialize tactic: {value['tactic']} for cache key {key_str}; skipping entry.",
                    key=value["tactic"])
                continue

            runner_id = value["runner_id"]
            min_time = value["min_time"]

            cache[key] = (runner_id, tactic, min_time)

        return cache


def _spec_in_bounds(spec, shapes_list) -> bool:
    """Validate that spec.input_idx and spec.dim_idx are in range for shapes_list.

    On out-of-bounds access, log a warning and return False. Otherwise return True.
    Pass the SHAPE LIST (list-of-lists or equivalent indexable), not the wrapping
    object. Both ``OptimizationProfile.shapes`` and the raw list-of-lists
    ``base_profile`` used inside :meth:`AutoTuner._find_nearest_profile` are valid
    inputs because the helper only uses ``len(...)`` and indexing.

    Args:
        spec: A ``DynamicTensorSpec`` or ``ConstraintSpec`` with ``input_idx`` and
            ``dim_idx`` attributes. The spec's class name is used in the warning
            message.
        shapes_list: An indexable container of per-input shape lists.
    """
    spec_kind = type(spec).__name__
    if spec.input_idx < 0 or spec.input_idx >= len(shapes_list):
        logger.warning(
            f"[Autotuner] Skipping {spec_kind} with input_idx={spec.input_idx}: "
            f"only {len(shapes_list)} inputs available.")
        return False
    if spec.dim_idx < 0 or spec.dim_idx >= len(shapes_list[spec.input_idx]):
        logger.warning(
            f"[Autotuner] Skipping {spec_kind} with dim_idx={spec.dim_idx} for "
            f"input {spec.input_idx}: shape has only "
            f"{len(shapes_list[spec.input_idx])} dims.")
        return False
    return True


def _default_parallel_jit_warmup_threads() -> int:
    """Threads for ``parallel_jit_warmup=-1``: this rank's CPUs, up to 32.

    The compilers the warmup starts inherit this process's CPU affinity. A rank
    pinned to a subset of the host's CPUs has that subset to itself; unpinned
    ranks share all of them, so each takes its share by the ranks on the node.
    """
    try:
        cpus = len(os.sched_getaffinity(0))
    except AttributeError:  # no affinity API (non-Linux)
        cpus = os.cpu_count() or 1
    if cpus >= (os.cpu_count() or cpus):
        local_ranks = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
        if not mpi_disabled():
            try:
                local_ranks = local_mpi_size()
            except Exception:  # noqa: BLE001 - MPI cannot split by node here
                pass
        cpus //= max(1, local_ranks)
    return min(32, cpus)


class AutoTuner:
    """AutoTuner for optimizing TensorRT LLM operations.

    This class handles automatic performance tuning of tensor operations by profiling
    different implementations and caching the best performing configurations.

    Args:
        warmup (int): Number of warmup iterations before profiling (default: 3)
        repeat (int): Number of profiling iterations for averaging (default: 10)
        stream_delay_micro_secs (int): Delay on CUDA stream before the profiled kernel runs in microseconds (default: 1000)
    """
    _CUDA_GRAPH_DELAY_MICRO_SECS = 100
    _instance = None

    def __init__(self, warmup=2, repeat=10, stream_delay_micro_secs=1000):
        # Increase log level for AutoTuner associated logger`
        self._log_level_to_info = os.getenv(
            "TLLM_AUTOTUNER_LOG_LEVEL_DEBUG_TO_INFO", '0') == '1'
        self._debug_logger = logger.info if self._log_level_to_info else logger.debug

        self.repeat = repeat
        self.warmup = warmup
        self.stream_delay_micro_secs = stream_delay_micro_secs
        self.profiling_cache = AutoTunerProfilingCache()
        self._primed_cached_tactics: set[tuple[str, str, str, str]] = set()
        # Cache keys whose runner ran in this process (so any JIT it drives
        # has happened here), as opposed to keys loaded from disk or peers.
        self._keys_profiled_in_process: set[Tuple] = set()
        self.is_tuning_mode = False
        self.skip_dynamic_tuning_buckets = False
        # nvMatmulHeuristics guidance for CuTe DSL GEMM/BMM runners; None
        # profiles their full tactic space.
        self.nvmmh_config: Optional[NvMMHConfig] = None

        # Timing backend: globaltimer kernel vs cuda events.
        # TLLM_PROFILING_TIMER env var overrides auto-detection:
        #   "globaltimer" -> force globaltimer
        #   "cuda_event"  -> force cuda events
        #   unset/default -> auto-detect via confidential_compute_enabled()
        timer_env = os.getenv("TLLM_PROFILING_TIMER", "").lower()
        if timer_env == "globaltimer":
            self._use_global_timer = True
        elif timer_env == "cuda_event":
            self._use_global_timer = False
        else:
            self._use_global_timer = confidential_compute_enabled()

        logger.debug(f"[Autotuner] use_global_timer: {self._use_global_timer}")

        # Add statistics tracking
        self.stats = AutoTunerStatistics()

        # Current captured choose_one() contexts
        self._active_capture: Optional['AutoTuner.TacticsCapture'] = None
        # Last captured choose_one() contexts
        self._last_capture: Optional['AutoTuner.TacticsCapture'] = None

        # CUDA graph memory pool shared by the profiling captures of one
        # choose_one() tuning pass (see _shared_profile_graph_pool).
        self._profile_graph_pool: Optional[torch.cuda.MemPool] = None
        self._profile_graph_pool_depth = 0

        # Dsitributed tuning state
        self._dist: Optional[Distributed] = None
        self._has_received_cache: bool = False
        self.mapping: Mapping = Mapping()
        # Depth of this rank's _profile_runners_impl calls, which may run on this rank alone.
        self._profiling_depth = 0

    @classmethod
    def get(cls):
        if cls._instance is None:
            cls._instance = AutoTuner()
        return cls._instance

    class TacticsCapture:
        """Object returned by capture() that can be iterated to get all tactic combinations.

        This class encapsulates all state related to capturing and replaying tactics:
        - Captured execution contexts
        - Generated tactic configurations
        - Current replay state (which config and call index)
        """

        runner_tactic_comb_checkers: List[Callable] = []

        def __init__(self, autotuner):
            # State for captured contexts
            self._captured_contexts: List[Dict[str, Any]] = []
            self._context_tactics_lists: Optional[List[List[Tuple[int,
                                                                  Any]]]] = None
            # State for replay mode
            self._replay_runner_tactic_list: Optional[List[Tuple[int,
                                                                 int]]] = None
            self._replay_context_idx: int = 0

        def __iter__(self):
            """Iterate through all tactic configurations.

            For single context: yields (runner, tactic)
            For multiple contexts: yields ((runner_ctx0, tactic_ctx0), (runner_ctx1, tactic_ctx1), ...)
            """
            if self._context_tactics_lists is None:
                self._context_tactics_lists = self._generate_context_tactics_lists(
                )

            # Generate cartesian product from context and tactics where all_configrations[i][ctx] = (runner, tactic)
            # Such that each element in all_configrations is a replay of multiple contexts of all possible replays
            for config in itertools.product(*self._context_tactics_lists):
                # config is a tuple of (runner_idx, tactic) for each context
                # Convert to (runner, tactic) format for user
                runner_tactic_pairs = []
                for ctx_idx, (runner_idx, tactic) in enumerate(config):
                    runners = self._captured_contexts[ctx_idx]['runners']
                    runner = runners[runner_idx]
                    runner_tactic_pairs.append((runner, tactic))

                if not all(
                        checker(runner_tactic_pairs) for checker in
                        self.__class__.runner_tactic_comb_checkers):
                    continue

                yield tuple(runner_tactic_pairs)

        def _generate_context_tactics_lists(self):
            """Generate all valid tactic combinations."""
            if not self._captured_contexts:
                raise RuntimeError(
                    "No context available for testing.\n"
                    "Use capture() to capture the operation context first:\n"
                    "  with AutoTuner.get().capture() as tactics_capture:\n"
                    "      output = operation.forward(...)\n")

            # Collect valid tactics for each context separately
            context_tactics_lists = []

            for context in self._captured_contexts:
                runners = context['runners']
                inputs = context['inputs']
                kwargs = context.get('kwargs', {})

                # Collect all valid (runner, tactic) combinations for this context
                tactics_lists = []
                for runner_idx, runner in enumerate(runners):
                    valid_tactics = runner.get_valid_tactics(
                        inputs, OptimizationProfile(), **kwargs)
                    for tactic in valid_tactics:
                        tactics_lists.append((runner_idx, tactic))
                context_tactics_lists.append(tactics_lists)

            return context_tactics_lists

        def is_replaying(self) -> bool:
            """Check if this TacticsCapture is currently in replay mode."""
            return self._replay_runner_tactic_list is not None

        @classmethod
        def register_runner_tactic_comb_checker(cls, checker: Callable):
            cls.runner_tactic_comb_checkers.append(checker)
            return checker

    def choose_one(
        self,
        custom_op: str,
        runners: List[TunableRunner],
        tuning_config: TuningConfig,
        inputs: List[torch.Tensor],
        **kwargs,
    ) -> Tuple:
        """Choose the best runner and tactic combination through performance profiling.

        Args:
            custom_op (str): The name of the custom operation to be tuned
            runners (List[TunableRunner]): List of candidate implementations to profile
            tuning_config (TuningConfig): Configuration for the tuning process
            inputs (List[torch.Tensor]): Input tensors for profiling
            **kwargs: Arbitrary keyword arguments, will be passed to get_valid_tactics and forward method of each runner

        Returns:
            Tuple: A tuple containing:
                - The selected runner implementation
                - The best tactic ID for that runner (-1 if using fallback)
                - The best config for that runner (if configs is not empty)

        Note:
            The method profiles different implementations and tactics to find the
            optimal combination based on performance measurements. It caches results
            to avoid redundant profiling of the same configuration.
            Although runners[0] with tactic=-1 is always treated as the fallback runner.
            Runner authors are suggested to provide a fallback implementation for each runner to avoid potential issues.
        """

        # Check if we're in replay mode via active TacticsCapture
        if self._active_capture is not None and self._active_capture.is_replaying(
        ):
            tactics_capture = self._active_capture
            call_idx = tactics_capture._replay_context_idx

            assert call_idx < len(tactics_capture._replay_runner_tactic_list
                                  ), "call_idx out of range"
            assert call_idx < len(
                tactics_capture._captured_contexts), "call_idx out of range"
            assert len(tactics_capture._replay_runner_tactic_list) == len(
                tactics_capture._captured_contexts)

            # Check if we have a forced tactic for this call and both custom_op match
            captured_custom_op = tactics_capture._captured_contexts[
                call_idx].get('custom_op')
            if captured_custom_op != custom_op:
                raise RuntimeError(
                    f"Custom op mismatch in kernel testing mode.\n"
                    f"Expected operation: '{captured_custom_op}'\n"
                    f"Actual operation: '{custom_op}'\n"
                    f"Context index: {call_idx}\n"
                    f"Make sure the forward() call in test mode uses the same operation as captured."
                )

            runner_idx, tactic = tactics_capture._replay_runner_tactic_list[
                call_idx]
            # Increment context counter
            tactics_capture._replay_context_idx += 1
            # Reset counter after all contexts have been used
            if tactics_capture._replay_context_idx >= len(
                    tactics_capture._replay_runner_tactic_list):
                tactics_capture._replay_context_idx = 0
            return (runners[runner_idx], tactic)

        # Capture context for testing all underlying kernels
        if self._active_capture is not None and not self._active_capture.is_replaying(
        ):
            self._active_capture._captured_contexts.append({
                'custom_op': custom_op,
                'runners': runners,
                'tuning_config': tuning_config,
                'inputs': inputs,
                'kwargs': kwargs,
            })

        input_shapes = tuple(self._get_input_sizes(inputs))

        # For ops opted out of the persistent cache (e.g. JIT-driven
        # kernels whose "tuning" triggers compilation as a side effect),
        # mark them in the cache so save_cache filters them out, and
        # — in tuning mode — force the full profile path so the JIT
        # warmup actually runs even if a stale on-disk entry was loaded.
        # In inference mode we still consult the in-process cache, which
        # was populated by this process's own warmup.
        if tuning_config.exclude_from_cache:
            self.profiling_cache.add_excluded_op(custom_op)
        is_cache_hit, best_runner_id, best_tactic, min_time = (
            self.profiling_cache.search_cache(custom_op,
                                              runners,
                                              input_shapes,
                                              tuning_config,
                                              apply_map_to_tuning_buckets=True))
        if (tuning_config.exclude_from_cache and self.is_tuning_mode
                and is_cache_hit and self.profiling_cache.get_cache_key(
                    custom_op,
                    runners[best_runner_id],
                    input_shapes,
                    tuning_config,
                    apply_map_to_tuning_buckets=True,
                    candidates=runners) not in self._keys_profiled_in_process):
            # Only an entry this process profiled proves its JIT side
            # effects already ran here; re-profile anything else.
            is_cache_hit = False
            best_runner_id, best_tactic, min_time = (
                self.profiling_cache.fallback_entry())

        # Early return if it's not tuning, use cache found one or fallback one
        if not self.is_tuning_mode:
            best_runner = runners[best_runner_id]
            # TODO: check the stored runner and tactic can implement this shape here
            # Log the cache miss. Expect no cache miss in inference.
            if not is_cache_hit:
                logger.warning_once(
                    f"[AutoTuner] {custom_op} using the fallback tactic, due to cache miss on input shapes={input_shapes}",
                    key=(custom_op, "warning_autotuning_cache_miss_fallback"))
            if logger.level == 'debug':
                fine_grained_on = os.environ.get("TLLM_USE_FINE_GRAINED_SYNC",
                                                 "0") == "1"
                logger.debug_once(
                    f"[Autotuner] Inference dispatch: custom_op={custom_op}, runner={best_runner}, "
                    f"tactic={best_tactic}, fine_grained={'ON' if fine_grained_on else 'OFF'}",
                    key=(custom_op, "inference_dispatch_fine_grained"))
            return (best_runner, best_tactic)

        # Collective strategies tune in lockstep, but ranks can load different
        # caches from a shared file: tune unless every rank has the entry.
        if self._needs_rank_agreement(
                tuning_config.distributed_tuning_strategy):
            is_cache_hit = self._agree_on_cache_hit(runners, is_cache_hit)

        # If it's tuning mode and cache hit, return the best runner and tactic to avoid redundant profiling.
        if self.is_tuning_mode and is_cache_hit:
            self._prime_cached_tactics(custom_op, runners, tuning_config,
                                       inputs, **kwargs)
            return (runners[best_runner_id], best_tactic)

        # PP rank does not have cache hit, so we try to receive the cache from the previous rank
        # Notice that only under tuning mode, pp_recv will be called
        self.cache_pp_recv()

        assert len(runners) > 0, "At least one runner is required"
        assert all([isinstance(r, TunableRunner) for r in runners]), \
            "All Given runners must be subclass of TunableRunner"

        # Record the distributed tuning strategy for the custom_op
        self.profiling_cache.add_independent_op(
            custom_op, strategy=tuning_config.distributed_tuning_strategy)

        tuning_start_time = time.perf_counter()
        profiles = self._optimization_profiles(tuning_config, inputs)

        # Initialize the statistics for the custom_op
        if custom_op not in self.stats.tuned_op_profiled_configs:
            self.stats.tuned_op_profiled_configs[custom_op] = 0
        if custom_op not in self.stats.failed_profiling_count:
            self.stats.failed_profiling_count[custom_op] = set()
        new_tuning_failure_occurred = False

        strategy = tuning_config.distributed_tuning_strategy
        if self._should_current_rank_tune(strategy):

            def is_cached(p: OptimizationProfile) -> bool:
                return self.profiling_cache.search_cache(
                    custom_op,
                    runners,
                    p.get_opt_shapes(),
                    tuning_config,
                    apply_map_to_tuning_buckets=False,
                )[0]

            # Every rank profiles under MERGE / PARALLEL, so they agree on the profiles to tune.
            agreed_hits = None
            if (self._needs_rank_agreement(strategy)
                    and strategy != DistributedTuningStrategy.BROADCAST):
                agreed_hits = dict(
                    zip([p.get_opt_shapes() for p in profiles],
                        self._profile_hits_on_all_ranks(
                            profiles, [is_cached(p) for p in profiles])))

            def is_pending(p: OptimizationProfile) -> bool:
                if agreed_hits is not None:
                    return not agreed_hits[p.get_opt_shapes()]
                return not is_cached(p)

            def profile_one(p: OptimizationProfile) -> None:
                nonlocal new_tuning_failure_occurred
                if is_pending(p):
                    tensors = self._prepare_input_tensors(p, inputs)
                    # Initialize runner and tactic as None in case of no valid tactic or runners are found
                    with nvtx_range(f"{custom_op}, shape {p.get_opt_shapes()}"):
                        *_, has_tuning_failure_occurred = self._profile_runners(
                            custom_op, runners, tensors, p, tuning_config,
                            **kwargs)
                    new_tuning_failure_occurred = new_tuning_failure_occurred or has_tuning_failure_occurred

            share_builds = (tuning_config.share_jit_builds
                            and jit_share.enabled())
            one_pair = share_builds and self._one_pair_per_profile(
                runners, profiles, inputs, tuning_config, is_pending, **kwargs)
            if share_builds:
                jit_share.share_kernel_caches(runners)
                if (not one_pair and tuning_config.distributed_tuning_strategy
                        == DistributedTuningStrategy.INDEPENDENT):
                    self._build_across_ranks(custom_op, runners, profiles,
                                             inputs, tuning_config, is_pending,
                                             **kwargs)
            else:
                self._parallel_jit_warmup(custom_op, runners, profiles, inputs,
                                          tuning_config, **kwargs)
            with self._shared_profile_graph_pool():
                if one_pair:
                    self._profile_sharing_jit_builds(custom_op, profiles,
                                                     profile_one, is_pending)
                else:
                    # Profiling waits for the kernels other processes still build in
                    # the build pass, all of them together for one wait at most.
                    with (jit_share.wait_until(time.time() +
                                               jit_share.wait_seconds())
                          if share_builds else contextlib.nullcontext()):
                        for p in profiles:
                            profile_one(p)

        self._maybe_sync_cache_data(tuning_config.distributed_tuning_strategy,
                                    custom_op)
        self._prime_cached_tactics(custom_op, runners, tuning_config, inputs,
                                   **kwargs)

        # If failed profiling tactics occurs, log the error.
        if new_tuning_failure_occurred:
            logger.warning_once(
                f"[Autotuner] New tuning error occurs:"
                f"Total failed profiling tactics occurs: {len(self.stats.failed_profiling_count[custom_op])} for custom_op={custom_op}. "
                f"This will not block the tuning process. "
                f"Please set TLLM_LOG_LEVEL=WARNING to find out when the tactic profiling fails. "
                f"Set TLLM_LOG_LEVEL=DEBUG to get more details of the failures.",
                key=(custom_op, "warning_autotuning_tuning_error_summary"),
            )

        # Get the best runner and tactic from cache
        # If no valid tactic is found, the fallback runner and tactic will be used
        _, runner_id, tactic, _ = self.profiling_cache.search_cache(
            custom_op, runners, input_shapes, tuning_config)

        tuning_end_time = time.perf_counter()
        self.stats.tuned_op_time_cost[
            custom_op] = self.stats.tuned_op_time_cost.get(
                custom_op, 0) + tuning_end_time - tuning_start_time
        return (runners[runner_id], tactic)

    def _tuning_tensors(self, profile: OptimizationProfile,
                        inputs: List[torch.Tensor],
                        tuning_config: TuningConfig) -> List[torch.Tensor]:
        tensors = self._prepare_input_tensors(profile, inputs)
        if tuning_config.inputs_pre_hook is not None:
            tensors = tuning_config.inputs_pre_hook(tensors)
        return tensors

    def _one_pair_per_profile(self, runners: List[TunableRunner],
                              profiles: List[OptimizationProfile],
                              inputs: List[torch.Tensor],
                              tuning_config: TuningConfig,
                              is_pending: Callable[[OptimizationProfile],
                                                   bool], **kwargs) -> bool:
        """Whether the op profiles one (runner, tactic) per profile, judged on its first
        profile not yet tuned: the single-pair shortcut, where profiling a profile is the
        call that builds its kernel."""
        if tuning_config.distributed_tuning_strategy in (
                DistributedTuningStrategy.MERGE,
                DistributedTuningStrategy.PARALLEL):
            return False
        first = next((p for p in profiles if is_pending(p)), None)
        if first is None:
            return True
        tensors = self._tuning_tensors(first, inputs, tuning_config)
        return sum(
            len(r.get_valid_tactics(tensors, first, **kwargs))
            for r in runners) == 1

    def _build_across_ranks(self, custom_op: str, runners: List[TunableRunner],
                            profiles: List[OptimizationProfile],
                            inputs: List[torch.Tensor],
                            tuning_config: TuningConfig,
                            is_pending: Callable[[OptimizationProfile],
                                                 bool], **kwargs) -> None:
        """Build the kernels that profiling the pending profiles needs, split across the
        processes that share JIT builds.

        Each (profile, runner, tactic) runs once, on the process that claims it, so its
        kernels are built there and published for the others; one whose kernel another
        process is still building is skipped. Results are discarded and failures are left
        to the profiling that follows, which loads each kernel built here or elsewhere, or
        waits for one still being built.
        """
        pending = [p for p in profiles if is_pending(p)]
        items = []
        for i, p in enumerate(pending):
            tensors = self._tuning_tensors(p, inputs, tuning_config)
            for r, runner in enumerate(runners):
                items.extend((i, r, tactic)
                             for tactic in runner.get_valid_tactics(
                                 tensors, p, **kwargs))
        if not items:
            return
        sweep = jit_share.SharedSweep(
            repr((custom_op, [p.get_opt_shapes() for p in pending], items)))
        start = int(jit_share.start_fraction() * len(items))
        ran = busy = 0
        current, tensors, prepared = None, None, set()
        begin = time.perf_counter()
        with jit_share.defer_busy_builds():
            for k in list(range(start, len(items))) + list(range(start)):
                if not sweep.claim(k):
                    continue
                i, r, tactic = items[k]
                if i != current:
                    current, prepared = i, set()
                    tensors = self._tuning_tensors(pending[i], inputs,
                                                   tuning_config)
                runner = runners[r]
                try:
                    if r not in prepared and "do_preparation" in inspect.signature(
                            runner.forward).parameters:
                        runner(tensors,
                               tactic=-1,
                               do_preparation=True,
                               **kwargs)
                    prepared.add(r)
                    runner(tensors, tactic=tactic, **kwargs)
                    torch.cuda.current_stream().synchronize()
                    ran += 1
                except Exception as e:  # noqa: BLE001 - profiling reports failures
                    busy += jit_share.is_busy(e)
                finally:
                    sweep.done(k)
        self._debug_logger(
            f"[Autotuner] {custom_op}: shared JIT builds over {len(items)} (profile, "
            f"runner, tactic) items of {len(pending)} profiles: ran {ran} claimed here and "
            f"skipped {busy} whose kernel another process was building, in "
            f"{time.perf_counter() - begin:.1f}s")

    def _profile_sharing_jit_builds(
            self, custom_op: str, profiles: List[OptimizationProfile],
            profile_one: Callable[[OptimizationProfile], None],
            is_pending: Callable[[OptimizationProfile], bool]) -> None:
        """Run ``profile_one`` over ``profiles`` while the processes share DeepGEMM's builds.

        The processes split the not yet tuned profiles: each claims profiles from an
        offset of its own (claims in the shared directory) and runs them first. It sets
        aside a claimed profile whose kernel another process is building, so that it
        goes on to kernels nobody has claimed, and runs the set-aside profiles after
        the others. Last, it runs the profiles other processes claimed, each once that
        process has run it, when the profile's kernel is in DeepGEMM's shared cache.
        The set-aside and other processes' profiles wait for other processes at most
        one wait in total (:func:`jit_share.wait_seconds`); past it, this process runs
        each profile left and builds its kernel itself.
        """
        profiles = list(profiles)
        pending = [i for i, p in enumerate(profiles) if is_pending(p)]
        if not pending:
            for p in profiles:
                profile_one(p)
            return
        sweep = jit_share.SharedSweep(
            repr((custom_op, [profiles[i].get_opt_shapes() for i in pending])))
        start = int(jit_share.start_fraction() * len(pending))
        claimed, set_aside, others = [], [], []
        begin = time.perf_counter()
        with jit_share.defer_busy_builds():
            for i in pending[start:] + pending[:start]:
                if not sweep.claim(i):
                    others.append(i)
                    continue
                try:
                    profile_one(profiles[i])
                except Exception as e:
                    if not jit_share.is_busy(e):
                        sweep.done(i)
                        raise
                    set_aside.append(i)
                    continue
                claimed.append(i)
                sweep.done(i)
        aside_begin = time.perf_counter()
        late = 0
        with jit_share.wait_until(time.time() + jit_share.wait_seconds()):
            for i in set_aside:
                try:
                    profile_one(profiles[i])
                finally:
                    sweep.done(i)
            others_begin = time.perf_counter()
            for i in others:
                late += not sweep.wait_done(i)
                profile_one(profiles[i])
        pending_set = set(pending)
        for i, p in enumerate(profiles):
            if i not in pending_set:
                profile_one(p)
        self._debug_logger(
            f"[Autotuner] {custom_op}: shared JIT builds over {len(pending)} profiles: "
            f"ran {len(claimed) + len(set_aside)} claimed in {others_begin - begin:.1f}s "
            f"({len(set_aside)} set aside while another process built their kernel, "
            f"{others_begin - aside_begin:.1f}s), then {len(others)} claimed elsewhere in "
            f"{time.perf_counter() - others_begin:.1f}s ({late} not run there in time)"
        )

    def _prime_cached_tactics(
        self,
        custom_op: str,
        runners: List[TunableRunner],
        tuning_config: TuningConfig,
        inputs: List[torch.Tensor],
        **kwargs,
    ) -> None:
        """Compile cached winners in this process during tuning, not inference."""
        if os.environ.get("TLLM_AUTOTUNER_PRIME_CACHED_TACTICS", "1") != "1":
            return
        # Cache hits may differ across ranks; collective runners must stay
        # inside lockstep profiling, including profiling scratch allocation.
        if tuning_config.distributed_tuning_strategy == DistributedTuningStrategy.MERGE:
            return
        jit_classes = {
            type(r)
            for r in runners
            if isinstance(getattr(type(r), "kernel_cache", None), dict)
        }
        if not jit_classes:
            return
        for profile in self._optimization_profiles(tuning_config, inputs):
            hit, runner_id, tactic, _ = self.profiling_cache.search_cache(
                custom_op,
                runners,
                profile.get_opt_shapes(),
                tuning_config,
                apply_map_to_tuning_buckets=False,
            )
            if not hit:
                continue
            runner = runners[runner_id]
            if type(runner) not in jit_classes:
                continue
            key = (custom_op, type(runner).__name__, str(runner.unique_id()),
                   str(tactic))
            if key in self._primed_cached_tactics:
                continue
            tensors = self._prepare_input_tensors(profile, inputs)
            if tuning_config.inputs_pre_hook is not None:
                tensors = tuning_config.inputs_pre_hook(tensors)
            try:
                with nvtx_range(f"{custom_op} prime tactic {tactic}"):
                    if "do_preparation" in inspect.signature(
                            runner.forward).parameters:
                        runner(tensors,
                               tactic=-1,
                               do_preparation=True,
                               **kwargs)
                    runner(tensors, tactic=tactic, **kwargs)
                self._primed_cached_tactics.add(key)
            except Exception as e:
                try:
                    torch.cuda.synchronize()
                except RuntimeError:
                    pass
                logger.warning(
                    f"[Autotuner] Priming cached tactic failed for custom_op={custom_op}, "
                    f"runner={type(runner).__name__}, tactic={tactic}, "
                    f"shapes={profile.get_opt_shapes()}. Error: {e}")

    def _profile_runners(
        self,
        custom_op: str,
        runners: List[TunableRunner],
        input_tensors: List[torch.Tensor],
        profile: OptimizationProfile,
        tuning_config: TuningConfig,
        **kwargs,
    ) -> float:
        # PARALLEL ranks profile different tactics, so window requests here would diverge the
        # per-rank pools and let a later request register a window collectively on some ranks only.
        window_guard = (nccl_window_allocation_suppressed()
                        if tuning_config.distributed_tuning_strategy
                        == DistributedTuningStrategy.PARALLEL
                        and self._is_distributed() else
                        contextlib.nullcontext())
        with window_guard, nccl_window_tensor_scope(input_tensors) as scope:
            scope.escape(input_tensors)
            if tuning_config.inputs_pre_hook is not None:
                input_tensors = tuning_config.inputs_pre_hook(input_tensors)
            self._profiling_depth += 1
            try:
                return self._profile_runners_impl(custom_op, runners,
                                                  input_tensors, profile,
                                                  tuning_config, **kwargs)
            finally:
                self._profiling_depth -= 1

    def _profile_runners_impl(
        self,
        custom_op: str,
        runners: List[TunableRunner],
        input_tensors: List[torch.Tensor],
        profile: OptimizationProfile,
        tuning_config: TuningConfig,
        **kwargs,
    ) -> float:
        """Profile runners and select the best tactic.

        For multi-rank profiling, only rank 0 performs the actual profiling
        to avoid sync issues when different ranks select different tactics.
        The results are then broadcasted to all other ranks.
        """

        min_time = float('inf')
        has_tuning_failure_occurred = False
        best_runner_id, best_tactic = None, None
        # (runner, tactic) pairs this rank profiles: all of them, or under
        # PARALLEL its share of the deal, which can be empty.
        local_pairs = 0
        # Pre-walk the runners so we know the total candidate (runner, tactic)
        # pair count before deciding whether to short-circuit profiling. The
        # count is taken BEFORE distributed-strategy splitting; the PARALLEL
        # strategy can leave a rank with one tactic locally even though the
        # full set has several, and that case must still go through the
        # timed profile path so the merge picks a real winner.
        pairs_by_runner = []
        total_pairs = 0
        for runner_id, runner in enumerate(runners):
            # TODO: use FakeTensor here.
            runner_arg_names = {
                p.name
                for p in inspect.signature(runner.forward).parameters.values()
            }
            all_valid_tactics = runner.get_valid_tactics(
                input_tensors, profile, **kwargs)
            total_pairs += len(all_valid_tactics)
            pairs_by_runner.append(
                (runner_id, runner, runner_arg_names, all_valid_tactics))

        # MERGE/PARALLEL strategies rely on comparing per-rank timings to
        # pick the winning tactic across ranks. If a rank's local set of
        # tactics has only one entry we must still time it — the shortcut
        # would record min_time=0.0, which collides with peer ranks' 0.0
        # in the tie-break and silently drops slower tactics on the floor.
        if total_pairs == 1 and tuning_config.distributed_tuning_strategy not in (
                DistributedTuningStrategy.MERGE,
                DistributedTuningStrategy.PARALLEL):
            # Single-pair shortcut. There is nothing to compare against, so
            # the timed warmup + profile-repeat loop is pure overhead. We
            # fire one forward call on every rank to drive any JIT side
            # effects (e.g. DeepGEMM cubin compile for fp8_swap_ab_gemm),
            # then record the pair in the cache directly. `do_preparation`
            # still runs for runners that opt into it.
            runner_id, runner, runner_arg_names, valid_tactics = next(
                c for c in pairs_by_runner if len(c[3]) == 1)
            tac = valid_tactics[0]
            local_pairs = 1
            if "do_preparation" in runner_arg_names:
                runner(input_tensors, tactic=-1, do_preparation=True, **kwargs)
            try:
                with nvtx_range(f"r{runner_id}, tactic {tac} (single-pair)"):
                    self._run_profile_runner(runner, input_tensors, tac,
                                             **kwargs)
                best_runner_id, best_tactic = runner_id, tac
                # No comparable measurement exists; min_time stays 0.0 to
                # distinguish "recorded without profiling" from a real
                # timing.
                min_time = 0.0
            except Exception as e:
                if jit_share.is_busy(e):
                    raise  # Not a failure: the caller runs this profile later.
                shapes = self._get_input_sizes(input_tensors)
                logger.warning_once(
                    f"[Autotuner] Single-pair run failed for custom_op={custom_op}, runner={runner}, tactic={tac}, shapes={shapes}. Error: {e}",
                    key=(custom_op, "warning_autotuning_single_pair_failure"),
                )
                self.stats.failed_profiling_count[custom_op].add(
                    self.profiling_cache.get_cache_key(
                        custom_op,
                        runner,
                        profile.get_opt_shapes(),
                        tuning_config,
                        apply_map_to_tuning_buckets=False,
                        candidates=runners))
                has_tuning_failure_occurred = True
            # The shortcut's candidate is recorded like the timed path's, so a
            # run's log lists every pair either way: 0.000 is the documented
            # recorded-without-profiling marker, a failure shows as inf. Same
            # placement rationale as the timed path: after the handler, so a
            # formatting error is never recorded as a tactic failure.
            self._debug_logger(
                f"[Autotuner] Candidate: custom_op={custom_op}, "
                f"runner={runner}, tactic={tac}, "
                f"shapes={profile.get_opt_shapes()}, "
                f"time={(0.0 if best_runner_id is not None else float('inf')):.3f}ms"
            )
        else:
            # PARALLEL deals the (runner, tactic) pairs of all runners round
            # robin, so every rank of the group gets an even share even when
            # one runner has many tactics and the others few. The deal starts
            # at a rank taken from the op and the shape, the same on every
            # rank, so a runner's costly tactics do not fall on the same ranks
            # at every shape.
            pair_offset = self._parallel_deal_start(
                custom_op, profile, tuning_config.distributed_tuning_strategy)
            for runner_id, runner, runner_arg_names, all_valid_tactics in pairs_by_runner:
                valid_tactics = self._maybe_parallelize_tactics(
                    all_valid_tactics,
                    tuning_config.distributed_tuning_strategy,
                    offset=pair_offset)
                pair_offset += len(all_valid_tactics)
                local_pairs += len(valid_tactics)
                if "do_preparation" in runner_arg_names and len(
                        valid_tactics) > 0:
                    runner(
                        input_tensors,
                        tactic=-1,
                        do_preparation=True,
                        **kwargs,
                    )

                for tac in valid_tactics:
                    try:
                        with nvtx_range(f"r{runner_id}, tactic {tac}"):
                            time_measured = self._profile_single_kernel(
                                runner=runner,
                                inputs=input_tensors,
                                tactic=tac,
                                tuning_config=tuning_config,
                                use_cuda_graph=tuning_config.use_cuda_graph,
                                **kwargs,
                            )
                    except Exception as e:
                        # Synchronize to clear any pending CUDA errors left by the
                        # failed tactic.  Without this, the stale error propagates
                        # to subsequent CUDA/cuBLAS calls in the forward pass
                        # (observed as CUBLAS_STATUS_EXECUTION_FAILED on SM103).
                        try:
                            torch.cuda.synchronize()
                        except Exception:
                            pass
                        if jit_share.is_busy(e):
                            raise  # Not a failure: the caller runs this profile later.

                        # Handle None tensors for optional inputs
                        shapes = self._get_input_sizes(input_tensors)
                        logger.warning_once(
                            f"[Autotuner] Failed when profiling runner={runner}, tactic={tac}, shapes={shapes}. Error: {e}",
                            key=(custom_op,
                                 "warning_autotuning_profile_failure"),
                        )

                        # Record the failed profiling combinations
                        self.stats.failed_profiling_count[custom_op].add(
                            self.profiling_cache.get_cache_key(
                                custom_op,
                                runner,
                                profile.get_opt_shapes(),
                                tuning_config,
                                apply_map_to_tuning_buckets=False,
                                candidates=runners))

                        # Set time_measured to inf to notify the failure of the tactic. This can happen when `get_valid_tactics` mistakenly return wrong tactics
                        # or some runtime error occurs during profiling.
                        time_measured = float('inf')
                        has_tuning_failure_occurred = True
                    self._debug_logger(
                        f"[Autotuner] Candidate: custom_op={custom_op}, "
                        f"runner={runner}, tactic={tac}, "
                        f"shapes={profile.get_opt_shapes()}, "
                        f"time={time_measured:.3f}ms")
                    if time_measured < min_time:
                        min_time = time_measured
                        best_runner_id, best_tactic = runner_id, tac

        if best_runner_id is not None:
            # At least one valid (runner, tactic) pair is found
            cache_key = self.profiling_cache.get_cache_key(
                custom_op,
                runners[best_runner_id],
                profile.get_opt_shapes(),
                tuning_config,
                apply_map_to_tuning_buckets=False,
                candidates=runners)

            self._debug_logger(
                f"[Autotuner] Profiling runner={runners[best_runner_id]}, tactic={best_tactic} for cache_key={cache_key}."
            )
            self._debug_logger(
                f"[Autotuner] Selected: custom_op={custom_op}, runner={runners[best_runner_id]}, "
                f"tactic={best_tactic}, time={min_time:.3f}ms, fine_grained=OFF (disabled during tuning)"
            )
            # inspect call stack
            # TODO: use named tuple to make it more readable
            self.profiling_cache[cache_key] = (best_runner_id, best_tactic,
                                               min_time)
            self._keys_profiled_in_process.add(cache_key)

            self.stats.tuned_op_profiled_configs[custom_op] += 1
        elif local_pairs == 0 and total_pairs > 0:
            # The PARALLEL deal left this rank no pair to profile; the merge
            # brings in the winner the other ranks of the group find.
            self._debug_logger(
                f"[Autotuner] No (runner, tactic) pair dealt to this rank for custom_op={custom_op}, "
                f"input_shapes={profile.get_opt_shapes()}.")
        else:
            logger.warning_once(
                f"[Autotuner] No valid runner/tactic was found for custom_op={custom_op}, input_shapes={profile.get_opt_shapes()}. "
                f"At least one valid (runner, tactic) pair is required. "
                f"If get_valid_tactics is intended to return empty list, please ensure that this profile is not valid for the custom_op "
                f"and should not occurs during the inference stage, or fallback tactic is implemented. Otherwise, the the tuning process will crash.",
                key=(custom_op, "warning_autotuning_no_valid_tactic"),
            )

        return best_runner_id, best_tactic, min_time, has_tuning_failure_occurred

    def _parallel_jit_warmup(self, custom_op: str, runners: List[TunableRunner],
                             profiles: List[OptimizationProfile],
                             inputs: List[torch.Tensor],
                             tuning_config: TuningConfig, **kwargs) -> None:
        """Run each not-yet-tuned profile once on a thread pool to drive its JIT.

        See ``TuningConfig.parallel_jit_warmup``. Purely a warmup: results are
        discarded, and a failure here is left to the profiling loop, which
        runs every profile again and handles errors as usual.
        """
        num_threads = int(
            os.environ.get("TLLM_AUTOTUNER_PARALLEL_JIT_WARMUP",
                           tuning_config.parallel_jit_warmup))
        if num_threads < 0:
            num_threads = _default_parallel_jit_warmup_threads()
        if num_threads < 2:
            return
        pending = [
            p for p in profiles if not self.profiling_cache.search_cache(
                custom_op,
                runners,
                p.get_opt_shapes(),
                tuning_config,
                apply_map_to_tuning_buckets=False,
            )[0]
        ]
        num_threads = min(num_threads, len(pending))
        if num_threads < 2:
            return
        # Interleave contiguous blocks of the sweep, so the calls in flight
        # come from far-apart shapes and mostly select different kernels
        # rather than several threads compiling the same one.
        size = -(-len(pending) // num_threads)
        blocks = [pending[i * size:(i + 1) * size] for i in range(num_threads)]
        order = [b[j] for j in range(size) for b in blocks if j < len(b)]
        device = torch.cuda.current_device()
        # One stream for all workers: the allocator caches their blocks where
        # the profiling loop below allocates, and GPU time is not the cost here.
        stream = torch.cuda.current_stream()
        failures = []

        def warm(profile: OptimizationProfile) -> None:
            torch.cuda.set_device(device)
            try:
                with torch.cuda.stream(stream):
                    tensors = self._prepare_input_tensors(profile, inputs)
                    if tuning_config.inputs_pre_hook is not None:
                        tensors = tuning_config.inputs_pre_hook(tensors)
                    for runner in runners:
                        for tactic in runner.get_valid_tactics(
                                tensors, profile, **kwargs):
                            runner(tensors, tactic=tactic, **kwargs)
                    stream.synchronize()
            except Exception as e:  # noqa: BLE001 - the profiling loop reports it
                failures.append(e)

        start = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(
                max_workers=num_threads,
                thread_name_prefix="autotuner_jit_warmup") as pool:
            list(pool.map(warm, order))
        logger.info(f"[Autotuner] {custom_op}: ran {len(order)} profiles on "
                    f"{num_threads} threads to drive their JIT in "
                    f"{time.perf_counter() - start:.1f}s" +
                    (f"; {len(failures)} failed and are left to profiling "
                     f"({type(failures[0]).__name__}: {failures[0]})"
                     if failures else ""))

    @staticmethod
    def _run_profile_runner(runner: TunableRunner, inputs: List[torch.Tensor],
                            tactic: Any, **kwargs) -> None:
        with nccl_window_tensor_scope(inputs) as scope:
            try:
                runner(inputs, tactic=tactic, **kwargs)
            finally:
                # Profiling inputs outlive each candidate run; everything newly
                # allocated by the runner is an internal, discarded result.
                scope.escape(inputs)

    def _get_input_sizes(self, inputs: List[torch.Tensor]) -> List[torch.Size]:

        # Handle None tensors for optional inputs and non-Tensor scalar values
        sizes = [
            input.size() if isinstance(input, torch.Tensor) else torch.Size(
                (0, )) for input in inputs
        ]

        return sizes

    @contextlib.contextmanager
    def _shared_profile_graph_pool(self):
        """Let the profiling captures in this scope share one CUDA graph memory pool.

        Every profiling capture is reset before the next one starts. With a pool
        per capture, each reset drops that pool and the next capture maps its
        memory again, which under expandable segments costs milliseconds per
        capture. A shared pool keeps the freed blocks cached for the next
        capture. The pool is dropped when the outermost scope exits, so its
        memory is not held across forward passes.
        """
        self._profile_graph_pool_depth += 1
        try:
            yield
        finally:
            self._profile_graph_pool_depth -= 1
            if self._profile_graph_pool_depth == 0:
                self._profile_graph_pool = None

    def _profile_graph_pool_handle(self) -> Tuple[int, int]:
        """Pool handle for one profiling capture: the shared pool inside
        _shared_profile_graph_pool, otherwise a new pool."""
        if self._profile_graph_pool_depth == 0:
            return torch.cuda.graph_pool_handle()
        if self._profile_graph_pool is None:
            self._profile_graph_pool = torch.cuda.MemPool()
        return self._profile_graph_pool.id

    def _profile_single_kernel(
        self,
        runner: TunableRunner,
        inputs: List[torch.Tensor],
        tactic: Any,
        tuning_config: TuningConfig,
        use_cuda_graph: bool = False,
        **kwargs,
    ) -> float:
        """Profile a single kernel implementation for performance measurement.

        Args:
            runner (TunableRunner): The runner implementation to profile
            inputs (List[torch.Tensor]): Input tensors for the kernel
            tactic (Any): Tactic to use for this profiling run

        Returns:
            Average execution time in milliseconds

        Note:
            The method performs warmup runs, then measures multiple iterations
            to get an average execution time. Stream synchronization and delays
            are used to ensure accurate timing.
        """
        input_tensor_batches = self._prepare_input_tensors_with_batches(
            inputs, tuning_config)

        if tuning_config.cold_l2_flush_bytes > 0:
            avg_time = self._profile_cold_l2(runner, input_tensor_batches,
                                             tactic, tuning_config, **kwargs)
            self._debug_logger(
                f"[Autotuner] Profiled (cold L2) runner={runner}, tactic={tactic}, "
                f"shapes={self._get_input_sizes(inputs)}: {avg_time:.6f}ms.")
            return avg_time

        stream = torch.cuda.current_stream()
        # If the warm up time is longer than 0.5ms, we will profile the kernel with fewer repeats.
        profile_fewer_repeat = 2
        short_profile_threshold_ms = 1

        avg_time = float('inf')

        def pure_profile(stream: torch.cuda.Stream, repeat: int):
            graph = torch.cuda.CUDAGraph()

            @contextlib.contextmanager
            def graph_pool_lifetime():
                if not use_cuda_graph:
                    yield None
                    return
                graph_pool = self._profile_graph_pool_handle()
                completed = False
                try:
                    yield graph_pool
                    completed = True
                finally:
                    graph.reset()
                    # The NCCL-window owner stays per capture even when the
                    # memory pool is shared: a later capture of the same pool
                    # gets a new owner.
                    if completed:
                        release_nccl_window_graph_owner(graph_pool)
                    else:
                        # Local profiling failures need not occur on every rank.
                        abandon_nccl_window_graph_owner(graph_pool)
                        # Do not capture into a pool a failed capture used.
                        self._profile_graph_pool = None

            if self._use_global_timer:
                start_ts = torch.empty(1, dtype=torch.int64, device='cuda')
                end_ts = torch.empty(1, dtype=torch.int64, device='cuda')

                def record_start():
                    record_global_timer(start_ts.data_ptr(), stream)

                def record_end():
                    record_global_timer(end_ts.data_ptr(), stream)

                def elapsed_time():
                    # GPU %globaltimer counts in ns; convert to ms to match the
                    # units of Torch.cuda.Event.elapsed_time()
                    return (end_ts.item() - start_ts.item()) / 1e6
            else:
                start_evt = torch.cuda.Event(enable_timing=True)
                end_evt = torch.cuda.Event(enable_timing=True)

                def record_start():
                    start_evt.record()

                def record_end():
                    end_evt.record()

                def elapsed_time():
                    return start_evt.elapsed_time(end_evt)

            with graph_pool_lifetime() as graph_pool, torch.cuda.stream(stream):
                if use_cuda_graph:
                    with nccl_window_graph_capture(graph, graph_pool):
                        for r in range(repeat):
                            self._run_profile_runner(
                                runner,
                                input_tensor_batches[r %
                                                     len(input_tensor_batches)],
                                tactic,
                                **kwargs,
                            )

                stream.synchronize()
                if tuning_config.distributed_tuning_strategy == DistributedTuningStrategy.MERGE:
                    # Currently only AllReduce will use this strategy, and only MPI parallel will enable tuning.
                    self._dist.tp_barrier()

                # Delay the profiled kernel launch to eliminate affects of host time overhead in profiling.
                if use_cuda_graph:
                    delay_kernel(self._CUDA_GRAPH_DELAY_MICRO_SECS, stream)
                else:
                    delay_kernel(self.stream_delay_micro_secs, stream)

                record_start()

                if use_cuda_graph:
                    graph.replay()
                else:
                    for r in range(repeat):
                        self._run_profile_runner(
                            runner,
                            input_tensor_batches[r % len(input_tensor_batches)],
                            tactic,
                            **kwargs,
                        )

                record_end()
                stream.synchronize()

                return elapsed_time() / repeat

        # warm up, no timing
        for _ in range(self.warmup):
            self._run_profile_runner(runner, input_tensor_batches[-1], tactic,
                                     **kwargs)

        fewer_repeat_avg_time = pure_profile(stream, profile_fewer_repeat)

        disable_short_profile = os.environ.get(
            "TLLM_AUTOTUNER_DISABLE_SHORT_PROFILE", "0") == "1"

        # Disable this feature for merged tuning strategy to avoid potential hang due to asymmetric tuning.
        if fewer_repeat_avg_time > short_profile_threshold_ms and not disable_short_profile \
            and tuning_config.distributed_tuning_strategy != DistributedTuningStrategy.MERGE:
            # directly use the few repeat estimated time to avoid redundant profiling
            avg_time = fewer_repeat_avg_time
        else:
            # profile the kernel with the full repeat to get precise time
            avg_time = pure_profile(stream, self.repeat)

        shapes = self._get_input_sizes(inputs)
        self._debug_logger(
            f"[Autotuner] Profiled runner={runner}, tactic={tactic}, shapes={shapes}: {avg_time:.6f}ms."
        )

        return avg_time

    # Flush buffers of cold-L2 profiling, by (device, bytes); released when the ``autotune()`` that
    # turned tuning on exits.
    L2_FLUSH_BUFFERS: Dict[Tuple[int, int], torch.Tensor] = {}

    def _profile_cold_l2(self, runner: TunableRunner,
                         input_tensor_batches: List[List[torch.Tensor]],
                         tactic: Any, tuning_config: TuningConfig,
                         **kwargs) -> float:
        """Median per-repeat time (ms) of ``runner`` with L2 flushed before every repeat.

        One CUDA graph holds ``repeat`` x (read the flush buffer, stamp, runner, stamp); a repeat's
        time is its stamp difference, so neither the flush nor host launches are timed.
        """
        key = (torch.cuda.current_device(), tuning_config.cold_l2_flush_bytes)
        flush = self.L2_FLUSH_BUFFERS.get(key)
        if flush is None:
            flush = torch.ones(tuning_config.cold_l2_flush_bytes // 2,
                               dtype=torch.bfloat16,
                               device="cuda")
            self.L2_FLUSH_BUFFERS[key] = flush
        for _ in range(self.warmup):
            runner(input_tensor_batches[-1], tactic=tactic, **kwargs)
        repeat = max(self.repeat, 3)
        stamps = torch.zeros(2 * repeat, dtype=torch.int64, device="cuda")
        base = stamps.data_ptr()
        graph = torch.cuda.CUDAGraph()
        graph_pool = torch.cuda.graph_pool_handle()
        torch.cuda.synchronize()
        completed = False
        try:
            # Like every capture here: NCCL-window ownership, and no automatic GC until it ends.
            with nccl_window_graph_capture(graph, graph_pool):
                capture_stream = torch.cuda.current_stream()
                for r in range(repeat):
                    torch.amax(flush)
                    record_global_timer(base + 16 * r, capture_stream)
                    runner(input_tensor_batches[r % len(input_tensor_batches)],
                           tactic=tactic,
                           **kwargs)
                    record_global_timer(base + 16 * r + 8, capture_stream)
            for _ in range(2):
                graph.replay()
            torch.cuda.synchronize()
            completed = True
        finally:
            graph.reset()
            if completed:
                release_nccl_window_graph_owner(graph_pool)
            else:
                abandon_nccl_window_graph_owner(graph_pool)
        times = (stamps[1::2] - stamps[0::2]).double() / 1e6
        return float(times.median().item())

    def _optimization_profiles(
            self, tuning_config: TuningConfig,
            inputs: List[torch.Tensor]) -> List[OptimizationProfile]:
        """Generate optimization profiles for autotuning.

        Args:
            tuning_config (TuningConfig): Tuning configuration
            inputs (List[torch.Tensor]): List of input tensors

        Returns:
            List of OptimizationProfile objects representing different configurations

        Note:
            This method performs a cartesian product of all possible dimension
            combinations specified in dynamic_tensor_specs.
        """
        # every dimension created from the concrete input tensor shape
        # generate some dynamic dimension description based on the dynamic_tensors

        # Zero handles the case where a TRTLLM op has optional or scalar inputs.
        base_profile = OptimizationProfile(
            [[StaticDim(x) for x in t.size()]
             if isinstance(t, torch.Tensor) else [StaticDim(0)]
             for t in inputs])

        generated_profiles: List[OptimizationProfile] = []

        dynamic_dims = []

        for spec in tuning_config.dynamic_tensor_specs:
            assert callable(spec.gen_tuning_buckets) or isinstance(spec.gen_tuning_buckets, (list, tuple)), \
                "The given dynamic dimension must provide a opt value generation function or a list of opt values"
            if not _spec_in_bounds(spec, base_profile.shapes):
                continue
            if self.skip_dynamic_tuning_buckets:
                if spec.map_to_tuning_buckets is not None:
                    # Still include the bucketed value of the actual shape so the
                    # cache key used during profiling (raw) aligns with the key
                    # used during inference (bucketed via map_to_tuning_buckets).
                    actual_val = base_profile.shapes[spec.input_idx][
                        spec.dim_idx].val
                    mapped_val = spec.map_to_tuning_buckets(actual_val)
                    if mapped_val < 0:
                        # Lookup cannot use a measured winner for this input.
                        # Leave it as a cache miss so choose_one uses fallback.
                        return []
                    opt_shapes = (mapped_val, )
                else:
                    opt_shapes = ()
            elif callable(spec.gen_tuning_buckets):
                if tuning_config.tune_max_num_tokens is None:
                    # Use the current input size as the opt value
                    opt_shapes = spec.gen_tuning_buckets(
                        base_profile.shapes[spec.input_idx][spec.dim_idx].val)
                else:
                    # Use the tune_max_num_tokens as the opt value
                    opt_shapes = spec.gen_tuning_buckets(
                        tuning_config.tune_max_num_tokens)
            else:
                # Default values is an empty tuple, means that user does not want to tune this dimension.
                opt_shapes = spec.gen_tuning_buckets
            # Add the current input value as one of the opt values
            opt_shapes = set(opt_shapes)
            if tuning_config.tune_max_num_tokens is not None:
                opt_shapes.add(
                    min(
                        tuning_config.tune_max_num_tokens,
                        base_profile.shapes[spec.input_idx][spec.dim_idx].val,
                    ))
            else:
                opt_shapes.add(
                    base_profile.shapes[spec.input_idx][spec.dim_idx].val)
            opt_shapes = sorted(list(opt_shapes))
            opt_shapes_max = tuple(opt_shapes[1:]) + (float('inf'), )
            opt_shapes_max = {
                v1: v2
                for v1, v2 in zip(opt_shapes, opt_shapes_max)
            }
            dynamic_dims.append(
                (spec.input_idx, spec.dim_idx, opt_shapes_max, opt_shapes))

        # grid search, do cartesian product for all the dynamic axis
        dim_grids = itertools.product(*[d[-1] for d in dynamic_dims])
        for opt_point in dim_grids:
            p = copy.deepcopy(base_profile)
            for pos, (input_idx, dim_idx, opt_shapes_max,
                      opt_shapes) in enumerate(dynamic_dims):
                opt_value = opt_point[pos]
                #TODO: fix me, how to set the min and max?
                min_value = opt_value
                max_value = opt_shapes_max[opt_value]
                p.shapes[input_idx][dim_idx] = DynamicDim(
                    min_value, opt_value, max_value)

            # Adjust the profile to satisfy the constraints
            for spec in tuning_config.constraint_specs:
                if not _spec_in_bounds(spec, p.shapes):
                    continue
                if p.shapes[spec.input_idx] == [StaticDim(0)]:
                    continue
                min_value = opt_value = max_value = spec.infer_shape(
                    p.get_opt_shapes())
                p.shapes[spec.input_idx][spec.dim_idx] = DynamicDim(
                    min_value, opt_value, max_value)
            generated_profiles.append(p)
            self._debug_logger(f"[Autotuner] Generated profile: {p}")
        return generated_profiles

    @classmethod
    @lru_cache(maxsize=None)
    def _find_nearest_profile(
        cls,
        shapes: Tuple[torch.Size],
        dynamic_tensor_specs: Tuple[DynamicTensorSpec, ...],
        constraint_specs: Tuple[ConstraintSpec, ...],
        tune_max_num_tokens: int = None,
        apply_map_to_tuning_buckets: bool = True,
    ) -> Tuple:
        """Find the nearest optimization profile for given inputs
        User can define their own nearest profile generation method to reduce the host overhead.

        Args:
            shapes: Tuple of input tensor shapes
            tuning_config: Tuning configuration
            apply_map_to_tuning_buckets: If True, apply map_to_tuning_buckets for runtime cache lookups.
                If False, use raw bucket values for tuning cache storage.

        Return:
            Tuple: A tuple containing:
                - attributes: Tuple of runner attributes, sorted.
                - profile: Tuple of input tensor shapes
        """
        base_profile = list(list(shape) for shape in shapes)

        for spec in dynamic_tensor_specs:
            # Bounds check: skip specs that reference inputs or dimensions not present in the
            # current shapes tuple. This can happen on hardware (e.g. SM121 / DGX Spark) where
            # ops produce fewer or differently-shaped tensors than the specs were authored for.
            if not _spec_in_bounds(spec, base_profile):
                continue

            # During runtime: apply map_to_tuning_buckets to map input to bucket
            # During tuning: no mapper, use raw bucket value
            if apply_map_to_tuning_buckets:
                base_profile[spec.input_idx][
                    spec.dim_idx] = spec.map_to_tuning_buckets(
                        base_profile[spec.input_idx][spec.dim_idx])

            if tune_max_num_tokens is not None:
                base_profile[spec.input_idx][spec.dim_idx] = min(
                    base_profile[spec.input_idx][spec.dim_idx],
                    tune_max_num_tokens)

        # associated dimensions dependent on other free dynamic dimensions, so assign -1 in the profile
        for spec in constraint_specs:
            # Bounds check: same defensive guard as above for constraint specs.
            if not _spec_in_bounds(spec, base_profile):
                continue
            if base_profile[spec.input_idx] == [0]:
                continue
            base_profile[spec.input_idx][spec.dim_idx] = -1

        return tuple(tuple(shape) for shape in base_profile)

    def _create_tensor_like(self, origin_tensor: torch.Tensor,
                            dims: List[Dim]) -> torch.Tensor:
        """Create a new tensor matching the properties of the original tensor.

        Args:
            origin_tensor (torch.Tensor): Template tensor to match
            dims (List[Dim]): List of dimensions for the new tensor

        Returns:
            New tensor with specified dimensions and matching properties

        Note:
            Creates a zero tensor with the same dtype and device as the original,
            but with dimensions specified by the dims parameter.
        """
        dtype = origin_tensor.dtype
        device = origin_tensor.device
        shapes = []
        for i, d in enumerate(dims):
            if isinstance(d, StaticDim):
                assert d.val == origin_tensor.shape[i]
                shapes.append(d.val)
            else:
                # TODO: how to make sure the created Tensor has the min/max info
                assert isinstance(d, DynamicDim)
                shapes.append(d.opt)

        if len(dims) == 2 and isinstance(dims[0], DynamicDim) and isinstance(
                dims[1], StaticDim) and (dtype == torch.int32
                                         or dtype == torch.int64):
            # We should be carefully about int values, since they might be index like topk_index.
            # We want to keep them legal, so just repeating input tensor.
            repeat_times = (shapes[0] + origin_tensor.shape[0] -
                            1) // origin_tensor.shape[0]
            dup_tensor = origin_tensor.repeat(repeat_times, 1)[:shapes[0]]
            return dup_tensor

        # TODO: FIXME, sometimes the content of the tensor can affect the performance, like MOE
        # One solution is to manituplate the tensor content to make it more like the real data
        # during the tuning process. This can by controlled in the preparation phase by the runner.
        # It must not use all zero tensors. Otherwise the timing results become unreliable.
        if dtype == torch.float4_e2m1fn_x2:
            return (torch.rand(shapes, device=device) * 10 - 5).to(
                torch.uint8).view(dtype)
        return (torch.rand(shapes, device=device) * 10 - 5).to(dtype)

    def _prepare_input_tensors(
            self, profile: OptimizationProfile,
            inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        tensors = []
        for i, p in enumerate(profile.shapes):
            if any(isinstance(d, DynamicDim) for d in p):
                tensor = self._create_tensor_like(inputs[i], p)
            else:
                tensor = inputs[i]
            tensors.append(tensor)
        return tensors

    def _prepare_input_tensors_with_batches(
        self,
        inputs: List[torch.Tensor],
        tuning_config: TuningConfig,
    ) -> List[List[torch.Tensor]]:
        if not tuning_config.use_cold_l2_cache:
            return [inputs]

        one_buffer_bytes = sum(
            input.numel() *
            input.element_size() if isinstance(input, torch.Tensor) else 0
            for input in inputs)
        if one_buffer_bytes <= 0:
            self._debug_logger(
                "[Autotuner] No tensor inputs or zero-sized tensors; falling back to single-batch profiling."
            )
            return [inputs]

        num_buffers = self._get_l2_cache_size_in_bytes(
        ) * 3 // one_buffer_bytes + 1
        num_buffers = min(num_buffers, self.repeat + 1)

        inputs_list = [inputs]
        for _ in range(num_buffers - 1):
            inputs_list.append(
                list(t.clone() if isinstance(t, torch.Tensor) else t
                     for t in inputs))

        self._debug_logger(
            f"[Autotuner] use_cold_l2_cache={tuning_config.use_cold_l2_cache}, use {num_buffers} different tensors for profiling"
        )
        return inputs_list

    def clear_cache(self) -> None:
        """Clear the profiling cache."""
        self.profiling_cache.clear()
        self._keys_profiled_in_process.clear()
        if hasattr(torch.ops.trtllm, "clear_allreduce_tactic_cache"):
            torch.ops.trtllm.clear_allreduce_tactic_cache()

    def reset_statistics(self) -> None:
        """Reset all statistics counters."""
        self.stats = AutoTunerStatistics()

    def print_profiling_cache(self):
        self._debug_logger(f"[Autotuner] The profiling_cache entries:")
        self._debug_logger(
            f"[Autotuner] Cache contents: (custom_op, runner, unique_id, shape_profiles, code_fingerprint[, tactic_space]) -> (runner_id, tactic, min_time)"
        )
        for key, value in self.profiling_cache.cache.items():
            runner_id, tactic, min_time = value
            self._debug_logger(
                f"[Autotuner] {key}: (runner_id={runner_id}, tactic={tactic}, min_time={min_time})"
            )

        self.print_statistics()

    def print_statistics(self):
        self._debug_logger(f"[Autotuner] The statistics:")
        for line in self.stats.__str__().split("\n"):
            self._debug_logger(line)

    @contextlib.contextmanager
    def capture(self):
        """Context manager for capturing execution contexts for testing.

        Returns a TacticsCapture object that can be iterated to get all valid
        (runner, tactic) combinations.

        Example:
            >>> # Single context case
            >>> with AutoTuner.get().capture() as tactics_capture:
            ...     y = custom_op.forward(x)
            >>>
            >>> for runner, tactic in tactics_capture:
            ...     with AutoTuner.get().replay(runner, tactic):
            ...         y = custom_op.forward(x)

            >>> # Multiple contexts case
            >>> with AutoTuner.get().capture() as tactics_capture:
            ...     y = custom_op1.forward(x)
            ...     z = custom_op2.forward(y)
            >>>
            >>> for config in tactics_capture:
            ...     with AutoTuner.get().replay(config):
            ...         y = custom_op1.forward(x)
            ...         z = custom_op2.forward(y)
        """
        tactics_capture = self.TacticsCapture(self)
        self._active_capture = tactics_capture
        try:
            yield tactics_capture
        finally:
            self._active_capture = None
            self._last_capture = tactics_capture

    @contextlib.contextmanager
    def replay(self, *config: Tuple[Tuple[TunableRunner, int], ...]):
        """Context manager for replaying with specific runner/tactic configuration.

        Args:
            config:
                - A tuple of (runner, tactic) pairs. The tuple size matches the number of captured choose_one() contexts.
        """
        # Parse config argument
        if len(config) == 1:
            if isinstance(config[0], tuple):
                # Multiple contexts: replay(((r0,t0), (r1,t1), ...))
                runner_tactic_pairs = list(config[0])
            else:
                # Also handle single context passed as replay((runner, tactic))
                runner_tactic_pairs = [config[0]]
        else:
            raise ValueError(
                f"Invalid config for replay: {config}\n"
                "Expected replay(((runner, tactic), (runner, tactic), ...))")

        # Find the TacticsCapture to use
        tactics_capture = self._active_capture or self._last_capture

        if tactics_capture is None:
            raise RuntimeError(
                "No TacticsCapture available for replay. "
                "Make sure you've called capture() before replay().")

        # Temporarily set as active capture during replay
        prev_active = self._active_capture
        self._active_capture = tactics_capture

        runner_tactic_list = []
        for ctx_idx, (runner, tactic) in enumerate(runner_tactic_pairs):
            runners = tactics_capture._captured_contexts[ctx_idx]['runners']
            runner_idx = runners.index(runner)
            runner_tactic_list.append((runner_idx, tactic))

        self._debug_logger(
            f"[Autotuner][replay]: Testing configuration: {runner_tactic_list}")

        # Replay the contexts with given (runner, tactic) pairs
        tactics_capture._replay_runner_tactic_list = runner_tactic_list
        tactics_capture._replay_context_idx = 0

        try:
            yield
        finally:
            tactics_capture._replay_runner_tactic_list = None
            tactics_capture._replay_context_idx = 0
            # Restore previous active capture state
            self._active_capture = prev_active

    def _get_l2_cache_size_in_bytes(self, device_id: int = 0) -> int:
        device = self._checkCudaErrors(driver.cuDeviceGet(device_id))
        return self._checkCudaErrors(
            driver.cuDeviceGetAttribute(
                driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE,
                device,
            ))

    def _checkCudaErrors(self, result) -> Any:
        status = result[0]
        if status != driver.CUresult.CUDA_SUCCESS:
            code = getattr(status, "value", status)
            raise RuntimeError(
                f"CUDA error code={code}({self._cudaGetErrorEnum(status)})")
        # CUDA APIs always return the status as the first element of the result tuple
        if len(result) == 1:
            return None
        elif len(result) == 2:
            return result[1]
        else:
            return result[1:]

    def _cudaGetErrorEnum(self, error) -> str:
        from cuda.bindings import nvrtc
        if isinstance(error, driver.CUresult):
            err, name = driver.cuGetErrorName(error)
            return name if err == driver.CUresult.CUDA_SUCCESS else "<unknown>"
        elif isinstance(error, nvrtc.nvrtcResult):
            return nvrtc.nvrtcGetErrorString(error)[1]
        else:
            raise RuntimeError("Unknown error type: {}".format(error))

    def setup_distributed_state(self,
                                mapping: Mapping,
                                dist: Optional[Distributed] = ...):
        """Setup distributed communication state for autotuning."""
        self.mapping = mapping
        # Create dist only when dist is not provided.
        # Use the provided dist even if it is None. This is useful for testing.
        self._dist = Distributed.get(mapping) if dist is ... else dist
        self._debug_logger(
            f"[AutoTuner] Whether using distributed tuning: {self._is_distributed()}"
        )

    def _tuning_group(self) -> Tuple[int, int]:
        """Return ``(size, index)`` of this rank's tuning group.

        The group is the TP group. Helix CP ranks run the same SPMD forward
        on the same token batch, so Helix adds the CP ranks too. Otherwise a
        Helix generation worker with ``tp_size == 1`` would profile every
        tactic on every CP rank. The index matches the TP-major, CP-minor
        order of ``Distributed.tp_cp_allgather``, which merges the results.
        """
        if self.mapping is None:
            return 1, 0
        size = self.mapping.tp_size
        index = self.mapping.tp_rank
        if self.mapping.has_cp_helix():
            size *= self.mapping.cp_size
            index = index * self.mapping.cp_size + self.mapping.cp_rank
        return size, index

    def _is_distributed(self) -> bool:
        """Check if we're in a distributed environment."""
        return self._dist is not None and self._tuning_group()[0] > 1

    def _needs_rank_agreement(self,
                              strategy: DistributedTuningStrategy) -> bool:
        """Whether ranks must agree on cache hits under ``strategy``.

        Never for a choose_one() reached from a runner this rank is profiling:
        under BROADCAST rank 0 profiles alone, so a collective there never
        completes.
        """
        return (strategy != DistributedTuningStrategy.INDEPENDENT
                and self._is_distributed() and not self._profiling_depth)

    def _agree_on_cache_hit(self, runners: List[TunableRunner],
                            is_cache_hit: bool) -> bool:
        """Whether every rank of the tuning group has the entry (one host
        all-gather of each rank's hit and code fingerprint for ``runners``).

        Ranks that fingerprint the runners differently may be running
        different code, with different valid tactics: their keys would not
        meet in the merge or broadcast, and tuning them together could pick a
        tactic some rank cannot run. So may ranks whose sources changed after
        they started (an `edited-` fingerprint), even with equal fingerprints:
        the disk no longer tells what each rank loaded. Every rank sees the
        same gathered values, so they all raise here, before the collective
        profiling.
        """
        classes = tuple(type(r) for r in runners)
        gathered = self._dist.tp_cp_allgather(
            obj=(bool(is_cache_hit), _candidates_fingerprint(classes)))
        fingerprints = {fingerprint for _, fingerprint in gathered}
        names = sorted({".".join(_class_name(k)) for k in classes})
        if any(
                isinstance(f, str) and f.startswith(_EDITED_PREFIX)
                for f in fingerprints):
            raise RuntimeError(
                f"[Autotuner] TensorRT-LLM sources used by {names} changed "
                "after the ranks of a tuning group started, so their loaded "
                "code cannot be compared (see the warnings). Restart the "
                "workers, or set TLLM_AUTOTUNER_CACHE_KEY=none to key the "
                "cache without code fingerprints.")
        if len(fingerprints) > 1:
            raise RuntimeError(
                f"[Autotuner] The ranks of a tuning group fingerprint the code "
                f"of {names} differently ({sorted(fingerprints, key=str)}): "
                "they may be running different TensorRT-LLM sources (a file "
                "edited while they started, or one a node cannot read; see the "
                "warnings). Run every rank on the same files, or set "
                "TLLM_AUTOTUNER_CACHE_KEY=none to key the cache without code "
                "fingerprints.")
        return all(hit for hit, _ in gathered)

    def _profile_hits_on_all_ranks(self, profiles: List[OptimizationProfile],
                                   hits: List[bool]) -> List[bool]:
        """Per profile, whether every rank that builds it has it cached (one host all-gather).

        Ranks build their profiles from their own input shapes (attention DP), so the lists may differ.
        """
        keys = [p.get_opt_shapes() for p in profiles]
        missed = set()
        for rank_keys, rank_hits in self._dist.tp_cp_allgather(
                obj=(keys, [bool(hit) for hit in hits])):
            missed.update(key for key, hit in zip(rank_keys, rank_hits)
                          if not hit)
        return [key not in missed for key in keys]

    def _parallel_deal_start(self, custom_op: str, profile: OptimizationProfile,
                             strategy: DistributedTuningStrategy) -> int:
        """Where a PARALLEL deal of ``profile``'s (runner, tactic) pairs starts.

        Derived from the op and the shapes with a stable hash (not ``hash``,
        which differs between processes), so every rank deals the same way.
        """
        if strategy != DistributedTuningStrategy.PARALLEL or not self._is_distributed(
        ):
            return 0
        key = repr((custom_op, profile.get_opt_shapes())).encode()
        return zlib.crc32(key) % self._tuning_group()[0]

    def _maybe_parallelize_tactics(self,
                                   all_valid_tactics: List[Any],
                                   strategy: DistributedTuningStrategy,
                                   offset: int = 0) -> List[Any]:
        """Split tactics across the tuning group if strategy is PARALLEL.

        ``offset`` is where this runner's pairs start in the profile's deal:
        the deal's start plus the pairs of the runners before this one.
        """
        if strategy == DistributedTuningStrategy.PARALLEL and self._is_distributed(
        ):
            # Each rank of the tuning group only tunes its share of tactics;
            # _merge_cache_data then keeps the fastest across the group.
            group_size, group_index = self._tuning_group()
            return [
                tactic for idx, tactic in enumerate(all_valid_tactics)
                if (offset + idx) % group_size == group_index
            ]
        else:
            return all_valid_tactics

    def _maybe_sync_cache_data(self, strategy: DistributedTuningStrategy,
                               custom_op: str):
        """Synchronize cache data across all ranks."""
        if not self._is_distributed():
            return

        if strategy == DistributedTuningStrategy.BROADCAST:
            self._broadcast_cache_data(custom_op)
        elif strategy == DistributedTuningStrategy.INDEPENDENT:
            return
        elif strategy == DistributedTuningStrategy.MERGE:
            self._merge_cache_data(custom_op)
        elif strategy == DistributedTuningStrategy.PARALLEL:
            self._merge_cache_data(custom_op)
        else:
            logger.error(
                f"[AutoTuner] Unknown distributed tuning strategy: {strategy}, falling back to independent"
            )
            return

    def _merge_cache_data(self, custom_op: str):
        cache_data = self.profiling_cache.shareable(
            self.profiling_cache.get_specific_custom_op(custom_op))
        self.profiling_cache.merge_cache_data(
            _fastest_entries(self._dist.tp_cp_allgather(obj=cache_data)))

    def post_tune_merge_tactics(self,
                                custom_ops: Optional[Set[str]] = None,
                                majority: bool = False) -> int:
        """After tuning, keep per cache key (of ``custom_ops`` when given) the ranks' fastest tactic,
        or with ``majority`` the one most ranks chose. Collective — all ranks must call it.
        Returns how many of this rank's tactics changed."""
        if not self._is_distributed():
            return 0
        mine = self.profiling_cache.shareable({
            key: value
            for key, value in self.profiling_cache.cache.items()
            if custom_ops is None or key[0] in custom_ops
        })
        per_rank = self._dist.tp_cp_allgather(obj=mine)
        merged = (_majority_entries(per_rank)
                  if majority else _fastest_entries(per_rank))
        changed = sum(1 for key, value in merged.items()
                      if key not in mine or mine[key][1] != value[1])
        self.profiling_cache.merge_cache_data(merged)
        return changed

    def _broadcast_cache_data(
        self,
        custom_op: str,
    ) -> None:
        """Broadcast tactics from root rank to all other ranks."""
        cache_data = self.profiling_cache.shareable(
            self.profiling_cache.get_specific_custom_op(custom_op))
        root = 0
        cache_data = self._dist.tp_cp_broadcast(obj=cache_data, root=root)

        self.profiling_cache.merge_cache_data(cache_data)

    def _should_current_rank_tune(self,
                                  strategy: DistributedTuningStrategy) -> bool:
        """Determine if this rank should perform tuning based on strategy."""
        if not self._is_distributed():
            return True

        if strategy == DistributedTuningStrategy.BROADCAST:
            # Only the tuning group's root (the broadcast root) tunes
            return self._tuning_group()[1] == 0
        elif strategy in {
                DistributedTuningStrategy.INDEPENDENT,
                DistributedTuningStrategy.MERGE,
                DistributedTuningStrategy.PARALLEL,
        }:
            # All ranks tune independently
            return True
        else:
            logger.error(
                f"[AutoTuner] Unknown distributed tuning strategy: {strategy}, falling back to independent"
            )
            return True

    def cache_pp_recv(self):
        if self.mapping.has_pp() and not self.mapping.is_first_pp_rank(
        ) and not self._has_received_cache:
            self._debug_logger(
                f"[AutoTuner] Receiving cache data from previous pp rank {self.mapping.prev_pp_rank()}"
            )
            profiling_cache = self._dist.recv_object(
                src=self.mapping.prev_pp_rank(),
                tag=PP_COMM_TAG_AUTOTUNING,
            )
            # Guarantee that only receive cache once during a single warm-up run
            # Notice that this flag should be reset after each warm-up run because isend is always called
            self._has_received_cache = True
            self.profiling_cache.merge_cache_data(profiling_cache)

    def cache_pp_send(self):
        if self.mapping.has_pp() and not self.mapping.is_last_pp_rank():
            self._debug_logger(
                f"[AutoTuner] Sending cache data to next pp rank {self.mapping.next_pp_rank()}"
            )
            self._dist.isend_object(
                self.profiling_cache.shareable(self.profiling_cache.cache),
                dest=self.mapping.next_pp_rank(),
                tag=PP_COMM_TAG_AUTOTUNING,
            ).wait()

    def clean_pp_flag(self):
        self._has_received_cache = False
