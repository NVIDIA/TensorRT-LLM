# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Kimi K3.1 family uses no Modeling V1 code (modeling_v2 README, "Modeling V1 code").

Every non-test module of the family is parsed (nothing is imported). Each use of TensorRT-LLM code outside
modeling_v2 (an import at module level or inside a function, or a ``tensorrt_llm.x.y`` attribute chain after a bare
package import) must be one of:

* a runtime interface listed in ``RUNTIME``, allowed in any family file. The engine helpers of the DFlash worker among
  them (``WORKER_ONLY``) are referenced only inside the body of a class derived from ``DFlashWorker``;
* an engine kernel or collective function, constant or module from ``KERNELS`` (classes only as listed in
  ``KERNEL_CLASSES``), or a kernel function (lower-case name) from ``KERNEL_FUNCTION_MODULES``, used only by an op or
  kernel module: a file under ``kernels/``, or a file under ``ops/`` that defines an ``OpWrapper`` or registers a torch
  custom op. Every kernel call goes through an op.

Nothing comes from ``FORBIDDEN`` (V1 layers, fused MoE, V1 config classes, the TensorRT flow's models, tools), only op
and kernel modules reference ``torch.ops.trtllm``, and nothing is imported dynamically (except by the loaders in
``OWN_PACKAGE_LOADERS``, which load only modules of their own package that this scan parses). An import that is none of
these fails as unclassified: add it to ``RUNTIME`` or ``KERNELS`` with the reason, or move the code into the family.
Every allowlist entry is used, so the lists state exactly what the family depends on.

The sm_107 Rubin targets and their ``rubin_*`` modules (``RUBIN_EXEMPT``) are not checked here: they follow the rules of
``test_rubin_static_rules.py``.
The configurable NVFP4 default uses a flat forward over engine operations and its own
cache/runtime implementations rather than the MXFP4 family's OpWrapper contract.

The general target's Helix CP file (``HELIX_EXEMPT``) may import exactly the engine names listed for it there, for the
reason ``HELIX_EXEMPT_REASON`` states; every other use in it is checked like any family file's.
"""

import ast
from fnmatch import fnmatch
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu_only

_REPO = Path(__file__).resolve().parents[6]
_FAM = _REPO / "tensorrt_llm/_torch/_experimental/modeling_v2/models/kimi_k31"
_MV2 = "tensorrt_llm._torch._experimental.modeling_v2"
_ALL = "*"

#: Family files (paths relative to the family) these rules skip, and why: the sm_107 Rubin targets and the kernel and op
#: modules only they use, which run the engine interfaces their own static rules (``test_rubin_static_rules.py``) allow.
RUBIN_EXEMPT = (
    "kimi_k31_pro_nvfp4__sm_10x__default/*.py",  # configurable NVFP4 engine integration
    "kimi_k31_pro_nvfp4__sm_107__*/*.py",  # the three sm_107 targets (dep16, helix8, tep8)
    "kernels/*/rubin_*.py",  # their kernels
    "ops/*/rubin_*.py",  # their ops and runtime helpers (DSA metadata and backend, KDA cache manager, collectives)
    "ops/*/_rubin_*.py",  # the private helpers of those ops
)


#: Family files whose ``importlib.import_module`` loads only modules of their own package, which this scan parses like
#: any family file: the vendored vLLM trees' ``module(name)`` loaders (``kernels/SOURCE.json``).
OWN_PACKAGE_LOADERS = (
    "kernels/vendor/vllm_main/__init__.py",
    "kernels/vendor/vllm_mr8/__init__.py",
)


def _rubin_exempt(rel: Path) -> bool:
    return any(fnmatch(rel.as_posix(), pattern) for pattern in RUBIN_EXEMPT)


#: Why ``HELIX_EXEMPT`` exists.
HELIX_EXEMPT_REASON = (
    "Helix CP on the general target follows the engine Helix support (mapping + attention backend); "
    "functional only"
)
#: Family file (path relative to the family) -> the engine names (module -> names) it, and no other file, may import for
#: Helix CP. The general target's ``modeling.py`` and ``weights.py`` import none of them: they reach the engine Helix
#: support only through this file.
HELIX_EXEMPT = {
    "ops/attention/helix.py": {
        # The causal LM's Helix mapping swap (``_setup_helix_mappings``).
        "tensorrt_llm._torch.models.modeling_kimi_linear": {"KimiLinearForCausalLM"},
        # The MLA layers' attention backend and the combine of their CP partials.
        "tensorrt_llm._torch.attention.backends.utils": {"create_attention"},
        "tensorrt_llm._torch.attention.attention": {"_helix_post_process"},
        "tensorrt_llm._torch.attention.mla": {"helix_gen_zero_kv_mask"},
        # Runtime interfaces of the drafter's attention over CP-owned pages.
        "tensorrt_llm._torch.distributed.ops": {"cp_allgather"},
        "tensorrt_llm._torch.speculative.dflash_attention": {
            "HelixDraftShard",
            "combine_helix_partials",
        },
    },
}


#: Modules nothing in the family may import from (except the names ``RUNTIME`` / ``KERNEL_FUNCTION_MODULES`` allow).
FORBIDDEN = (
    "tensorrt_llm._torch.modules",
    "tensorrt_llm._torch.moe",
    "tensorrt_llm._torch.configs",
    "tensorrt_llm.models",
    "tensorrt_llm.tools",
)
#: Engine modules whose module-level kernel FUNCTIONS an op may import (the Mistral family's ops do the same with
#: ``modules.fla``). Classes and submodules from them stay forbidden. Nothing under ``_torch/modules`` is listed: the
#: family's kernels live in its own ``kernels/``.
KERNEL_FUNCTION_MODULES = (
    # The fused DSA decode-metadata Triton launcher.
    "tensorrt_llm._torch.attention.backends.sparse.dsa.kernels",
)

#: Runtime interfaces: module -> names (``_ALL``: every name). The engine's model, attention and speculative
#: interfaces, the configuration and KV cache manager types, and general utilities.
RUNTIME = {
    "tensorrt_llm._torch.model_config": {"ModelConfig"},
    "tensorrt_llm._torch.models.modeling_utils": {"DecoderModel", "DecoderModelForCausalLM"},
    # Attention: the backend interface and the metadata classes, not the backends.
    "tensorrt_llm._torch.attention.backends.interface": _ALL,
    "tensorrt_llm._torch.attention.backends.trtllm": {"TrtllmAttentionMetadata"},
    # The DSA metadata class and the helpers of its interface, and the runtime config / forward-argument data types
    # .
    "tensorrt_llm._torch.attention.backends.sparse.dsa.metadata": {
        "DSAtrtllmAttentionMetadata",
        "build_req_idx_per_token",
    },
    "tensorrt_llm._torch.attention.backends.sparse.dsa.params": {
        "DSABackendForwardArgs",
        "DSAMetadataParams",
        "DSAParams",
    },
    # Mistral precedent: the DSA workspace helper function.
    "tensorrt_llm._torch.attention.backends.sparse.params": {
        "get_indexer_mqa_logits_workspace_bytes"
    },
    # Mistral precedent: the multimodal embedding helper functions (not the mixin).
    "tensorrt_llm._torch.models.modeling_multimodal_utils": {
        "find_input_mm_embeds",
        "get_multimodal_embeddings",
    },
    # Mistral precedent: the quantization config type of the TensorRT flow.
    "tensorrt_llm.models.modeling_utils": {"QuantConfig"},
    # Speculative decoding runtime: the draft KV cache switch, the spec metadata interface.
    "tensorrt_llm._torch.speculative": {"should_use_separate_draft_kv_cache", "interface"},
    "tensorrt_llm._torch.speculative.interface": _ALL,
    # The engine DFlash worker: the targets subclass it for their model hooks (the worker owns the speculative
    # decoding loop the engine drives), with the engine helpers the subclass needs.
    "tensorrt_llm._torch.speculative.dflash": {
        "DFlashWorker",
        "DFlashSpecMetadata",
        "VocabParallelDraftTokens",
        "dflash_allocated_ctx_limit",
        "dflash_noise_block_embedding",
        "dflash_position_ceiling",
        "last_accepted_hidden",
    },
    "tensorrt_llm._torch.speculative.utils": {"get_num_extra_kv_tokens"},
    "tensorrt_llm._torch.speculative.vocab_parallel": {"VocabShardedLogits"},
    # Sampler host helpers the DFlash worker's sampling uses (the sampling kernels stay inside ops).
    "tensorrt_llm._torch.pyexecutor.sampler.ops.flashinfer": {"resolve_advanced_sampling_filters"},
    "tensorrt_llm._torch.pyexecutor.sampler.ops.vanilla": {"GREEDY_TEMPERATURE_THRESHOLD"},
    # The KDA state cache manager: a runtime cache-manager class the engine selects and builds, which the family cannot
    # move (product code outside the family stays unchanged).
    "tensorrt_llm._torch.modules.kimi_kda.cache_manager": {
        "KimiK3HybridCacheManager",
        "KDAReplayState",
    },
    # The engine's Mamba metadata base: the type of AttentionMetadata.mamba_metadata_cls and the engine's default; the
    # family's KdaMetadata (ops/ssm/kda_metadata.py) derives from it.
    "tensorrt_llm._torch.modules.mamba.mamba2_metadata": {"Mamba2Metadata"},
    # The engine's multi-stream switch, read only by ops/common/multi_stream.py so the same CUDA graphs use two streams.
    "tensorrt_llm._torch.modules.multi_stream_utils": {"do_multi_stream"},
    # Runtime hooks the engine calls into or the model reports through.
    "tensorrt_llm.serve.extensions.kimi_k3": {"KimiK3ServingExtension"},
    "tensorrt_llm._torch.route_capture": {"get_active_route_capture"},
    # Runtime utilities: the stream enum, the per-request prefill graph flag, the model's extra-attribute registry,
    # the torch.compile switch, the CuTe DSL availability flag and the SM count deep_gemm schedules for.
    "tensorrt_llm._torch.utils": {
        "AuxStreamType",
        "get_model_extra_attrs",
        "get_per_request_prefill_cuda_graph_flag",
        "maybe_compile",
    },
    "tensorrt_llm._torch.cute_dsl_utils": {"IS_CUTLASS_DSL_AVAILABLE"},
    "tensorrt_llm.deep_gemm": {"get_num_sms"},
    "tensorrt_llm._torch.speculative.spec_tree_manager": {"SpecTreeManager"},
    # The engine's own resolution of the disaggregated KV transceiver, which the family's support check must agree with.
    "tensorrt_llm._torch.pyexecutor._util": {"_resolve_disagg_transceiver_route"},
    # Routed-count recording for the DEP16 static expert placement workflow (inert unless EXPERT_STATISTIC_ITER_RANGE
    # is set).
    "tensorrt_llm._torch.moe.expert_statistic": {"ExpertStatistic"},
    # The all-to-all communication timeout budget the engine registers for a MoE communication op (warmup budget and
    # the TRTLLM_MOE_COMM_*TIMEOUT_SEC overrides).
    "tensorrt_llm._torch.moe.fused_moe.moe_comm_timeout_guard": {"register_moe_comm_timeout_proxy"},
    # A bare ``import tensorrt_llm`` (``{"*"}`` here is a name, not ``_ALL``). Every ``tensorrt_llm.x.y`` attribute
    # chain after it is classified like an import of ``y`` from ``tensorrt_llm.x``.
    "tensorrt_llm": {"*"},
    # KV / state cache managers and their metadata.
    "tensorrt_llm._torch.pyexecutor.resource_manager": _ALL,
    "tensorrt_llm._torch.pyexecutor.kv_cache": _ALL,
    "tensorrt_llm.runtime.kv_cache_manager_v2": _ALL,
    # Requests, CUDA graphs and the executor's input commit.
    "tensorrt_llm._torch.pyexecutor.llm_request": _ALL,
    "tensorrt_llm._torch.pyexecutor.cuda_graph_runner": _ALL,
    "tensorrt_llm._torch.pyexecutor.breakable_cuda_graph": _ALL,
    "tensorrt_llm._torch.pyexecutor.input_commit": _ALL,
    "tensorrt_llm._torch.memory_buffer_utils": _ALL,
    "tensorrt_llm._torch.nccl_window_tensor_scope": _ALL,
    "tensorrt_llm._device_copy": _ALL,
    # Configuration, inputs and utilities.
    "tensorrt_llm.llmapi": _ALL,
    "tensorrt_llm.inputs": _ALL,
    "tensorrt_llm.sampling_params": _ALL,
    "tensorrt_llm.mapping": _ALL,
    "tensorrt_llm.logger": _ALL,
    "tensorrt_llm._utils": _ALL,
    # The quantization algorithm enum (the quantization kernels and utilities are KERNELS).
    "tensorrt_llm.quantization": {"QuantAlgo"},
    "tensorrt_llm.quantization.mode": {"QuantAlgo"},
    # Configuration enums: the type of ModelConfig.allreduce_strategy (an IntEnum of ``functional``, imported through
    # its ``_torch.distributed`` re-export) and the position-embedding type.
    "tensorrt_llm.functional": {"PositionEmbeddingType"},
    "tensorrt_llm._torch.distributed": {"AllReduceStrategy"},
    "tensorrt_llm.bindings": _ALL,
}

#: Engine helpers of the DFlash worker subclass: allowed by ``RUNTIME``, but referenced only inside the
#: body of a class derived from ``DFlashWorker``. The drafter model itself is family code.
WORKER_ONLY = {
    "tensorrt_llm._torch.speculative.dflash": {
        "VocabParallelDraftTokens",
        "dflash_allocated_ctx_limit",
        "dflash_noise_block_embedding",
        "dflash_position_ceiling",
        "last_accepted_hidden",
    },
    "tensorrt_llm._torch.speculative.utils": {"get_num_extra_kv_tokens"},
    "tensorrt_llm._torch.speculative.vocab_parallel": {"VocabShardedLogits"},
}

#: Engine kernel and collective functions (module prefixes), allowed only inside an op or kernel module.
KERNELS = (
    "tensorrt_llm._torch.custom_ops",
    "tensorrt_llm._torch.attention.kernels",
    "tensorrt_llm._torch.flashinfer_utils",
    "tensorrt_llm._torch.cute_dsl_utils",
    "tensorrt_llm._torch.utils",
    "tensorrt_llm._torch.distributed",
    "tensorrt_llm._torch.pyexecutor.sampler.ops",
    "tensorrt_llm._torch.speculative.spec_tail_kernels",
    "tensorrt_llm._torch.speculative.vocab_parallel",
    "tensorrt_llm.deep_gemm",
    "tensorrt_llm.flash_mla",
    "tensorrt_llm._torch.kernels",
    "tensorrt_llm._torch.disaggregation.native.utils",
    "tensorrt_llm.quantization.utils",
)
#: Classes an op or kernel module may import from ``KERNELS`` (everything else imported from there must be a function,
#: an upper-case constant or a module): kernel objects and the engine's communication handles the ops drive.
KERNEL_CLASSES = {
    # The TP all-reduce handle, its fusion enum and parameters (built by the all-reduce op; the targets hold the handle
    # so that the engine's MNNVL prescale walk finds it).
    "tensorrt_llm._torch.distributed": {"AllReduce", "AllReduceFusionOp", "AllReduceParams"},
    "tensorrt_llm._torch.distributed.ops": {"MNNVLAllReduce"},
    # MNNVL workspace memory of the all-to-all and push all-reduce ops.
    "tensorrt_llm._torch.distributed.mnnvl_memory": {"CftMnnvlMemory", "MnnvlMemory"},
    # The CuTe DSL FP8 paged MQA-logits kernel the indexer op compiles.
    "tensorrt_llm._torch.attention.kernels.blackwell.paged_mqa_logits.fp8_paged_mqa_logits": {
        "FP8MQALogitsKernel"
    },
    # The engine's vocab-parallel pair exchange (private, no public name), which the vocab-parallel head op drives.
    "tensorrt_llm._torch.speculative.vocab_parallel": {"_PairExchange"},
}

#: Allowlist entries the family uses, filled by ``_violations``: ``(module, name)`` of ``RUNTIME`` (name ``_ALL`` for a
#: whole-module entry), and the ``KERNEL_FUNCTION_MODULES`` prefixes.
_USED: set = set()


def _runtime_allows(module: str, name: str) -> bool:
    for key, names in RUNTIME.items():
        if module == key or (names is _ALL and module.startswith(key + ".")):
            if names is _ALL or name in names:
                _USED.add((key, _ALL if names is _ALL else name))
                return True
            return False
    return False


def _is_function_name(module: str, name: str) -> bool:
    """A lower-case name that is not a submodule: a function (or constant), not a class or a module."""
    sub = _REPO / Path(*f"{module}.{name}".split("."))
    return name[:1].islower() and not sub.with_suffix(".py").exists() and not sub.is_dir()


_OP_MODULES: dict = {}


def _kernel_name_allowed(module: str, name: str) -> bool:
    """A name an op may import from ``KERNELS``: a module, a function, an upper-case constant, or a listed class."""
    if name == _ALL:
        return True
    sub = _REPO / Path(*f"{module}.{name}".split("."))
    if sub.with_suffix(".py").exists() or sub.is_dir():
        return True
    bare = name.lstrip("_")
    if bare[:1].islower() or bare.upper() == bare:
        return True
    if name in KERNEL_CLASSES.get(module, ()):
        _USED.add((module, name))
        return True
    return False


def _is_op_or_kernel(rel: Path) -> bool:
    """A kernel module (under ``kernels/``), or an op module: a file under ``ops/`` that defines an ``OpWrapper``
    subclass or registers a torch custom op. Other files under ``ops/`` (planners, cache managers, metadata) are
    support code and call kernels only through ops.
    """
    if "kernels" in rel.parts[:-1]:
        return True
    if "ops" not in rel.parts[:-1]:
        return False
    if rel not in _OP_MODULES:
        tree = ast.parse((_FAM / rel).read_text(encoding="utf-8"))
        wraps = any(
            isinstance(node, ast.ClassDef)
            and any(_dotted(base).endswith("OpWrapper") for base in node.bases)
            for node in ast.walk(tree)
        )
        registers = any(
            isinstance(node, ast.Call)
            and _dotted(node.func) in ("torch.library.custom_op", "torch.library.define")
            for node in ast.walk(tree)
        )
        _OP_MODULES[rel] = wraps or registers
    return _OP_MODULES[rel]


def _imports_of(node, module: str, is_package: bool):
    """(line, module, name) of one import statement, relative imports resolved."""
    if isinstance(node, ast.ImportFrom):
        if node.level:
            base = module.split(".") if is_package else module.split(".")[:-1]
            base = base[: len(base) - node.level + 1] if node.level > 1 else base
            source = ".".join(base + ([node.module] if node.module else []))
        else:
            source = node.module or ""
        for alias in node.names:
            yield node.lineno, source, alias.name
    elif isinstance(node, ast.Import):
        for alias in node.names:
            yield node.lineno, alias.name, _ALL


def _imports(path: Path, module: str, is_package: bool):
    """(line, module, name) of every import in ``path``, relative imports resolved."""
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        yield from _imports_of(node, module, is_package)


def _parents(tree: ast.AST) -> dict:
    return {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}


def _dotted(node: ast.AST) -> str:
    """``a.b.c`` of a ``Name`` / ``Attribute`` chain, or ``""``."""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return ""
    return ".".join([node.id, *reversed(parts)])


def _trtllm_op_references(path: Path):
    """(line, name) of every code reference to ``torch.ops.trtllm`` (a call, an alias, ``getattr``; strings do not
    count): ``torch.ops.trtllm.<op>`` when an op name follows, else ``torch.ops.trtllm``.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    parents = _parents(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and _dotted(node) == "torch.ops.trtllm":
            parent = parents.get(node)
            name = (
                f"torch.ops.trtllm.{parent.attr}"
                if isinstance(parent, ast.Attribute)
                else "torch.ops.trtllm"
            )
            yield node.lineno, name


def _binds_package(path: Path) -> bool:
    """Whether an ``import tensorrt_llm[...]`` without ``as`` binds the name ``tensorrt_llm`` in the module."""
    return any(
        isinstance(node, ast.Import)
        and any(
            alias.name.split(".")[0] == "tensorrt_llm" and alias.asname is None
            for alias in node.names
        )
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
    )


def _package_attribute_uses(path: Path):
    """(line, module, name) of every ``tensorrt_llm.<...>`` attribute chain in code (after a bare package import),
    classified like an import of its last name from the module before it.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    parents = _parents(tree)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute) or isinstance(parents.get(node), ast.Attribute):
            continue
        dotted = _dotted(node)
        if dotted.startswith("tensorrt_llm.") and not dotted.startswith(_MV2):
            module, name = dotted.rsplit(".", 1)
            yield node.lineno, module, name


def _dynamic_imports(path: Path):
    """Lines of ``importlib.import_module(...)`` / ``__import__(...)`` calls: imports the static check cannot see."""
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Call) and _dotted(node.func) in (
            "importlib.import_module",
            "__import__",
        ):
            yield node.lineno


def _worker_only_misuses(path: Path, module: str, is_package: bool):
    """(line, name) of every reference to a ``WORKER_ONLY`` import outside a ``DFlashWorker`` subclass body."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    aliases = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            _, source, _ = next(_imports_of(node, module, is_package), (0, "", ""))
            for alias in node.names:
                if alias.name in WORKER_ONLY.get(source, ()):
                    aliases.add(alias.asname or alias.name)
    if not aliases:
        return []
    # The engine worker under whatever local name it is imported as, plus any class named like a DFlash worker that
    # the file imports from the family (a family subclass); then every class of this file derived from one of them.
    workers = {"DFlashWorker"}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            _, source, _ = next(_imports_of(node, module, is_package), (0, "", ""))
            for alias in node.names:
                local = alias.asname or alias.name
                if (
                    source == "tensorrt_llm._torch.speculative.dflash"
                    and alias.name == "DFlashWorker"
                ) or (source.startswith(_MV2) and alias.name.endswith("DFlashWorker")):
                    workers.add(local)
    changed = True
    while changed:
        changed = False
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name not in workers:
                bases = {
                    b.id if isinstance(b, ast.Name) else getattr(b, "attr", "") for b in node.bases
                }
                if bases & workers:
                    workers.add(node.name)
                    changed = True
    found = []

    def visit(node, inside):
        if isinstance(node, ast.ClassDef):
            inside = inside or node.name in workers
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            return
        if isinstance(node, ast.Name) and node.id in aliases and not inside:
            found.append((node.lineno, node.id))
        for child in ast.iter_child_nodes(node):
            visit(child, inside)

    visit(tree, False)
    return found


def _violations():
    found = []
    for path in sorted(_FAM.rglob("*.py")):
        rel = path.relative_to(_FAM)
        if _rubin_exempt(rel):
            continue
        module = ".".join(path.relative_to(_REPO).with_suffix("").parts)
        is_package = path.name == "__init__.py"
        if is_package:
            module = module.rsplit(".", 1)[0]
        uses = list(_imports(path, module, is_package))
        if _binds_package(path):
            uses += list(_package_attribute_uses(path))
        helix = HELIX_EXEMPT.get(rel.as_posix(), {})
        for line, source, name in uses:
            if not source.startswith("tensorrt_llm") or source.startswith(_MV2):
                continue
            where = f"{rel}:{line} {source}.{name}"
            if name in helix.get(source, ()):
                _USED.add((rel.as_posix(), source, name))
                continue
            if _runtime_allows(source, name):
                continue
            if (
                source.startswith(KERNEL_FUNCTION_MODULES)
                and _is_op_or_kernel(rel)
                and _is_function_name(source, name)
            ):
                _USED.update(
                    ("", prefix) for prefix in KERNEL_FUNCTION_MODULES if source.startswith(prefix)
                )
                continue
            if source.startswith(FORBIDDEN):
                found.append(f"forbidden (Modeling V1 / TensorRT flow): {where}")
            elif source.startswith(KERNELS):
                if not _is_op_or_kernel(rel):
                    found.append(f"kernel function outside an op: {where}")
                elif not _kernel_name_allowed(source, name):
                    found.append(f"kernel class not in KERNEL_CLASSES: {where}")
            elif source.startswith(("tensorrt_llm._torch.models", "tensorrt_llm._torch.attention")):
                found.append(f"Modeling V1 model / attention code: {where}")
            else:
                found.append(f"unclassified engine import: {where}")
        for line in () if rel.as_posix() in OWN_PACKAGE_LOADERS else _dynamic_imports(path):
            found.append(f"dynamic import (not checkable): {rel}:{line}")
        for line, name in _worker_only_misuses(path, module, is_package):
            found.append(
                f"DFlash worker helper outside a DFlashWorker subclass: {rel}:{line} {name}"
            )
        if not _is_op_or_kernel(rel):
            for line, op in _trtllm_op_references(path):
                found.append(f"direct torch.ops.trtllm call outside an op: {rel}:{line} {op}")
    return found


def test_family_uses_no_modeling_v1_code():
    found = _violations()
    assert not found, f"{len(found)} Modeling V1 / direct-kernel uses:\n" + "\n".join(found)


def test_allowlist_has_no_unused_entries():
    """Every allowlist entry is used by the family, so the lists state exactly what it depends on."""
    _USED.clear()
    _violations()
    unused = [
        f"{key}.{name}" if names is not _ALL else key
        for key, names in RUNTIME.items()
        for name in ([_ALL] if names is _ALL else sorted(names))
        if (key, name) not in _USED
    ]
    unused += [prefix for prefix in KERNEL_FUNCTION_MODULES if ("", prefix) not in _USED]
    unused += [
        f"{module}.{name}"
        for module, names in KERNEL_CLASSES.items()
        for name in sorted(names)
        if (module, name) not in _USED
    ]
    unused += [
        f"{rel}: {module}.{name}"
        for rel, modules in HELIX_EXEMPT.items()
        for module, names in modules.items()
        for name in sorted(names)
        if (rel, module, name) not in _USED
    ]
    assert not unused, "allowlist entries no family file uses (remove them):\n" + "\n".join(unused)


def test_own_package_loaders_load_only_scanned_modules():
    """Each ``OWN_PACKAGE_LOADERS`` file loads dynamically, and only by a name that starts with its own package
    (``__name__``, or a module-level constant ``__name__ + "..."``); every module under that package is one
    ``_violations`` parses (none is ``RUBIN_EXEMPT``), so the loaders hide no import from the scan."""
    found = []
    for rel in OWN_PACKAGE_LOADERS:
        path = _FAM / rel
        tree = ast.parse(path.read_text(encoding="utf-8"))
        own = {"__name__"}
        for node in tree.body:
            if (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and isinstance(node.value, ast.BinOp)
                and isinstance(node.value.op, ast.Add)
                and isinstance(node.value.left, ast.Name)
                and node.value.left.id == "__name__"
            ):
                own.add(node.targets[0].id)
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and _dotted(node.func) in ("importlib.import_module", "__import__")
        ]
        if not calls:
            found.append(f"{rel}: loads nothing dynamically (remove it from OWN_PACKAGE_LOADERS)")
        for call in calls:
            name = call.args[0] if call.args else None
            head = name.values[0] if isinstance(name, ast.JoinedStr) and name.values else None
            if not (
                isinstance(head, ast.FormattedValue)
                and isinstance(head.value, ast.Name)
                and head.value.id in own
            ):
                found.append(f"{rel}:{call.lineno}: loads a module not named from its own package")
        for module in sorted(path.parent.rglob("*.py")):
            if _rubin_exempt(module.relative_to(_FAM)):
                found.append(
                    f"{rel}: {module.relative_to(_FAM)} is RUBIN_EXEMPT, so the scan skips it"
                )
    assert not found, "\n".join(found)
