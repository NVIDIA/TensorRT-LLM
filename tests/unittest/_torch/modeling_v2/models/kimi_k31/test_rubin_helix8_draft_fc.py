# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""helix8's drafter projection on CUDA graph steps: the fc GEMM and the hidden norm of the
step's captured target rows run on the aux stream before the acceptance, beside it and its
bookkeeping, instead of after them inside the fused preparation.

The projection reads only the captured rows (written by the target forward), fc and the hidden
norm weight: nothing the acceptance computes. The preparation (the engine's
``_prepare_paged_latent_gen_inputs``) takes the projection of exactly those rows after waiting for
its event. Same kernels on the same rows, so the same bits; only the stream and the place in the
step change.

CPU (``cpu_only``), through the recorder of ``test_rubin_helix8_fork_q.py``: the worker's
``_forward_impl`` and the drafter's projection methods run from helix8's modeling.py, and the
preparation from the engine's dflash.py, over recording stand-ins, on a generation step of three
requests at k5:

* on a graph step the fc GEMM and the hidden norm run once each, on the aux stream, issued before
  the acceptance and after a wait that covers the target forward's captures, over the
  preparation's rows of the capture buffer; the once-per-process INFO line is logged;
* nothing the acceptance or its bookkeeping wrote reaches the projection; the preparation's
  latent projection reads the aux stream's result after a wait that covers it, and every kernel
  touches memory after its earlier users;
* the same step without that wait is flagged; an issue of other rows than the preparation's is
  refused;
* an eager step projects inside the preparation on the current stream, after the acceptance, and
  logs nothing.
"""

from __future__ import annotations
import __future__

import ast
import contextlib
import copy
import functools
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

_SPEC = importlib.util.spec_from_file_location(
    "_helix8_fork_q_recorder", Path(__file__).with_name("test_rubin_helix8_fork_q.py")
)
fq = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(fq)

pytestmark = pytest.mark.cpu_only

_ENGINE = fq._PKG_ROOT / "_torch/speculative/dflash.py"
_LINE_PREFIX = "Kimi K3.1 Rubin DFlash2: drafter fc projection"

# K3.1's drafter: hidden 7168, six captured target layers; a k5 verify block of 6 rows a request.
_H, _CAPTURES, _K = 7168, 6, 5
_WIDTH = _K + 1
_GENS = 3
_ROWS = _GENS * _WIDTH
_VOCAB = 64
_LAYERS = 5
# The drafter's methods and the worker's, run from helix8's modeling.py; the preparation from
# the engine.
_DRAFT_METHODS = (
    "project_target_hidden",
    "_project",
    "issue_projection",
    "join_taps",
    "projection_pending",
)
_WORKER_METHODS = (
    "_issue_draft_projection",
    "_forward_impl",
    "_block_gather_ids",
    "_block_rows",
    "_fused_select_accept",
    "_greedy_tokens_apply",
    "_fused_kv_rewind_operands",
)
_ENGINE_METHODS = (
    "prepare_1st_drafter_inputs",
    "_fused_gen_prepare_applies",
    "_prepare_paged_latent_gen_inputs",
)
# The acceptance and its bookkeeping, as the step issues them before the preparation.
_ACCEPT_KERNELS = ("acceptance", "update_mamba_states", "prepare_attn", "prepare_kv")


@functools.lru_cache(maxsize=None)
def _engine_tree() -> ast.Module:
    return ast.parse(_ENGINE.read_text(), filename=str(_ENGINE))


def _class(tree: ast.Module, name: str, methods: tuple, base: str, ns: dict, filename: str):
    """Class ``name`` of ``tree`` cut down to ``methods``, on top of ``ns[base]``."""
    (cls,) = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name]
    body = [copy.deepcopy(n) for n in cls.body if isinstance(n, ast.FunctionDef)]
    body = [n for n in body if n.name in methods]
    missing = set(methods) - {n.name for n in body}
    assert not missing, f"{filename} defines no {name}.{sorted(missing)}"
    kwargs = {"type_params": []} if "type_params" in ast.ClassDef._fields else {}
    node = ast.ClassDef(
        name=name,
        bases=[ast.Name(base, ast.Load())],
        keywords=[],
        body=body,
        decorator_list=[],
        **kwargs,
    )
    module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    flags = __future__.annotations.compiler_flag
    exec(compile(module, filename, "exec", flags=flags, dont_inherit=True), ns)
    return ns[name]


def _functions(tree: ast.Module, names: tuple, ns: dict, filename: str) -> None:
    body = [copy.deepcopy(n) for n in tree.body if isinstance(n, ast.FunctionDef)]
    body = [n for n in body if n.name in names]
    assert {n.name for n in body} == set(names), f"{filename} defines no {names}"
    module = ast.fix_missing_locations(ast.Module(body=body, type_ignores=[]))
    flags = __future__.annotations.compiler_flag
    exec(compile(module, filename, "exec", flags=flags, dont_inherit=True), ns)


class _ClockPlan(fq._Plan):
    """The fork_q plan, which also keeps what each kernel's stream had waited for when it ran."""

    def __init__(self):
        super().__init__()
        self.waited: list = []

    def kernel(self, name: str, payload) -> None:
        super().kernel(name, payload)
        self.waited.append(dict(self.clock[self.stream]))


class _TorchHere(fq._Torch):
    """The recorder's torch, with factory calls that name ``device="cuda"`` made on the CPU."""

    _FACTORIES = frozenset({"arange", "zeros", "ones", "empty", "full", "tensor"})

    def __getattr__(self, name):
        fn = getattr(torch, name)
        if name not in self._FACTORIES:
            return fn

        def call(*args, **kwargs):
            kwargs.pop("device", None)
            return fn(*args, **kwargs)

        return call


def _kernel(plan: _ClockPlan, name: str, result=None):
    """A stand-in kernel ``name`` on the current stream over what it is passed and returns."""

    def run(*args, **kwargs):
        out = None if result is None else result(*args, **kwargs)
        plan.kernel(name, (args, kwargs, out))
        return out

    return run


class _DraftBase:
    """The drafter container's attributes the methods under test read."""

    def __init__(self, plan, streams, weights):
        self._plan = plan
        self._weights = weights
        self.config = SimpleNamespace(rms_norm_eps=1e-5)
        self._rt = SimpleNamespace(
            aux_stream=streams["aux"], draft=SimpleNamespace(fc_ready=fq._Event(plan)), taps=None
        )
        self._early_projection = None
        self.context_kv_only = False
        self.mask_token_id = 7
        self.mask_token_embedding = None
        self.draft_model_full = SimpleNamespace(
            model=SimpleNamespace(embed_tokens=weights["embed"]), lm_head=None
        )
        self.lm_head = weights["lm_head"]
        self.logits_processor = _kernel(
            plan, "draft_logits", lambda h, head, md, gather: torch.empty(h.shape[0], _VOCAB)
        )
        self.precompute_context_kv = _kernel(
            plan,
            "precompute_context_kv",
            lambda x, pos: (torch.empty(x.shape[0], _LAYERS, 1, 576, dtype=x.dtype), None),
        )
        self.dflash_forward = _kernel(
            plan,
            "dflash_forward",
            lambda **kw: torch.empty(
                kw["noise_embedding"].shape[0] * kw["noise_embedding"].shape[1], _H
            ),
        )

    def _w(self, key: str):
        return self._weights[key]


class _WorkerBase:
    """The engine worker's state and methods ``_forward_impl`` reaches, as stand-ins."""

    def __init__(self, plan):
        p = plan
        self._plan = plan
        self.spec_config = SimpleNamespace(max_draft_len=_K, mask_token_id=None)
        self._resolved_mask_token_id = 7
        self._compute_block_size = _WIDTH
        self._paged_latent_prepare = True
        self._ctx_buf_inited = True
        self._ctx_kv_manager = self._manager = object()
        self._accept_stats = None
        self._dummy_slot = _GENS
        self._max_ctx = 4096
        self._ctx_page_size = 64
        self._ctx_len = torch.zeros(_GENS + 1, dtype=torch.long)
        self._ctx_position_offset = torch.zeros(_GENS + 1, dtype=torch.long)
        self._ctx_block_counts = torch.ones(_GENS + 1, dtype=torch.long)
        self._ctx_block_tables = torch.zeros(_GENS + 1, 4, dtype=torch.int32)
        self._ctx_kv_buf = [torch.empty(4, 1, 1, 64, 576, dtype=torch.float8_e4m3fn)]
        self._ctx_kv_ptrs = torch.zeros(1, dtype=torch.int64)
        self._ctx_k_buf = self._ctx_v_buf = None
        self._batch_to_slot_buffer = SimpleNamespace(tensor=torch.arange(_GENS + 1))
        self._execute_guided_decoder_if_present = lambda logits: None
        self.sample_and_accept_draft_tokens = _kernel(
            p,
            "acceptance",
            lambda logits, md, spec: (
                torch.empty(_GENS, _K + 1, dtype=torch.int32),
                torch.empty(_GENS, dtype=torch.int32),
            ),
        )
        self._prepare_attn_metadata_for_dflash = _kernel(p, "prepare_attn")
        self._prepare_kv_for_draft_forward = _kernel(p, "prepare_kv")
        self._draft_slot_ids = lambda dm, n, block, k: torch.arange(n * k)
        self._apply_dflash2_selector = _kernel(p, "selector", lambda dm, logits, *rest: logits)
        self.sample_draft_tokens = _kernel(
            p, "draft_tokens", lambda logits, spec, bs, num_contexts: torch.empty(_GENS, _K)
        )
        self.write_context_onehot_draft_probs = lambda *args: None
        self._restore_attn_metadata_from_spec_dec = lambda md: None
        self._apply_kv_rewind_after_draft = lambda md, spec: None
        self._rollback_guided_decoder_after_verify = lambda n: None
        self._prepare_next_new_tokens = lambda *args: None

    @property
    def _draft_tokens_per_req(self) -> int:
        return self.spec_config.max_draft_len + 1

    def get_draft_kv_cache_manager(self, resource_manager):
        return self._manager

    def _helix_shard(self):
        return None

    def _has_unified_draft_cache(self) -> bool:
        return True

    def _draft_shard_head(self, draft_model, spec_metadata):
        """No vocab-parallel head: the drafter's logits are gathered (the path the projection
        test was written on). The shard path, and when the worker takes it, are
        test_rubin_helix8_draft_shard.py's; the projection's stream order does not depend on which
        head the logits come from."""
        return None

    @contextlib.contextmanager
    def draft_kv_cache_context(self, md, manager):
        yield


def _prep(plan):
    """``dflash_paged_latent_prepare``: the acceptance-derived positions and stores."""

    def result(num_accepted, accepted, slots, *rest, **kwargs):
        n = _GENS
        return SimpleNamespace(
            bonus=torch.empty(n, dtype=torch.long),
            ctx_positions=torch.empty(n * _WIDTH, dtype=torch.long),
            query_positions=torch.empty(n, _WIDTH, dtype=torch.long),
            num_ctx=torch.empty(n, dtype=torch.long),
            store_pages=torch.empty(n * _WIDTH, dtype=torch.int32),
            store_offsets=torch.empty(n * _WIDTH, dtype=torch.int32),
            store_mask=torch.empty(n * _WIDTH, dtype=torch.bool),
            blk_pages=torch.empty(n, _WIDTH, dtype=torch.int32),
            blk_offsets=torch.empty(n, _WIDTH, dtype=torch.int32),
            blk_seq_lens=torch.empty(n, dtype=torch.int32),
            blk_kv_bounds=torch.empty(n, 2, dtype=torch.int32),
            blk_owned=torch.empty(n, _WIDTH, dtype=torch.int32),
        )

    return _kernel(plan, "dflash_paged_latent_prepare", result)


def _run(cuda_graph: bool = True, tree: ast.Module | None = None, issue_offset: int = 0):
    """One generation step of ``_forward_impl`` of ``tree`` (helix8's modeling.py by default).

    ``issue_offset`` moves the rows ``_issue_draft_projection`` issues by that many rows (a
    preparation of other rows than the issued ones)."""
    plan = _ClockPlan()
    streams = {s: fq._Stream(plan, s) for s in fq._Plan.STREAMS}
    log: list = []
    logger = SimpleNamespace(
        info_once=lambda *msg, key: log.append((" ".join(msg), key)),
        info=lambda *msg: None,
        warning=lambda *msg: None,
    )
    torch_here = _TorchHere(plan, streams)
    weights = {
        "d_fc": fq._shape_only(_H, _CAPTURES * _H),
        "d_hidden_norm": fq._shape_only(_H),
        "embed": fq._shape_only(_VOCAB, _H),
        "lm_head": fq._shape_only(_VOCAB, _H),
    }
    tree = tree or fq._tree()
    target_ns = {
        **fq._constants(),
        # The step as the engine issues it: the drafter-tail switches off (their fused launches
        # and their order against the early projection have their own tests).
        "_DRAFT_SELECT_ACCEPT": False,
        "_DRAFT_GREEDY_TOKENS": False,
        "_DRAFT_INDEX_PAIRS": False,
        "_DRAFT_TAIL_REWIND": False,
        "_DRAFT_NOISE_BLOCK": False,
        "torch": torch_here,
        "logger": logger,
        "rubin_linear": _kernel(
            plan, "rubin_linear", lambda x, w, tf32: torch.empty(x.shape[0], w.shape[0])
        ),
        "flashinfer_rmsnorm": _kernel(
            plan, "flashinfer_rmsnorm", lambda x, w, eps: torch.empty(x.shape, dtype=x.dtype)
        ),
        "VocabShardedLogits": type("VocabShardedLogits", (), {}),
        "dflash2_sampled_step": lambda worker, spec: False,
        "_DraftBase": _DraftBase,
    }
    _functions(tree, ("_rows_key",), target_ns, str(fq._MODELING))
    draft_cls = _class(tree, "DraftModel", _DRAFT_METHODS, "_DraftBase", target_ns, "modeling.py")
    engine_ns = {
        "torch": torch_here,
        "dflash_paged_latent_prepare": _prep(plan),
        "dflash_noise_block_embedding": _kernel(
            plan,
            "noise_embedding",
            lambda emb, bonus, mask, block, trained: torch.empty(
                bonus.shape[0], block, _H, dtype=torch.bfloat16
            ),
        ),
        "store_context_latents": _kernel(plan, "store_context_latents"),
        "DraftBlockPlacement": lambda *fields: SimpleNamespace(fields=fields),
        "_WorkerBase": _WorkerBase,
    }
    engine_cls = _class(
        _engine_tree(), "DFlashWorker", _ENGINE_METHODS, "_WorkerBase", engine_ns, str(_ENGINE)
    )
    target_ns["_EngineWorker"] = engine_cls
    worker_cls = _class(tree, "DFlashWorker", _WORKER_METHODS, "_EngineWorker", target_ns, "m.py")
    if issue_offset:
        inner = worker_cls._issue_draft_projection

        def shifted(self, dm, md, spec, total, num_gens):
            moved = SimpleNamespace(
                **{**vars(md), "num_ctx_tokens": md.num_ctx_tokens + issue_offset}
            )
            return inner(self, dm, moved, spec, total, num_gens)

        worker_cls._issue_draft_projection = shifted

    draft = draft_cls(plan, streams, weights)
    worker = worker_cls(plan)
    captured = torch.empty(_ROWS + 4, _CAPTURES * _H, dtype=torch.bfloat16)
    spec = SimpleNamespace(
        runtime_draft_len=_K,
        is_cuda_graph=cuda_graph,
        captured_hidden_states=captured,
        get_hidden_states=lambda n: captured[:n, : _CAPTURES * _H],
        hidden_size=_H,
        request_ids=None,
        batch_indices_cuda=None,
        draft_tokens=torch.empty(_GENS * _K, dtype=torch.int32),
    )
    md = SimpleNamespace(
        num_seqs=_GENS,
        num_contexts=0,
        num_ctx_tokens=0,
        _seq_lens=torch.zeros(_GENS, dtype=torch.int32),
        _seq_lens_cuda=torch.zeros(_GENS, dtype=torch.int32),
        kv_cache_manager=SimpleNamespace(update_mamba_states=_kernel(plan, "update_mamba_states")),
        mamba_metadata=SimpleNamespace(state_indices=torch.empty(_GENS, dtype=torch.int32)),
    )
    logits = torch.empty(_ROWS, _VOCAB)
    with fq._Record(plan):
        # The target forward's capture taps, the last work on the current stream before the step's
        # worker: the projection must follow them.
        plan.kernel("target_capture", ((captured,), {}, None))
        out = worker._forward_impl(
            torch.empty(_ROWS, dtype=torch.long),
            torch.empty(1, _ROWS, dtype=torch.long),
            torch.empty(_ROWS, _H, dtype=torch.bfloat16),
            logits,
            md,
            spec,
            draft,
        )
    assert out["logits_vocab_shard"] is False
    return SimpleNamespace(plan=plan, log=log, draft=draft, captured=captured, weights=weights)


def _key(t: torch.Tensor) -> tuple:
    return (t.data_ptr(), tuple(t.shape), tuple(t.stride()), t.dtype)


def _index(plan: _ClockPlan, name: str) -> list:
    return [k for k, (_, _, kname, _) in enumerate(plan.kernels) if kname == name]


def _mine(log: list) -> list:
    return [line for line in log if line[0].startswith(_LINE_PREFIX)]


def test_a_graph_step_projects_on_the_aux_stream_before_the_acceptance():
    r = _run()
    plan = r.plan
    assert not plan.problems, "\n".join(plan.problems)
    (gemm,) = _index(plan, "rubin_linear")
    (norm,) = _index(plan, "flashinfer_rmsnorm")
    (accept,) = _index(plan, "acceptance")
    assert plan.kernels[gemm][1] == "aux" and plan.kernels[norm][1] == "aux"
    assert gemm < norm < accept, [name for _, _, name, _ in plan.kernels]
    args = plan.kernels[gemm][3][0]
    assert args[1] is r.weights["d_fc"] and args[2] is False, "not the fc GEMM in BF16"
    rows = r.captured[:_ROWS, : _CAPTURES * _H]
    assert _key(args[0]) == _key(rows), "the GEMM reads other rows than the capture"
    (capture,) = _index(plan, "target_capture")
    assert plan.waited[gemm].get("main", 0) > plan.kernels[capture][0], "before the captures"
    norm_args = plan.kernels[norm][3][0]
    assert norm_args[0] is plan.kernels[gemm][3][2] and norm_args[1] is r.weights["d_hidden_norm"]
    assert len(_mine(r.log)) == 1 and _mine(r.log)[0][0] == _mine(r.log)[0][1], r.log
    assert not r.draft.projection_pending()


def test_the_acceptance_never_reaches_the_projection_and_the_preparation_waits_for_it():
    plan = _run().plan
    (gemm,) = _index(plan, "rubin_linear")
    (norm,) = _index(plan, "flashinfer_rmsnorm")
    (latent,) = _index(plan, "precompute_context_kv")
    (prep,) = _index(plan, "dflash_paged_latent_prepare")
    accept_outputs = set()
    for name in _ACCEPT_KERNELS + ("dflash_paged_latent_prepare",):
        for k in _index(plan, name):
            out = plan.kernels[k][3][2]
            out = vars(out) if isinstance(out, SimpleNamespace) else out
            accept_outputs |= {fq._memory(t) for t in fq._tensors(out)}
    accept_outputs.discard(0)
    for k in (gemm, norm):
        read = {fq._memory(t) for t in fq._tensors(plan.kernels[k][3][:2])}
        assert not read & accept_outputs, "the projection reads something the acceptance wrote"
    for name in _ACCEPT_KERNELS:
        (k,) = _index(plan, name)
        assert plan.kernels[k][1] == "main", f"{name} left the current stream"
    projected = plan.kernels[norm][3][2]
    latent_args = plan.kernels[latent][3][0]
    assert latent_args[0] is projected, "the preparation does not take the aux stream's projection"
    assert latent_args[1] is plan.kernels[prep][3][2].ctx_positions
    assert plan.waited[latent].get("aux", 0) > plan.kernels[norm][0], "read before its event"
    assert plan.clock["main"]["aux"] > plan.kernels[norm][0], "the aux stream is not joined"
    assert len(_index(plan, "rubin_linear")) == 1, "the preparation projected a second time"


def _without_the_wait(tree: ast.Module) -> ast.Module:
    """``tree`` with ``project_target_hidden`` returning the early projection without its wait."""
    tree = copy.deepcopy(tree)
    (cls,) = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "DraftModel"]
    (fn,) = [
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "project_target_hidden"
    ]
    removed = []

    class _Drop(ast.NodeTransformer):
        def visit_Expr(self, node):
            if ast.unparse(node) == "ready.wait()":
                removed.append(node)
                return None
            return node

    _Drop().visit(fn)
    assert len(removed) == 1, "project_target_hidden has no ready.wait()"
    return ast.fix_missing_locations(tree)


def test_the_check_flags_a_preparation_without_the_wait():
    plan = _run(tree=_without_the_wait(fq._tree())).plan
    flagged = [p for p in plan.problems if p.startswith("precompute_context_kv on the main stream")]
    assert flagged, plan.problems


def test_an_issue_of_other_rows_is_refused():
    with pytest.raises(RuntimeError, match="DFlash2: the drafter projection issued before"):
        _run(issue_offset=1)


def test_an_eager_step_projects_inside_the_preparation_after_the_acceptance():
    r = _run(cuda_graph=False)
    plan = r.plan
    assert {s for _, s, _, _ in plan.kernels} == {"main"} and not plan.problems
    (gemm,) = _index(plan, "rubin_linear")
    (accept,) = _index(plan, "acceptance")
    (prep,) = _index(plan, "dflash_paged_latent_prepare")
    (latent,) = _index(plan, "precompute_context_kv")
    assert accept < prep < gemm < latent, [name for _, _, name, _ in plan.kernels]
    args = plan.kernels[gemm][3][0]
    assert _key(args[0]) == _key(r.captured[:_ROWS, : _CAPTURES * _H])
    assert not _mine(r.log), r.log
