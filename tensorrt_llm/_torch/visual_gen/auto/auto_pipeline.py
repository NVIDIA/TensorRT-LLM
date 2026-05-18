# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end pipeline wrapper for the auto path.

`AutoDiffusersPipeline` is the Diffusers-shaped factory used internally by
the user-facing `AutoTransformerPipeline` (see ``pipeline.py``): it loads
a Diffusers `DiffusionPipeline` from a checkpoint, captures + rewrites the
transformer through the auto path, and swaps the captured `GraphModule`
back into the Diffusers pipeline as the new `.transformer`. End-to-end
generation then goes through the Diffusers pipeline's `__call__`, which
runs text encoding, the scheduler loop (now invoking our captured
transformer), and VAE decode.

`AutoDiffusersPipeline` is *not* a `BasePipeline` subclass. The user-facing
`AutoTransformerPipeline` is the `BasePipeline`-shaped adapter that the
loader/executor see; it delegates the heavy lifting here.

Note: the Diffusers pipeline calls the transformer in eager — its
`__call__` accesses attributes like `.config`, `.dtype`, expects a
specific forward signature, and may attach forward hooks. The captured
`GraphModule` doesn't carry all of these. `_GraphModuleAsTransformer`
wraps the GraphModule with the minimum attribute surface needed.
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any, Optional

import torch
import torch.nn as nn

from tensorrt_llm.logger import logger

from . import ops  # noqa: F401 — registers torch.ops.visgen_auto.*

if TYPE_CHECKING:
    from ..config import DiffusionModelConfig
    from .adapter import VisGenFamilyAdapter


# --- Ulysses sequence sharding helpers ---------------------------------------
# Keys whose tensor lives in the image-sequence axis (S_img). The shard pass
# slices these along dim=1 (B, S, ...) so each rank only feeds its `S/P` tokens
# into the captured transformer. img_ids and txt_ids are RoPE positional ids;
# img_ids tracks the image sequence and is sharded the same way. txt_ids stays
# replicated (the encoder side has a fixed text seq length).
_IMG_SEQ_KWARGS = ("hidden_states", "img_ids")


def _shard_seq_kwargs(kwargs: dict[str, Any], vgm) -> dict[str, Any]:
    """Slice image-sequence-axis tensors so each rank sees `S/P` tokens.

    Operates on a *copy* of kwargs — leaves caller's dict alone. Tensors
    whose sequence dim isn't divisible by `ulysses_size` raise rather than
    silently misalign across ranks.
    """
    P = vgm.ulysses_size
    rank = vgm.ulysses_rank
    out = dict(kwargs)
    for k in _IMG_SEQ_KWARGS:
        t = out.get(k)
        if not isinstance(t, torch.Tensor):
            continue
        # Diffusers passes `img_ids` un-batched in some families (FLUX.1:
        # shape `(S, 3)`) and batched in others (FLUX.2: `(B, S, 4)`).
        # Heuristic: 2-D tensors are `(S, coord_dim)` (shard dim 0);
        # 3-D or higher are `(B, S, ...)` (shard dim 1).
        seq_dim = 0 if t.ndim == 2 else 1
        S = t.shape[seq_dim]
        if S % P != 0:
            raise ValueError(
                f"AutoDiffusersPipeline Ulysses: `{k}` seq dim {S} (axis "
                f"{seq_dim} of shape {tuple(t.shape)}) not divisible by "
                f"ulysses_size {P}. Pick a resolution where the post-patch "
                f"seq length is divisible by {P}."
            )
        local = S // P
        out[k] = t.narrow(seq_dim, rank * local, local).contiguous()
    return out


def _split_batch_args_kwargs(args, kwargs, vgm):
    """Slice the CFG-doubled batch (dim-0) by ``cfg_rank``.

    Diffusers pipelines that use classifier-free guidance build a
    CFG-doubled batch ``[uncond, cond]`` (or vice-versa) and pass it to
    the transformer. With ``cfg_size > 1`` we split that batch across the
    cfg group so each rank only forwards on its half — the gather on
    the way out re-assembles ``[uncond, cond]`` invisibly to the pipeline.

    Heuristic: only tensors whose dim-0 is a non-trivial multiple of
    ``cfg_size`` get sliced. Singleton or non-batch tensors (e.g. FLUX.1's
    un-batched ``img_ids`` of shape ``(S, 3)``) pass through unchanged.
    """
    P = getattr(vgm, "cfg_size", 1)
    if P <= 1:
        return args, kwargs
    R = vgm.cfg_rank

    def _is_batch_tensor(t) -> bool:
        # Heuristic: a 2-D tensor with shape `(S, coord_dim)` (FLUX.1 `img_ids`)
        # is un-batched; everything ≥3-D with non-trivial dim-0 is batched.
        return isinstance(t, torch.Tensor) and t.ndim >= 2 and t.shape[0] > 1

    # Detect the CFG-doubled batch. We expect at least one ≥3-D tensor whose
    # dim-0 is a non-trivial multiple of `cfg_size`. If we can't find any, the
    # caller is passing a single-sample batch under cfg_size>1 — silently
    # bypassing the split would let the gather (which expects `cfg_size` ranks
    # of output) collide with shape mismatches later. Fail loud.
    candidates = [t for t in list(args) + list(kwargs.values()) if _is_batch_tensor(t)]
    if not any(t.shape[0] % P == 0 and t.shape[0] >= P for t in candidates):
        raise ValueError(
            f"CFG-parallel ({P}-way) requested but no input tensor has a "
            f"dim-0 divisible by {P}. Diffusers pipelines must produce a "
            f"CFG-doubled batch (guidance_scale > 1.0) for CFG-parallel to "
            "make sense. Got dim-0 values: "
            f"{[t.shape[0] for t in candidates]}."
        )

    def _slice(t):
        if not isinstance(t, torch.Tensor) or t.ndim == 0:
            return t
        B = t.shape[0]
        if B % P != 0 or B // P < 1:
            return t
        local = B // P
        return t.narrow(0, R * local, local).contiguous()

    new_args = tuple(_slice(a) for a in args)
    new_kwargs = {k: _slice(v) for k, v in kwargs.items()}
    return new_args, new_kwargs


def _gather_batch_output(out, vgm):
    """All-gather output along batch dim (0) across the cfg group.

    Mirrors `_gather_seq_output` but on dim-0. Diffusers reads back
    `(B_full, ...)` after the transformer call (it then does the
    `uncond + gs * (cond - uncond)` guidance combine), so we restore the
    full batch before returning.
    """
    import torch.distributed as dist

    P = getattr(vgm, "cfg_size", 1)
    if P <= 1:
        return out
    pg = vgm.cfg_group
    if pg is None:
        # `cfg_group` is None when the device mesh hasn't been built yet
        # (`VisualGenMapping.__init__` only builds the mesh when
        # `dist.is_initialized()`). Falling through would default to the
        # world group — silent corruption across the whole world. Fail
        # loud instead so the caller fixes the init order.
        raise RuntimeError(
            "AutoDiffusersPipeline: cfg_size > 1 but VisualGenMapping has no "
            "cfg_group (device mesh not built). Initialize torch.distributed "
            "and re-construct VisualGenMapping before pipeline creation."
        )

    def _gather(t: torch.Tensor) -> torch.Tensor:
        if not isinstance(t, torch.Tensor):
            return t
        gathered = [torch.empty_like(t) for _ in range(P)]
        dist.all_gather(gathered, t.contiguous(), group=pg)
        return torch.cat(gathered, dim=0)

    if isinstance(out, torch.Tensor):
        return _gather(out)
    if isinstance(out, tuple):
        # LTX-2 returns `(noise_pred_video, noise_pred_audio)` — every
        # CFG-doubled head needs the gather. Walk the whole tuple.
        return tuple(_gather(x) if isinstance(x, torch.Tensor) else x for x in out)
    if hasattr(out, "sample"):
        out.sample = _gather(out.sample)
        return out
    return out


def _gather_seq_output(out, vgm):
    """All-gather the image-sequence-sharded output back to full seq.

    Captured Diffusers transformers return either a tensor or a tuple/object
    whose first element is the noise prediction shaped `(B, S_img, ...)`.
    We mirror the input call's `return_dict=False` convention by gathering
    the first tensor and re-packing.
    """
    import torch.distributed as dist

    P = vgm.ulysses_size
    pg = vgm.ulysses_group
    if pg is None:
        raise RuntimeError(
            "AutoDiffusersPipeline: ulysses_size > 1 but VisualGenMapping has "
            "no ulysses_group (device mesh not built). Initialize "
            "torch.distributed and re-construct VisualGenMapping before "
            "pipeline creation."
        )

    def _gather(t: torch.Tensor) -> torch.Tensor:
        if not isinstance(t, torch.Tensor) or t.dim() < 2:
            return t
        gathered = [torch.empty_like(t) for _ in range(P)]
        dist.all_gather(gathered, t.contiguous(), group=pg)
        return torch.cat(gathered, dim=1)

    if isinstance(out, torch.Tensor):
        return _gather(out)
    if isinstance(out, tuple):
        return (_gather(out[0]),) + tuple(out[1:])
    # Diffusers Transformer2DModelOutput-like — has `.sample`
    if hasattr(out, "sample"):
        out.sample = _gather(out.sample)
        return out
    return out


def _resolve_adapter(diffusers_transformer_cls: str) -> "VisGenFamilyAdapter":
    """Look up the family adapter for a Diffusers transformer class.

    Adapters declare which class they target via `diffusers_transformer_cls`.
    Falls back to the generic MM-DiT adapter (S2 best-effort) if no
    family-specific adapter matches.
    """
    from .families import (
        Flux2Adapter,
        FluxAdapter,
        LTX2Adapter,
        LTXAdapter,
        MMDiTAdapter,
        PixArtAdapter,
        SanaAdapter,
        SD3Adapter,
        WanAdapter,
    )

    registry = [
        FluxAdapter(),
        Flux2Adapter(),
        SD3Adapter(),
        WanAdapter(),
        LTXAdapter(),
        LTX2Adapter(),
        PixArtAdapter(),
        SanaAdapter(),
    ]
    for adapter in registry:
        if adapter.diffusers_transformer_cls == diffusers_transformer_cls:
            return adapter
    logger.warning(
        f"No family adapter for {diffusers_transformer_cls!r}; "
        f"falling back to generic MMDiTAdapter (S2 best-effort)."
    )
    return MMDiTAdapter()


# Tested-against Diffusers version range. Bump the upper bound only after
# re-running the family-adapter smoke tests against the new release (see
# `tests/unittest/_torch/visual_gen/test_auto_*`).
_DIFFUSERS_MIN = (0, 39)
_DIFFUSERS_MAX_EXCL = (0, 41)


def _check_diffusers_version() -> None:
    """Warn (loudly) if the runtime Diffusers is outside the auto path's
    tested range. Adapters monkey-patch `transformer.forward` and reach for
    family-specific sub-module names — a Diffusers minor bump that renames
    `patch_embedding` → `patch_embed` etc. silently breaks Ulysses for the
    affected family. The version gate makes the bump an intentional act.

    Set `VISGEN_AUTO_DIFFUSERS_VERSION_CHECK=skip` to suppress the warning
    (e.g. when re-running family smokes against a candidate new release).
    """
    import os

    if os.environ.get("VISGEN_AUTO_DIFFUSERS_VERSION_CHECK", "").lower() == "skip":
        return

    import diffusers

    raw = getattr(diffusers, "__version__", "0.0.0")
    # `0.39.0.dev0` → (0, 39, 0)
    parts = raw.split("+")[0].split(".dev")[0].split(".")
    try:
        major, minor = int(parts[0]), int(parts[1])
    except (ValueError, IndexError):
        logger.warning(
            f"AutoDiffusersPipeline: could not parse diffusers version {raw!r}; "
            "version compatibility check skipped."
        )
        return

    cur = (major, minor)
    if cur < _DIFFUSERS_MIN or cur >= _DIFFUSERS_MAX_EXCL:
        logger.warning(
            f"AutoDiffusersPipeline: diffusers=={raw} is outside the auto "
            f"path's tested range [>= {'.'.join(map(str, _DIFFUSERS_MIN))}, "
            f"< {'.'.join(map(str, _DIFFUSERS_MAX_EXCL))}). Family adapters "
            "(WAN, LTX-2, LTX, SD3, PixArt) reach into specific Diffusers "
            "transformer sub-modules via `pre_capture_patch` monkey-patches — "
            "a renamed attribute or signature change will silently break "
            "Ulysses for the affected family. Re-run "
            "`tests/unittest/_torch/visual_gen/test_auto_*` before relying on "
            "this combination. Set VISGEN_AUTO_DIFFUSERS_VERSION_CHECK=skip "
            "to suppress."
        )


@contextlib.contextmanager
def _family_preload_overrides(checkpoint_dir: str):
    """Context-managed Diffusers class-attribute overrides for the load.

    Some Diffusers transformer classes set `_keep_in_fp32_modules` to keep
    sub-modules (e.g. `condition_embedder` on WAN) in FP32 even when the user
    requests `torch_dtype=BF16`. The auto path's `torch.export` capture trips
    on the FP32↔BF16 boundary inside `patch_embedding`'s Conv3d. The
    handwritten path materializes weights itself and doesn't care.

    Earlier versions of this code mutated the Diffusers class attribute
    permanently (`WanTransformer3DModel._keep_in_fp32_modules = []`), which
    affected any handwritten / parity / ablation code that later instantiated
    the same Diffusers class in the same Python process. Wrap the mutation
    in a context manager and restore the original value on exit so the
    mutation is scoped strictly to the `from_pretrained` call.
    """
    import json
    import os

    overrides: list[tuple[type, str, Any]] = []  # (cls, attr, original_value)

    index_path = os.path.join(checkpoint_dir, "model_index.json")
    if os.path.exists(index_path):
        try:
            with open(index_path) as f:
                class_name = json.load(f).get("_class_name", "")
        except Exception:  # noqa: BLE001 — best-effort
            class_name = ""

        if "Wan" in class_name:
            try:
                from diffusers.models.transformers.transformer_wan import WanTransformer3DModel

                # Snapshot a shallow *copy*, not a reference — Diffusers
                # could mutate the list in-place between save and restore
                # and we'd otherwise re-install the mutated list.
                original = WanTransformer3DModel._keep_in_fp32_modules
                snapshot = list(original) if isinstance(original, list) else original
                overrides.append((WanTransformer3DModel, "_keep_in_fp32_modules", snapshot))
                WanTransformer3DModel._keep_in_fp32_modules = []
            except ImportError:
                pass

    try:
        yield
    finally:
        for cls, attr, original in overrides:
            setattr(cls, attr, original)


class _GraphModuleAsTransformer(nn.Module):
    """Wrap a captured `GraphModule` to expose the Diffusers transformer's
    attribute surface: ``.config``, ``.dtype``, ``.device``, etc.

    The wrapped GraphModule retains all the model parameters (so attribute
    lookups for weights resolve correctly) and forwards `__call__` through
    the captured graph.

    Kwarg filtering: Diffusers pipelines pass kwargs (e.g.,
    ``joint_attention_kwargs`` for SD3) that the captured graph wasn't
    traced with. The wrapper restricts incoming kwargs to the set the
    captured graph accepts (`expected_kwargs`) and discards the rest.
    """

    def __init__(
        self,
        captured_gm: torch.fx.GraphModule,
        original_transformer: nn.Module,
        expected_kwargs: tuple[str, ...] | None = None,
        uses_internal_seq_shard: bool = False,
    ) -> None:
        super().__init__()
        # `_gm` holds the captured graph + its parameters (as attributes on the
        # GraphModule itself). Wrapping in a ModuleList-style attribute keeps
        # PyTorch's module registry happy.
        self._gm = captured_gm
        # Mirror attributes from the original transformer so the Diffusers
        # pipeline's calling code sees a familiar object surface.
        self.config = original_transformer.config
        # Diffusers reads `transformer.dtype` and casts hidden_states to it
        # before calling the transformer. For quantized models the first
        # parameter is FP8 storage; reporting that here would make Diffusers
        # feed FP8 inputs (which torch.randn-style buffers also can't represent)
        # to the captured graph. Use the compute dtype instead.
        from .capture import _infer_module_device_dtype

        _, self._target_dtype = _infer_module_device_dtype(original_transformer)
        # Forward-chunking / gradient-checkpointing attributes the diffusers
        # pipeline may read — keep them False/None.
        self.gradient_checkpointing = False
        # Kwarg names the captured graph was traced with — incoming kwargs
        # are filtered to this set.
        self._expected_kwargs: tuple[str, ...] = expected_kwargs or ()
        # When True, the captured graph already shards the sequence internally
        # (e.g. WAN via pre_capture_patch). Skip the pipeline-boundary shard
        # and gather hooks — the captured model.forward takes care of it.
        self._uses_internal_seq_shard = uses_internal_seq_shard
        # Stash a reference to the original transformer without registering
        # it as a child module — pipelines (e.g., LTX-2) sometimes invoke
        # helpers like `self.transformer.rope.prepare_video_coords(...)` *before*
        # calling forward, so attribute lookups must fall through to it.
        # Storing in `__dict__` avoids nn.Module's auto-registration which
        # would double the parameter count in the wrapper's state_dict.
        object.__setattr__(self, "_original_transformer", original_transformer)

    @property
    def dtype(self) -> torch.dtype:
        return self._target_dtype

    @property
    def device(self) -> torch.device:
        for p in self._gm.parameters():
            return p.device
        return torch.device("cpu")

    def forward(self, *args, **kwargs) -> Any:
        if self._expected_kwargs:
            # Restrict kwargs to the captured-graph signature so torch.export's
            # pre-call check doesn't reject extras like `joint_attention_kwargs`.
            kwargs = {k: kwargs[k] for k in self._expected_kwargs if k in kwargs}

        # Multi-GPU CFG / Ulysses orchestration. CFG goes first (outermost):
        # the captured graph operates on `B_local = B_full / cfg_size`, so we
        # slice the incoming CFG-doubled batch by `cfg_rank` before the
        # Ulysses step sees it. The gathers reverse the order on the way out.
        from . import ops as _ops

        vgm = _ops.get_current_mapping()
        cfg_size = getattr(vgm, "cfg_size", 1) if vgm is not None else 1
        ulysses_size = getattr(vgm, "ulysses_size", 1) if vgm is not None else 1
        do_cfg = cfg_size > 1
        if do_cfg:
            args, kwargs = _split_batch_args_kwargs(args, kwargs, vgm)

        # Adapters that shard the flat sequence inside the captured model
        # (`uses_internal_seq_shard=True`) skip the pipeline-boundary hooks —
        # the captured graph already does slice + all_gather_seq.
        do_boundary_seq = ulysses_size > 1 and not self._uses_internal_seq_shard
        if do_boundary_seq:
            kwargs = _shard_seq_kwargs(kwargs, vgm)

        out = self._gm(*args, **kwargs)

        if do_boundary_seq:
            out = _gather_seq_output(out, vgm)
        if do_cfg:
            out = _gather_batch_output(out, vgm)
        return out

    def cache_context(self, *args, **kwargs):
        """No-op context manager for Diffusers pipelines that use
        `with self.transformer.cache_context(...)` for attention caching.

        The captured graph doesn't participate in Diffusers' attention cache
        infrastructure; replaying the captured graph naively is the auto
        path's caching story (it skips Python-level cache machinery).
        """
        from contextlib import nullcontext

        return nullcontext()

    def __getattr__(self, name: str):
        """Fall back to the wrapped original transformer for attribute
        accesses Diffusers makes on the transformer that aren't on the
        captured `GraphModule` (e.g., random `.config` sub-fields, methods
        like `enable_*`, etc.). nn.Module's `__getattr__` only fires after
        normal attribute lookup misses, so this doesn't shadow `_gm`.
        """
        try:
            return super().__getattr__(name)
        except AttributeError:
            # Look on the wrapped GraphModule first, then fall through to
            # the original transformer (which carries helper submodules like
            # `rope`, `pos_embed`, ... that pipelines may call directly).
            gm = self.__dict__.get("_modules", {}).get("_gm")
            if gm is not None and hasattr(gm, name):
                return getattr(gm, name)
            orig = self.__dict__.get("_original_transformer")
            if orig is not None and hasattr(orig, name):
                return getattr(orig, name)
            raise


class AutoDiffusersPipeline:
    """Minimum-viable end-to-end pipeline using a Diffusers shell + captured transformer."""

    def __init__(
        self,
        checkpoint_dir: str,
        model_config: "DiffusionModelConfig",
        adapter: Optional["VisGenFamilyAdapter"] = None,
        visual_gen_mapping=None,
        enable_torch_compile: bool = False,
        torch_compile_mode: str = "default",
    ) -> None:
        """
        Args:
            checkpoint_dir: HF / Diffusers checkpoint path
            model_config: visual_gen DiffusionModelConfig
            adapter: optional explicit family adapter (default: auto-resolve from transformer class)
            visual_gen_mapping: optional `VisualGenMapping` for multi-GPU. CFG
                parallel splits the CFG-doubled batch across `cfg_group`;
                Ulysses shards the sequence across `ulysses_group`. TP is gated
                behind upstream PR NVIDIA/TensorRT-LLM#13614.
            enable_torch_compile: when True, wraps the captured `GraphModule`'s
                forward with `torch.compile(..., mode=torch_compile_mode)` after
                rewrites. Inductor fuses pointwise ops + generates Triton
                kernels for the captured graph, closing the gap with the
                handwritten path's `TorchCompileConfig` (`pipeline.py:420`).
                First forward pays Inductor codegen cost (1-5 min on a 14-19B
                transformer); subsequent forwards use the cached compiled
                graph.
            torch_compile_mode: passed straight to `torch.compile`. Default
                **`"default"`** — Inductor codegen without CUDA graphs.
                `"reduce-overhead"` enables CUDA graphs but is incompatible
                with Diffusers' per-step output-tensor re-allocation (the
                scheduler returns freshly allocated `latents` each step, and a
                replayed CUDA graph still references the previous step's
                buffers — silent corruption). Only switch to `reduce-overhead`
                if the caller pins the latents buffer across steps.
                `"max-autotune"` tries more kernel candidates (longer warmup).
        """
        from diffusers import DiffusionPipeline

        # Fail-loud if the runtime Diffusers is outside the range the family
        # adapters were tested against. Adapters monkey-patch Diffusers'
        # transformer.forward (5 of 8 families) and reach into specific
        # sub-module attributes (`patch_embedding`, `scale_shift_table`,
        # `norm_out`, ...); a Diffusers minor bump that renames any of these
        # silently breaks Ulysses for the affected family.
        _check_diffusers_version()

        self.checkpoint_dir = checkpoint_dir
        self.model_config = model_config
        self.enable_torch_compile = enable_torch_compile
        self.torch_compile_mode = torch_compile_mode

        # Plumb the mapping into model_config so adapters and capture see it
        # (mirroring `pipeline_loader._setup_visual_gen_mapping`).
        if visual_gen_mapping is not None:
            model_config.visual_gen_mapping = visual_gen_mapping
            # Make the mapping available to `auto/ops.py` (which is called from
            # inside the captured graph at runtime and can't take a tensor arg).
            from . import ops as _ops

            _ops.set_current_mapping(visual_gen_mapping)
        self.visual_gen_mapping = visual_gen_mapping

        # Family-specific pre-load mutations Diffusers needs us to apply before
        # `from_pretrained` builds the transformer (currently: WAN's
        # `_keep_in_fp32_modules = []`). Scoped to this `from_pretrained` call
        # only — restored on context exit so we don't action-at-a-distance
        # any handwritten / parity code that later instantiates the same
        # Diffusers class in the same Python process.
        logger.info(f"AutoDiffusersPipeline: loading Diffusers pipeline from {checkpoint_dir}")
        dtype = model_config.torch_dtype
        with _family_preload_overrides(checkpoint_dir):
            self._pipe = DiffusionPipeline.from_pretrained(checkpoint_dir, torch_dtype=dtype)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if visual_gen_mapping is not None:
            device = torch.device(
                f"cuda:{visual_gen_mapping.local_rank}"
                if hasattr(visual_gen_mapping, "local_rank")
                else f"cuda:{visual_gen_mapping.rank % torch.cuda.device_count()}"
            )
        self._pipe = self._pipe.to(device)
        self._device = device

        # Resolve adapter once — for dual-transformer pipelines (Wan 2.2 has
        # `transformer` for high-noise stage + `transformer_2` for low-noise
        # stage; same class for both), the adapter is shared.
        primary = self._pipe.transformer
        cls_name = type(primary).__name__
        if adapter is None:
            adapter = _resolve_adapter(cls_name)
        self.adapter = adapter
        logger.info(
            f"AutoDiffusersPipeline: transformer={cls_name}, adapter={type(adapter).__name__}"
        )

        # *Forward-compat gate*: visual_gen TP is currently blocked on
        # NVIDIA/TensorRT-LLM#13614. Check once at __init__ before any
        # per-transformer processing; same gate covers both passes.
        tp_size = getattr(visual_gen_mapping, "tp_size", 1) if visual_gen_mapping is not None else 1
        if tp_size > 1:
            from tensorrt_llm._torch.device_mesh import DeviceMeshTopologyImpl

            mesh = DeviceMeshTopologyImpl.device_mesh
            if mesh is not None and "pp" not in mesh.mesh_dim_names:
                raise NotImplementedError(
                    "VisGen-Auto TP is blocked on NVIDIA/TensorRT-LLM#13614. "
                    "Until that PR (which patches `VisualGenMapping.build_mesh` "
                    "to add dummy pp/cp dims) merges, `tp_size > 1` will crash "
                    "inside TRT-LLM `AllReduce` on `mapping.pp_rank`. The "
                    "auto-path TP code (`auto/tp_annotate.py` + the `mapping` "
                    "thread-through in `replace_linear_with_trtllm`) is staged "
                    "and will compose automatically once #13614 lands. "
                    "Use Ulysses (`ulysses_size > 1`) for multi-GPU in the "
                    "meantime — it doesn't go through `AllReduce`."
                )

        tp_mapping = (
            visual_gen_mapping.to_llm_mapping()
            if (visual_gen_mapping is not None and tp_size > 1)
            else None
        )

        # Snapshot user's quant_config.exclude_modules so the sensitivity
        # heuristic (which mutates the list in-place) starts from the same
        # baseline for each transformer pass.
        qc = getattr(model_config, "quant_config", None)
        baseline_excludes: Optional[list] = (
            list(qc.exclude_modules) if (qc is not None and qc.exclude_modules) else None
        )

        # Find all transformer slots on the Diffusers pipeline. The standard
        # convention is `transformer` (single-transformer pipelines: Flux/Flux2/
        # SD3) plus optional `transformer_2` (Wan 2.2 two-stage denoising).
        # Generalizing here means any future Diffusers pipeline that adds a
        # `transformer_3` etc. also lands cleanly.
        transformer_attrs = [
            a for a in ("transformer", "transformer_2") if getattr(self._pipe, a, None) is not None
        ]
        logger.info(
            f"AutoDiffusersPipeline: will capture {len(transformer_attrs)} transformer(s): "
            f"{transformer_attrs}"
        )
        self._original_transformers: dict[str, "nn.Module"] = {}
        for attr in transformer_attrs:
            self._capture_one_transformer(
                attr,
                adapter=adapter,
                model_config=model_config,
                tp_size=tp_size,
                tp_mapping=tp_mapping,
                baseline_excludes=baseline_excludes,
            )
        # Keep one reference for parity comparisons (backwards-compatible
        # shape with the prior single-transformer API).
        self._original_transformer = self._original_transformers.get("transformer")

    def _capture_one_transformer(
        self,
        attr: str,
        *,
        adapter: "VisGenFamilyAdapter",
        model_config: "DiffusionModelConfig",
        tp_size: int,
        tp_mapping,
        baseline_excludes: Optional[list],
    ) -> None:
        """Process one transformer slot (`transformer` or `transformer_2`):
        TP-annotate → quant-replace Linears → torch.export → apply rewrites →
        wrap as `_GraphModuleAsTransformer` → write back to `self._pipe.<attr>`.

        Each pass starts from the user-provided `exclude_modules` baseline so
        the sensitivity heuristic (which mutates `quant_config.exclude_modules`
        in-place) re-derives patterns per transformer independently.
        """
        import torch  # local to keep linter happy in the function scope

        orig = getattr(self._pipe, attr)
        self._original_transformers[attr] = orig
        logger.info(f"AutoDiffusersPipeline: processing `{attr}` ({type(orig).__name__})")

        # Pre-export TP annotation (per-transformer, since each has its own
        # attn.heads attributes and Linear instances).
        if tp_size > 1:
            from .tp_annotate import annotate_tp_roles

            annotate_tp_roles(orig, tp_size)

        # Pre-export Ulysses patch for families that shard inside the model.
        # Image-DiTs default to no-op; video-DiTs (e.g. WAN) monkey-patch the
        # forward to inject `x = x[rank-slice]` after patchify+flatten and
        # `torch.ops.visgen_auto.all_gather_seq(x)` before output proj.
        # The patch reads ulysses_rank/size as Python constants so the
        # captured graph bakes them in.
        if adapter.uses_internal_seq_shard:
            adapter.pre_capture_patch(orig, self.visual_gen_mapping)

        # Reset exclude_modules to the user's baseline so the sensitivity
        # heuristic re-derives patterns from THIS transformer's module tree
        # rather than inheriting whatever was merged from a previous pass.
        qc = getattr(model_config, "quant_config", None)
        if qc is not None:
            qc.exclude_modules = list(baseline_excludes) if baseline_excludes else None

        dtype = model_config.torch_dtype
        need_replace = (qc is not None and qc.quant_algo is not None) or tp_size > 1
        if need_replace:
            if qc is None or qc.quant_algo is None:
                from tensorrt_llm.models.modeling_utils import QuantConfig

                qc = QuantConfig()
                model_config.quant_config = qc

            from .quantize import replace_linear_with_trtllm

            family_excludes = adapter.default_quant_exclude_modules()
            if family_excludes:
                user_excludes = list(qc.exclude_modules or [])
                merged = list(dict.fromkeys(user_excludes + family_excludes))
                if merged != user_excludes:
                    qc.exclude_modules = merged
                    logger.info(
                        f"AutoDiffusersPipeline[{attr}]: merged family exclude_modules "
                        f"{family_excludes} -> {qc.exclude_modules}"
                    )

            n = replace_linear_with_trtllm(orig, qc, dtype, mapping=tp_mapping)
            logger.info(
                f"AutoDiffusersPipeline[{attr}]: pre-export Linear replacement "
                f"(quant={qc.quant_algo}, tp={tp_size}): {n} Linears wrapped"
            )

        # Capture + rewrite
        from .capture import _infer_module_device_dtype, capture_transformer
        from .rewrite import apply_rewrites

        logger.info(f"AutoDiffusersPipeline[{attr}]: capturing via torch.export ...")
        cfg_size = (
            getattr(self.visual_gen_mapping, "cfg_size", 1)
            if self.visual_gen_mapping is not None
            else 1
        )
        ep = capture_transformer(orig, adapter, model_config, cfg_size=cfg_size)
        gm = ep.module()
        policy = adapter.rewrite_policy(model_config)
        # `adapter` is passed so its `customize_passes(pm)` hook can splice
        # in family-specific passes (insert_after / replace / etc.) before
        # the default pipeline runs. Default hook is a no-op.
        apply_rewrites(gm, policy, adapter=adapter)
        logger.info(f"AutoDiffusersPipeline[{attr}]: rewrites applied")

        # Discover the kwarg set the captured graph was traced with so the
        # wrapper can filter out-of-band kwargs from the Diffusers pipeline.
        device_, dtype_ = _infer_module_device_dtype(orig)
        _, sample_kwargs = adapter.example_inputs(model_config, device_, dtype_)
        expected_kwargs = tuple(sample_kwargs.keys())

        wrapped = _GraphModuleAsTransformer(
            gm,
            orig,
            expected_kwargs=expected_kwargs,
            uses_internal_seq_shard=adapter.uses_internal_seq_shard,
        )
        if self.enable_torch_compile:
            # Compile the captured GraphModule (not the wrapper). The wrapper's
            # `forward` does pre/post sequence shard/gather + kwarg filtering
            # in Python; we want those to stay eager so the compiled boundary
            # only covers the dense transformer math. Inductor handles
            # GraphModules natively + our `visgen_auto::*` custom ops have
            # `register_fake` shims (see `auto/ops.py`) so the codegen path
            # has the meta info it needs.
            logger.info(
                f"AutoDiffusersPipeline[{attr}]: wrapping with "
                f"torch.compile(mode={self.torch_compile_mode!r}) "
                f"— first forward will pay Inductor codegen cost"
            )
            wrapped._gm = torch.compile(wrapped._gm, mode=self.torch_compile_mode)
        setattr(self._pipe, attr, wrapped)

    @property
    def diffusers_pipe(self):
        return self._pipe

    @property
    def captured_transformer(self) -> _GraphModuleAsTransformer:
        return self._pipe.transformer

    @property
    def device(self) -> torch.device:
        # The Diffusers pipeline is `.to(device)`'d in `__init__`; expose the
        # selected device so callers (e.g. `AutoTransformerPipeline.infer`)
        # don't reach into the underscored `_device` field.
        return self._device

    def __call__(self, *args, **kwargs):
        return self._pipe(*args, **kwargs)

    def warmup(self, **pipeline_kwargs) -> None:
        """Trigger torch.compile codegen by running a short denoise pass.

        When `enable_torch_compile=True`, the captured GraphModule is
        wrapped with `torch.compile` but compilation is lazy — codegen
        fires on the first call. For long-denoise pipelines (50-step
        Wan, 40-step LTX-2) this codegen lands *inside* the measured
        denoise window, eating most of the steady-state perf win on
        single-run benchmarks. The handwritten path solves this with
        `PipelineLoader(...).load(skip_warmup=False)` which calls a
        `_run_warmup(...)` hook on each pipeline class. We mirror that
        here without taking a dependency on the handwritten
        `PipelineLoader`: the caller passes the same kwargs they'd hand
        to `__call__`, with a small `num_inference_steps` (default 4 —
        enough to cross Wan 2.2's `boundary_ratio=0.875` so both
        transformer + transformer_2 codegen fires; single-tx pipelines
        only need ≥1).

        No-op when `enable_torch_compile=False`.

        Args:
            **pipeline_kwargs: forwarded to `self._pipe(...)`. Override
                `num_inference_steps` for finer control, otherwise 4 is
                the safe default for dual-tx pipelines.
        """
        if not self.enable_torch_compile:
            return
        pipeline_kwargs.setdefault("num_inference_steps", 4)
        # Use a no-op prompt; the compile cost is shape-driven, not
        # prompt-driven. Match the handwritten Wan warmup conventions.
        pipeline_kwargs.setdefault("prompt", "warmup")
        # Note: don't default `negative_prompt` — Flux/Flux2 are
        # guidance-distilled and reject it. Caller passes it when the
        # diffusers pipeline accepts it (SD3, WAN, LTX-2).
        # Suppress the per-step tqdm bar during warmup to keep logs clean.
        # `set_progress_bar_config` is a no-op on some Diffusers pipelines
        # that don't define it (e.g. older custom community pipelines);
        # an `AttributeError` from a missing method is the only expected
        # failure here.
        try:
            self._pipe.set_progress_bar_config(disable=True)
        except AttributeError:
            pass
        with torch.no_grad():
            self._pipe(**pipeline_kwargs)
        try:
            self._pipe.set_progress_bar_config(disable=False)
        except AttributeError:
            pass
