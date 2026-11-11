import functools
import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from tensorrt_llm._torch.visual_gen.models.ltx_core.model.transformer.rope import LTXRopeType, apply_rotary_emb


def _torch_default_sdpa_priority() -> list[SDPBackend]:
    """Fetch torch's current default SDPA priority order at runtime.
    Used as the default for ``PytorchAttention`` so the wrapper-always
    code path matches torch's native dispatch order without hard-coding it
    (which would drift if torch updates the default).
    ``torch._C._get_sdp_priority_order`` is a private API; we accept that
    risk because the project pins ``torch`` in the lockfile, so any
    rename/removal surfaces on a controlled torch bump rather than silently.
    """
    return [SDPBackend(p) for p in torch._C._get_sdp_priority_order()]


flash_attn_interface = None
flash_attn_4_func = None
try:
    import flash_attn_interface
except ImportError:
    flash_attn_interface = None
try:
    # FA4's cute path depends on a matching nvidia-cutlass-dsl; a newer cutlass
    # can raise AttributeError (e.g. missing cute.core.ThrMma) rather than
    # ImportError — treat any probe failure as "FA4 unavailable".
    from flash_attn.cute import flash_attn_func as flash_attn_4_func
except Exception:
    flash_attn_4_func = None
try:
    # macOS only: routes SDPA to Apple's prebuilt MPSGraph attention kernel.
    from mps_sdpa import sdpa_opt as _mps_sdpa_opt
except ImportError:
    _mps_sdpa_opt = None


class AttentionCallable(Protocol):
    """Unmasked attention. Backends without a mask kernel (FA3/FA4) implement only
    this protocol; backends that support masks too (Pytorch/SDPA) are
    structurally usable here and as :class:`MaskedAttentionCallable`."""

    def __call__(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, heads: int) -> torch.Tensor: ...


class MaskedAttentionCallable(Protocol):
    """Masked attention. Mask is required (not optional) -- the caller has already
    decided this is the masked path and chosen a backend that can serve it. Used
    by :class:`Attention` when its forward receives a non-None ``mask``."""

    def __call__(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, heads: int, mask: torch.Tensor
    ) -> torch.Tensor: ...


class PytorchAttention(AttentionCallable):
    def __init__(self, priority: list[SDPBackend] | None = None) -> None:
        # priority=None -> snapshot torch's default SDPA priority at construction.
        # Always passed through ``sdpa_kernel(..., set_priority=True)`` so the
        # call site is uniform regardless of how the priority was chosen.
        self._priority = priority if priority is not None else _torch_default_sdpa_priority()

    @property
    def label(self) -> str:
        """Human-readable identifier. Encodes the SDPA priority list so a
        single-backend pin reads differently from the full-priority dispatcher walk."""
        return f"SDPA[{'>'.join(b.name for b in self._priority)}]"

    def __call__(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, heads: int, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        b, _, dim_head = q.shape
        dim_head //= heads
        q, k, v = (t.view(b, -1, heads, dim_head).transpose(1, 2) for t in (q, k, v))

        if mask is not None:
            # add a batch dimension if there isn't already one
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
            # add a heads dimension if there isn't already one
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)

        with sdpa_kernel(self._priority, set_priority=True):
            out = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False
            )
        out = out.transpose(1, 2).reshape(b, -1, heads * dim_head)
        return out


class MPSSdpaAttention(AttentionCallable):
    """Apple-fused scaled-dot-product attention on MPS.
    Routes to ``mps_sdpa.sdpa_opt``, which calls Apple's prebuilt
    ``MPSGraph.scaledDotProductAttention`` kernel (via a zero-copy bridge)
    instead of torch's ``sdpa_general_mps`` graph. The Apple kernel does not
    materialize the ``[B, H, Nq, Nk]`` score matrix, so it avoids the
    long-sequence memory wall that makes torch's materializing MPS SDPA
    unusable on video latents (~32x faster at a 14k-token latent on an M4 Pro).
    It is a hard dependency on Apple Silicon (the ``mps-sdpa`` platform-marked
    requirement), so AUTOMATIC always has it on MPS. Unlike a JIT-compiled Metal
    flash kernel it needs no runtime shader compilation, so it is robust
    across macOS / Metal revisions.
    Accepts an optional additive-float or boolean ``mask`` broadcastable to
    ``[B, H, Nq, Nk]``, so it serves both the unmasked and masked protocols.
    """

    @property
    def label(self) -> str:
        return "MPS-SDPA"

    def __call__(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, heads: int, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if _mps_sdpa_opt is None:
            raise RuntimeError("MPSSdpaAttention was selected but `mps-sdpa` is not installed.")
        if q.device.type != "mps":
            raise RuntimeError("MPSSdpaAttention requires MPS. Use PyTorch SDPA on CPU or CUDA.")

        b, _, dim_head = q.shape
        dim_head //= heads
        q, k, v = (t.view(b, -1, heads, dim_head).transpose(1, 2) for t in (q, k, v))

        if mask is not None:
            # add a batch dimension if there isn't already one
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
            # add a heads dimension if there isn't already one
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)

        out = _mps_sdpa_opt(q, k, v, attn_mask=mask)
        out = out.transpose(1, 2).reshape(b, -1, heads * dim_head)
        return out


class FlashAttention3(AttentionCallable):
    label = "FlashAttention3"

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        heads: int,
    ) -> torch.Tensor:
        if flash_attn_interface is None:
            raise RuntimeError("FlashAttention3 was selected but `FlashAttention3` is not installed.")
        if q.device.type != "cuda":
            raise RuntimeError("FlashAttention3 requires CUDA. Use PyTorch SDPA on CPU or MPS.")

        b, _, dim_head = q.shape
        dim_head //= heads

        q, k, v = (t.view(b, -1, heads, dim_head) for t in (q, k, v))

        out = flash_attn_interface.flash_attn_func(q.to(v.dtype), k.to(v.dtype), v)
        out = out.reshape(b, -1, heads * dim_head)
        return out


def _flash_attention_4_output(result: torch.Tensor | tuple[torch.Tensor, ...]) -> torch.Tensor:
    """Normalize FA4 return value: tensor (current releases) or legacy (out, aux) tuple."""
    if isinstance(result, tuple):
        return result[0]
    return result


class FlashAttention4(AttentionCallable):
    label = "FlashAttention4"

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        heads: int,
    ) -> torch.Tensor:
        if flash_attn_4_func is None:
            raise RuntimeError("FlashAttention4 was selected but `flash-attn-4` is not installed.")
        if q.device.type != "cuda":
            raise RuntimeError("FlashAttention4 requires CUDA. Use PyTorch SDPA on CPU or MPS.")

        b, _, dim_head = q.shape
        dim_head //= heads

        q, k, v = (t.view(b, -1, heads, dim_head) for t in (q, k, v))

        result = flash_attn_4_func(q.to(v.dtype), k.to(v.dtype), v)
        out = _flash_attention_4_output(result)
        out = out.reshape(b, -1, heads * dim_head)
        return out


# --- Automatic selection -----------------------------------------------------
# AUTOMATIC inspects installed extras and the GPU arch and returns the fastest
# usable callable for each path. The selection runs once per process (cached).
# The unmasked and masked picks are independent: each calls its own helper and
# may end up on different backends (e.g. FA3 unmasked + SDPA masked on H100).


def _sdpa_can_use(backend: SDPBackend, *, with_mask: bool) -> bool:
    """Ask torch whether *backend* can run with the given mask shape.
    ``MATH`` is the universal SDPA fallback (pure PyTorch ops, no kernel
    requirements) so it returns True everywhere, CPU included. The other
    backends use ``torch.backends.cuda.can_use_*`` capability checks (no GPU
    compute, no synchronization) and are False without CUDA. The probe shapes
    are small but realistic enough to surface constraints (head dim, dtype)
    that the per-backend rules care about.
    """
    if backend is SDPBackend.MATH:
        return True
    if not torch.cuda.is_available():
        return False
    q = torch.empty(1, 4, 128, 64, device="cuda", dtype=torch.bfloat16)
    k = torch.empty(1, 4, 128, 64, device="cuda", dtype=torch.bfloat16)
    v = torch.empty(1, 4, 128, 64, device="cuda", dtype=torch.bfloat16)
    mask = torch.zeros(1, 4, 128, 128, device="cuda", dtype=torch.bfloat16) if with_mask else None
    params = torch.backends.cuda.SDPAParams(q, k, v, mask, 0.0, False, False)
    if backend is SDPBackend.CUDNN_ATTENTION:
        return torch.backends.cuda.can_use_cudnn_attention(params, debug=False)
    if backend is SDPBackend.FLASH_ATTENTION:
        return torch.backends.cuda.can_use_flash_attention(params, debug=False)
    if backend is SDPBackend.EFFICIENT_ATTENTION:
        return torch.backends.cuda.can_use_efficient_attention(params, debug=False)
    return False


_SDPA_FULL_PRIORITY: tuple[SDPBackend, ...] = (
    SDPBackend.CUDNN_ATTENTION,
    SDPBackend.FLASH_ATTENTION,
    SDPBackend.EFFICIENT_ATTENTION,
    SDPBackend.MATH,
)


def _on_macos() -> bool:
    """True on macOS, where torch's native SDPA materializes the score matrix and
    AUTOMATIC routes to Apple's fused ``mps-sdpa`` kernel instead."""
    return sys.platform == "darwin"


def _mps_sdpa_available() -> bool:
    """True when the ``mps-sdpa`` package is importable. It is a platform-marked
    hard dependency on Apple Silicon, so this is always True there; it is only
    False on non-Apple-Silicon macs (e.g. Intel/CPU), where AUTOMATIC falls back
    to torch's SDPA (acceptable on CPU, which has no MPS memory wall)."""
    return _mps_sdpa_opt is not None


def _sdpa_full_priority() -> PytorchAttention:
    """Hand SDPA the full backend priority order; let torch's dispatcher pick at call time.
    ``sdpa_kernel(_SDPA_FULL_PRIORITY, set_priority=True)`` enables all four
    backends and orders them; torch then walks the order at call time and picks
    the first backend whose ``can_use_*`` check passes for the actual
    shapes/dtype/mask. FLASH is rejected automatically when a mask is present;
    CUDNN may be rejected under deterministic mode; MATH is the universal
    fallback. Probing per-backend usability up front from generic probe shapes
    cannot anticipate the variety of real call sites (e.g. broadcast key-only
    masks, large head dim), so we defer the choice to the dispatcher.
    """
    return PytorchAttention(priority=list(_SDPA_FULL_PRIORITY))


def _select_primary_attention() -> AttentionCallable:
    """Pick the fastest unmasked attention based on installed extras and GPU arch.
    Priority by arch:
    - Hopper (sm_90, H100): FA3 > FA4 > SDPA.
    - Datacenter Blackwell (sm_100, B200): FA4 > SDPA. FA4 is intentionally *not*
      picked on consumer Blackwell (sm_120) -- known regressions in newer
      FA4 betas; users who want it on sm_120 must opt in explicitly.
    - macOS (Apple Silicon / MPS): Apple's fused MPSGraph kernel via ``mps-sdpa``
      (a platform-marked hard dependency on Apple Silicon) -- it avoids the
      full-score-matrix memory wall on long video sequences. On a non-Apple-Silicon
      mac (Intel/CPU) it falls back to torch's SDPA.
    - Everywhere else (Ada, Ampere, CPU): SDPA with the full backend priority
      list -- torch's runtime dispatcher picks the best fit at call time.
    """
    if torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability(0)
        if major == 9:
            if flash_attn_interface is not None:
                return FlashAttention3()
            if flash_attn_4_func is not None:
                return FlashAttention4()
        if major == 10 and flash_attn_4_func is not None:
            return FlashAttention4()
    if _on_macos():
        return MPSSdpaAttention() if _mps_sdpa_available() else _sdpa_full_priority()
    return _sdpa_full_priority()


def _select_masked_attention() -> MaskedAttentionCallable:
    """Pick a mask-aware attention. On macOS, Apple's fused MPSGraph kernel via
    ``mps-sdpa`` (a hard dependency on Apple Silicon, else torch's SDPA on
    Intel/CPU macs); else SDPA with the full priority list (the dispatcher
    rejects FLASH automatically when a mask is present and walks past it --
    torch SDPA handles the additive mask directly)."""
    if _on_macos():
        return MPSSdpaAttention() if _mps_sdpa_available() else _sdpa_full_priority()
    return _sdpa_full_priority()


@functools.cache
def automatic_attention() -> AttentionCallable:
    """Cached AUTOMATIC pick for the unmasked path.
    Cached so every ``AttentionOps`` in the process shares one instance."""
    return _select_primary_attention()


@functools.cache
def automatic_masked_attention() -> MaskedAttentionCallable:
    """Cached AUTOMATIC pick for the masked path. See :func:`automatic_attention`."""
    return _select_masked_attention()


def attention_label(fn: AttentionCallable | MaskedAttentionCallable) -> str:
    """Best-effort human-readable backend name.
    Built-in callables expose ``.label`` (encoding the SDPA priority list for the
    Pytorch backends); fall back to the class name for custom or wrapped callables
    (e.g. the multi-GPU All2All wrappers) that don't define one."""
    return getattr(fn, "label", type(fn).__name__)


def _resolve_sdpa_variant(backend: SDPBackend, name: str, *, with_mask: bool) -> PytorchAttention:
    """Build a single-backend ``PytorchAttention`` pin, raising if the backend
    can't actually serve the call on this machine. Used by both
    :meth:`AttentionFunction.to_callable` and :meth:`MaskedAttentionFunction.to_callable`;
    ``with_mask`` differs between the two so the capability check considers
    the protocol the caller intends to use. Not used for ``MATH`` -- MATH is
    the universal fallback and would falsely fail the CUDA-only probe on CPU.
    """
    if not _sdpa_can_use(backend, with_mask=with_mask):
        raise RuntimeError(
            f"{name} selected but the SDPA {backend.name} backend is not usable on this machine "
            "(either no CUDA, the backend rejected the probe shapes, or "
            "torch.use_deterministic_algorithms(True) excluded it)."
        )
    return PytorchAttention(priority=[backend])


class AttentionFunction(Enum):
    PYTORCH = "pytorch"
    FLASH_ATTENTION_3 = "flash_attention_3"
    FLASH_ATTENTION_4 = "flash_attention_4"
    SDPA_CUDNN = "sdpa_cudnn"
    SDPA_FLASH = "sdpa_flash"
    SDPA_EFFICIENT = "sdpa_efficient"
    SDPA_MATH = "sdpa_math"
    # Apple's fused MPSGraph SDPA via the `mps-sdpa` package (macOS/MPS only, a
    # platform-marked hard dependency on Apple Silicon). The AUTOMATIC default on
    # MPS; never materializes the score matrix.
    MPS_SDPA = "mps_sdpa"
    # Pick the fastest unmasked backend for the current GPU/extras combo; see
    # :func:`automatic_attention`. Default for :class:`AttentionOps`.
    AUTOMATIC = "automatic"

    def to_callable(self) -> AttentionCallable:  # noqa: PLR0911, PLR0912
        """Resolve to a concrete callable. Use this at module init time so that
        torch.compile can trace through the attention call without graph breaks.
        Every non-AUTOMATIC variant raises :class:`RuntimeError` when the backend
        isn't usable on this machine -- missing package or SDPA backend rejected
        on this hardware (e.g. cuDNN under ``torch.use_deterministic_algorithms``).
        Opting in means "this kernel or fail loudly", with one exception: under the
        ``ltx_kernels_inductor`` and ``ltx_kernels_eager`` compile backends, an unmasked
        call from ``PYTORCH`` or an ``SDPA_*`` variant runs ``ltx_kernels::sdpa_select``,
        which picks cuDNN or flash attention per shape regardless of the pinned backend.
        ``AUTOMATIC`` returns the cached :func:`automatic_attention` instance so every
        build shares one callable.
        """
        match self:
            case AttentionFunction.AUTOMATIC:
                return automatic_attention()
            case AttentionFunction.PYTORCH:
                return PytorchAttention()
            case AttentionFunction.FLASH_ATTENTION_3:
                if flash_attn_interface is None:
                    raise RuntimeError(
                        "AttentionFunction.FLASH_ATTENTION_3 selected but `flash-attn-3` is not installed."
                    )
                if not torch.cuda.is_available():
                    raise RuntimeError(
                        "AttentionFunction.FLASH_ATTENTION_3 requires CUDA. Use PyTorch SDPA on CPU or MPS."
                    )
                return FlashAttention3()
            case AttentionFunction.FLASH_ATTENTION_4:
                if flash_attn_4_func is None:
                    raise RuntimeError(
                        "AttentionFunction.FLASH_ATTENTION_4 selected but `flash-attn-4` is not installed."
                    )
                if not torch.cuda.is_available():
                    raise RuntimeError(
                        "AttentionFunction.FLASH_ATTENTION_4 requires CUDA. Use PyTorch SDPA on CPU or MPS."
                    )
                return FlashAttention4()
            case AttentionFunction.SDPA_MATH:
                return PytorchAttention(priority=[SDPBackend.MATH])
            case AttentionFunction.MPS_SDPA:
                if _mps_sdpa_opt is None:
                    raise RuntimeError("AttentionFunction.MPS_SDPA selected but `mps-sdpa` is not installed.")
                return MPSSdpaAttention()
            case AttentionFunction.SDPA_CUDNN:
                return _resolve_sdpa_variant(
                    SDPBackend.CUDNN_ATTENTION, "AttentionFunction.SDPA_CUDNN", with_mask=False
                )
            case AttentionFunction.SDPA_FLASH:
                return _resolve_sdpa_variant(
                    SDPBackend.FLASH_ATTENTION, "AttentionFunction.SDPA_FLASH", with_mask=False
                )
            case AttentionFunction.SDPA_EFFICIENT:
                return _resolve_sdpa_variant(
                    SDPBackend.EFFICIENT_ATTENTION, "AttentionFunction.SDPA_EFFICIENT", with_mask=False
                )


class MaskedAttentionFunction(Enum):
    """Backends usable on the masked path. Mirrors :class:`AttentionFunction` minus
    the variants the torch SDPA dispatcher (or the wrapped kernel) rejects with a
    mask: ``SDPA_FLASH`` -- FLASH kernel cannot serve an additive ``attn_mask``;
    ``FLASH_ATTENTION_3``/``FLASH_ATTENTION_4`` -- neither has a mask kernel at all.
    Keeping them out makes "this backend cannot mask" a type error, not a runtime one."""

    PYTORCH = "pytorch"
    SDPA_CUDNN = "sdpa_cudnn"
    SDPA_EFFICIENT = "sdpa_efficient"
    SDPA_MATH = "sdpa_math"
    # Apple's fused MPSGraph SDPA via the `mps-sdpa` package (macOS/MPS only, a
    # platform-marked hard dependency on Apple Silicon); the AUTOMATIC default on
    # MPS. Mask-aware.
    MPS_SDPA = "mps_sdpa"
    # Pick the fastest mask-capable backend for the current extras combo; see
    # :func:`automatic_masked_attention`. Default for the masked slot of
    # :class:`AttentionOps`.
    AUTOMATIC = "automatic"

    def to_callable(self) -> MaskedAttentionCallable:
        """Resolve to a concrete masked callable. Same backend classes as
        :meth:`AttentionFunction.to_callable`; the protocol returned just exposes
        the masked call signature.
        Non-AUTOMATIC variants raise :class:`RuntimeError` when the backend isn't
        usable for the masked path on this machine. SDPA probes run with
        ``with_mask=True`` so the capability check considers the protocol the
        caller will actually use."""
        match self:
            case MaskedAttentionFunction.AUTOMATIC:
                return automatic_masked_attention()
            case MaskedAttentionFunction.PYTORCH:
                return PytorchAttention()
            case MaskedAttentionFunction.SDPA_MATH:
                return PytorchAttention(priority=[SDPBackend.MATH])
            case MaskedAttentionFunction.MPS_SDPA:
                if _mps_sdpa_opt is None:
                    raise RuntimeError("MaskedAttentionFunction.MPS_SDPA selected but `mps-sdpa` is not installed.")
                return MPSSdpaAttention()
            case MaskedAttentionFunction.SDPA_CUDNN:
                return _resolve_sdpa_variant(
                    SDPBackend.CUDNN_ATTENTION, "MaskedAttentionFunction.SDPA_CUDNN", with_mask=True
                )
            case MaskedAttentionFunction.SDPA_EFFICIENT:
                return _resolve_sdpa_variant(
                    SDPBackend.EFFICIENT_ATTENTION, "MaskedAttentionFunction.SDPA_EFFICIENT", with_mask=True
                )


@dataclass(frozen=True)
class AttentionOps:
    """Pluggable callables consumed by :class:`Attention`."""

    attention_function: AttentionCallable = field(default_factory=lambda: AttentionFunction.AUTOMATIC.to_callable())
    masked_attention_function: MaskedAttentionCallable = field(
        default_factory=lambda: MaskedAttentionFunction.AUTOMATIC.to_callable()
    )


class Attention(torch.nn.Module):
    def __init__(
        self,
        query_dim: int,
        context_dim: int | None = None,
        heads: int = 8,
        dim_head: int = 64,
        norm_eps: float = 1e-6,
        rope_type: LTXRopeType = LTXRopeType.SPLIT,
        ops: AttentionOps | None = None,
        apply_gated_attention: bool = False,
    ) -> None:
        super().__init__()
        if ops is None:
            ops = AttentionOps()
        self.rope_type = rope_type
        self.attention_function = ops.attention_function
        self.masked_attention_function = ops.masked_attention_function

        inner_dim = dim_head * heads
        context_dim = query_dim if context_dim is None else context_dim

        self.heads = heads
        self.dim_head = dim_head

        self.q_norm = torch.nn.RMSNorm(inner_dim, eps=norm_eps)
        self.k_norm = torch.nn.RMSNorm(inner_dim, eps=norm_eps)

        self.to_q = torch.nn.Linear(query_dim, inner_dim, bias=True)
        self.to_k = torch.nn.Linear(context_dim, inner_dim, bias=True)
        self.to_v = torch.nn.Linear(context_dim, inner_dim, bias=True)

        # Optional per-head gating
        if apply_gated_attention:
            self.to_gate_logits = torch.nn.Linear(query_dim, heads, bias=True)
        else:
            self.to_gate_logits = None

        self.to_out = torch.nn.Sequential(torch.nn.Linear(inner_dim, query_dim, bias=True), torch.nn.Identity())

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        pe: torch.Tensor | None = None,
        k_pe: torch.Tensor | None = None,
        perturbation_mask: torch.Tensor | None = None,
        all_perturbed: bool = False,
    ) -> torch.Tensor:
        """Multi-head attention with optional RoPE, perturbation masking, and per-head gating.
        When ``perturbation_mask`` is all zeros, the expensive query/key path
        (linear projections, RMSNorm, RoPE) is skipped entirely and only the
        value projection is used as a pass-through.
        Args:
            x: Query input tensor of shape ``(B, T, query_dim)``.
            context: Key/value context tensor of shape ``(B, S, context_dim)``.
                Falls back to ``x`` (self-attention) when *None*.
            mask: Optional attention mask. Interpretation depends on the attention
                backend (additive bias for PyTorch SDPA). A non-None
                ``mask`` routes to ``masked_attention_function``; ``None`` keeps
                the unmasked path.
            pe: Rotary positional embeddings applied to both ``q`` and ``k``.
            k_pe: Separate rotary positional embeddings for ``k`` only. When
                *None*, ``pe`` is reused for keys.
            perturbation_mask: Optional mask in ``[0, 1]`` that
                blends the attention output with the raw value projection:
                ``out = attn_out * mask + v * (1 - mask)``.
                **1** keeps the full attention output, **0** bypasses attention
                and passes the value projection through unchanged.
                *None* or all-ones means standard attention; all-zeros skips
                the query/key path entirely for efficiency.
            all_perturbed: Whether all perturbations are active for this block.
        Returns:
            Output tensor of shape ``(B, T, query_dim)``.
        """
        context = x if context is None else context
        use_attention = not all_perturbed

        v = self.to_v(context)

        if not use_attention:
            out = v
        else:
            q = self.to_q(x)
            k = self.to_k(context)
            q = self.q_norm(q)
            k = self.k_norm(k)
            if pe is not None:
                q = apply_rotary_emb(q, pe, self.rope_type)
                k = apply_rotary_emb(k, pe if k_pe is None else k_pe, self.rope_type)
            if mask is None:
                out = self.attention_function(q, k, v, self.heads)  # (B, T, H*D)
            else:
                out = self.masked_attention_function(q, k, v, self.heads, mask)

        # Apply per-head gating if enabled
        if self.to_gate_logits is not None and use_attention:
            gate_logits = self.to_gate_logits(x)
            b, t, _ = out.shape
            out = out.view(b, t, self.heads, self.dim_head)
            gates = 2.0 * torch.sigmoid(gate_logits)
            out = out * gates.unsqueeze(-1)
            out = out.view(b, t, self.heads * self.dim_head)

        if use_attention and perturbation_mask is not None:
            out = out * perturbation_mask + v * (1 - perturbation_mask)

        return self.to_out(out)
