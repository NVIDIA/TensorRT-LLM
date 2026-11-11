"""DiffVAE decode presets and resolved install recipes (pure data — no module I/O)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

# Default eager DiffVAE pin: lower VRAM than hopper-fna TokPerm.
_CUTLASS_FNA_BACKEND = "cutlass-fna"

# Fixed W split for Chunked* modes (matrix sweet spot at ~1088x1920).
_CHUNKED_W_CHUNKS = 4


class NAttentionKind(Enum):
    """Which neighborhood-attention backend ``apply`` installs on NA modules."""

    NATTEN = "natten"
    BLACKWELL_DSL = "blackwell_dsl"
    TRITON = "triton"
    EAGER_SDPA = "eager_sdpa"


class DiffVAEBlockKind(Enum):
    """Which DiffusionNABlock subclass ``apply`` installs via ``__class__`` swap."""

    COMBINED = "combined"
    CHUNKED = "chunked"
    BLACKWELL_DSL = "blackwell_dsl"


@dataclass(frozen=True, slots=True)
class DiffVAEConfig:
    """Resolved install recipe — facts ``apply`` reads, not runtime forward knobs.
    ``block`` selects the pathway (chunked vs combined vs Blackwell DSL); class swap
    implies deferred inject + W-chunked attn + modulating MLP, combined context +
    full-volume attn + residual MLP, or deferred stage-4 + fused CuTe DSL block.
    """

    block: DiffVAEBlockKind
    w_chunks: int
    natten_backend: str | None
    attention: NAttentionKind
    compile_blocks: bool
    compile_det_stages: bool


class DiffVAEMode(Enum):
    """User-facing DiffVAE decode presets.
    Relative performance (order-of-magnitude; hardware varies — no absolute
    timings or VRAM figures):
    - **Compile:** ``chunked_compile`` roughly ~2x faster to compile than
      ``combined_compile`` (det stages not compiled).
    - **Warm runtime:** relative to ``combined_compile`` (fastest):
      ``chunked_compile`` roughly ~1.4x slower, ``chunked_eager`` roughly
      ~2-2.5x slower.
    - **Peak VRAM:** ``chunked_*`` roughly ~½ of ``combined_compile``.
    - **BLACKWELL_DSL:** datacenter Blackwell CuTe DSL NA + fused stage-5
      (deferred stage-4 inputs; upsample+context_proj inside the fused kernel).
    CHUNKED_* always use deferred stage-4 + w_chunks=4. COMBINED_COMPILE is
    combined context + w_chunks=1 (best runtime, highest memory). Every mode falls
    back to Triton/eager when natten is missing; only the chunked ones also drop
    compilation there (see ``apply.resolve_attention_for_host``).
    A keyframe decode is never compiled -- the compiled entry points are the
    keyframe-free ones -- so its memory budget follows the block shape rather than
    the mode's name. ``diffusion_tiling`` owns that table.
    """

    CHUNKED_EAGER = "chunked_eager"
    CHUNKED_COMPILE = "chunked_compile"
    COMBINED_COMPILE = "combined_compile"
    BLACKWELL_DSL = "blackwell_dsl"

    def resolve(self) -> DiffVAEConfig:
        """Expand this preset into a concrete install recipe.
        This is the only place preset → recipe expansion lives.
        """
        if self is DiffVAEMode.CHUNKED_EAGER:
            return DiffVAEConfig(
                block=DiffVAEBlockKind.CHUNKED,
                w_chunks=_CHUNKED_W_CHUNKS,
                natten_backend=_CUTLASS_FNA_BACKEND,
                attention=NAttentionKind.NATTEN,
                compile_blocks=False,
                compile_det_stages=False,
            )
        if self is DiffVAEMode.CHUNKED_COMPILE:
            return DiffVAEConfig(
                block=DiffVAEBlockKind.CHUNKED,
                w_chunks=_CHUNKED_W_CHUNKS,
                natten_backend=None,
                attention=NAttentionKind.NATTEN,
                compile_blocks=True,
                compile_det_stages=False,
            )
        if self is DiffVAEMode.COMBINED_COMPILE:
            return DiffVAEConfig(
                block=DiffVAEBlockKind.COMBINED,
                w_chunks=1,
                natten_backend=None,
                attention=NAttentionKind.NATTEN,
                compile_blocks=True,
                compile_det_stages=True,
            )
        if self is DiffVAEMode.BLACKWELL_DSL:
            return DiffVAEConfig(
                block=DiffVAEBlockKind.BLACKWELL_DSL,
                w_chunks=1,
                natten_backend=None,
                attention=NAttentionKind.BLACKWELL_DSL,
                compile_blocks=True,
                compile_det_stages=True,
            )
        raise ValueError(f"Unknown DiffVAEMode: {self!r}")
