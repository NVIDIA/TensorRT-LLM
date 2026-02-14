import inspect
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
from diffusers.models.modeling_outputs import Transformer2DModelOutput

from tensorrt_llm.logger import logger

# =============================================================================
# Core Data Structures
# =============================================================================


@dataclass
class CacheContext:
    """Context returned by model extractors for TeaCache.

    Attributes:
        modulated_input: Timestep embedding used for cache distance calculation
        hidden_states: Input hidden states for the transformer
        encoder_hidden_states: Text/prompt embeddings
        run_transformer_blocks: Callable that executes the transformer forward pass
        postprocess: Callable that formats the output to the expected return type
    """

    modulated_input: torch.Tensor
    hidden_states: torch.Tensor
    encoder_hidden_states: Any = None
    run_transformer_blocks: Callable = None
    postprocess: Callable = None


# =============================================================================
# Extractor Registry
# =============================================================================

_EXTRACTORS = {}


def register_extractor(model_name, extractor_fn):
    """Register an extractor function for a model class."""
    _EXTRACTORS[model_name] = extractor_fn


def get_extractor(model_type):
    """Get the registered extractor for a model type."""
    if model_type not in _EXTRACTORS:
        raise ValueError(
            f"TeaCache: Unknown model '{model_type}'. Available: {list(_EXTRACTORS.keys())}"
        )
    return _EXTRACTORS[model_type]


# =============================================================================
# Config-Based Extractor System
# =============================================================================


@dataclass
class ExtractorConfig:
    """Configuration for model-specific TeaCache extractors.

    Only the timestep embedding logic is model-specific; all other logic is handled generically.

    Attributes:
        model_class_name: Model class name (e.g., "LTX2VideoTransformer3DModel")
        timestep_embed_fn: Callable(module, timestep, guidance=None) -> Tensor
        timestep_param_name: Parameter name for timestep in forward() (default: "timestep")
        guidance_param_name: Parameter name for guidance if used (default: None)
        forward_params: List of parameter names (None = auto-introspect from forward signature)
        return_dict_default: Default value for return_dict parameter (default: True)
        output_model_class: Output class name for return type (default: "Transformer2DModelOutput")
    """

    model_class_name: str
    timestep_embed_fn: Callable
    timestep_param_name: str = "timestep"
    guidance_param_name: Optional[str] = None
    forward_params: Optional[List[str]] = None
    return_dict_default: bool = True
    output_model_class: str = "Transformer2DModelOutput"


class GenericExtractor:
    """Handles common TeaCache logic for all diffusion models.

    Extracts forward() arguments, creates run_blocks and postprocess callbacks,
    and delegates only timestep embedding computation to model-specific logic.
    """

    def __init__(self, config: ExtractorConfig):
        self.config = config

    def _extract_forward_args(self, module: torch.nn.Module, *args, **kwargs) -> Dict:
        """Extract and normalize forward() arguments from *args and **kwargs."""
        # Get parameter names (auto-introspect or use config)
        if self.config.forward_params is not None:
            param_names = self.config.forward_params
        else:
            # Auto-introspect forward signature
            try:
                sig = inspect.signature(module._original_forward)
                param_names = [p for p in sig.parameters if p not in ("self", "args", "kwargs")]
            except Exception as e:
                logger.warning(f"Could not introspect forward signature: {e}")
                param_names = []

        # Map positional args to parameter names
        extracted = {param_names[i]: arg for i, arg in enumerate(args) if i < len(param_names)}

        # Merge kwargs (kwargs take precedence)
        extracted.update(kwargs)
        return extracted

    def _compute_timestep_embedding(self, module: torch.nn.Module, params: Dict) -> torch.Tensor:
        """Compute timestep embedding using configured callable."""
        timestep = params.get(self.config.timestep_param_name)
        if timestep is None:
            raise ValueError(f"Missing required parameter: {self.config.timestep_param_name}")

        # Flatten timestep if needed (common pattern)
        timestep_flat = timestep.flatten() if timestep.ndim == 2 else timestep
        guidance = (
            params.get(self.config.guidance_param_name) if self.config.guidance_param_name else None
        )

        # Call configured timestep embedding function
        try:
            return self.config.timestep_embed_fn(module, timestep_flat, guidance)
        except Exception as e:
            logger.error(f"Timestep embedder failed: {e}")
            # Last resort: use timestep as-is
            logger.warning("Using timestep fallback")
            return timestep_flat.unsqueeze(-1) if timestep_flat.ndim == 1 else timestep_flat

    def __call__(self, module: torch.nn.Module, *args, **kwargs) -> CacheContext:
        """Main extractor logic - called by TeaCacheHook.

        Extracts forward arguments, computes timestep embedding, and creates callbacks
        for running the transformer and post-processing the output.
        """
        # Extract forward arguments from positional and keyword args
        params = self._extract_forward_args(module, *args, **kwargs)

        # Compute timestep embedding (used for cache distance calculation)
        t_emb = self._compute_timestep_embedding(module, params)
        return_dict = params.get("return_dict", self.config.return_dict_default)

        def run_blocks():
            """Execute the full transformer forward pass with original parameters."""
            ret = module._original_forward(**params)
            # Normalize output to tuple format
            if return_dict and not isinstance(ret, tuple):
                sample = ret.sample if hasattr(ret, "sample") else ret
                return (sample,) if not isinstance(sample, tuple) else sample
            return ret if isinstance(ret, tuple) else (ret,)

        def postprocess(output):
            """Convert cached/computed output back to expected return format."""
            if return_dict:
                if isinstance(output, tuple):
                    return output
                return Transformer2DModelOutput(sample=output)
            # For return_dict=False, unwrap single-element tuple to raw tensor
            if isinstance(output, tuple) and len(output) == 1:
                return output[0]
            # Return raw tensor as-is (TeaCacheHook always passes tensors to postprocess)
            return output

        return CacheContext(
            modulated_input=t_emb,
            hidden_states=params.get("hidden_states"),
            encoder_hidden_states=params.get("encoder_hidden_states"),
            run_transformer_blocks=run_blocks,
            postprocess=postprocess,
        )


def register_extractor_from_config(config: ExtractorConfig):
    """Register a TeaCache extractor for a model. Call this in pipeline's load() method.

    Example:
        register_extractor_from_config(ExtractorConfig(
            model_class_name="LTX2VideoTransformer3DModel",
            timestep_embed_fn=self._compute_ltx2_timestep_embedding,
        ))
    """
    extractor = GenericExtractor(config)
    register_extractor(config.model_class_name, extractor)
    logger.debug(f"Registered TeaCache extractor for {config.model_class_name}")


# =============================================================================
# TeaCache Runtime (caching hook and lifecycle management)
# =============================================================================


class TeaCacheHook:
    """Caches transformer blocks when timestep embeddings change slowly.

    The hook monitors the relative change in timestep embeddings between steps.
    When the change is small (below threshold), it reuses the cached residual
    from the previous step instead of running the full transformer.

    Separate cache states are maintained for conditional and unconditional branches
    when using Classifier-Free Guidance (CFG).
    """

    def __init__(self, config):
        self.config = config
        # Polynomial function to rescale embedding distances
        self.rescale_func = np.poly1d(config.coefficients)
        self.extractor_fn = None

        # Separate cache state for conditional (pos) and unconditional (neg) branches
        self.state_pos = self._new_state()
        self.state_neg = self._new_state()
        self.stats = {"total": 0, "cached": 0}

    def _new_state(self):
        return {"cnt": 0, "acc_dist": 0.0, "prev_input": None, "prev_residual": None}

    def initialize(self, module):
        self.extractor_fn = get_extractor(module.__class__.__name__)

    def reset_state(self):
        self.state_pos = self._new_state()
        self.state_neg = self._new_state()
        self.stats = {"total": 0, "cached": 0}

    def get_stats(self):
        total = max(self.stats["total"], 1)
        cached = self.stats["cached"]
        return {
            "hit_rate": cached / total,
            "total": total,
            "cached": cached,
            # Backward compatibility
            "total_steps": total,
            "cached_steps": cached,
            "compute_steps": total - cached,
        }

    def __call__(self, module, *args, **kwargs):
        """Main hook called during transformer forward pass.

        Decides whether to run the full transformer or reuse cached residual
        based on timestep embedding distance.
        """
        # Extract context (timestep embedding, hidden states, callbacks)
        ctx = self.extractor_fn(module, *args, **kwargs)

        # Select cache state (for CFG: separate tracking for conditional/unconditional)
        cache_branch = getattr(module, "_cache_branch", None)
        state = self.state_neg if cache_branch == "uncond" else self.state_pos

        # Decide: compute transformer or use cache?
        should_compute = self._should_compute(state, ctx.modulated_input)
        self.stats["total"] += 1

        if not should_compute and state["prev_residual"] is not None:
            # Cache hit: Add cached residual to skip transformer computation
            logger.debug(f"TeaCache: SKIP step {state['cnt']}")
            # For I2V: output might have fewer channels than input
            # Apply residual only to the latent channels
            if ctx.hidden_states.shape[1] != state["prev_residual"].shape[1]:
                # Extract latent channels (match output channels)
                num_output_channels = state["prev_residual"].shape[1]
                latent_channels = ctx.hidden_states[:, :num_output_channels]
                output = latent_channels + state["prev_residual"]
            else:
                output = ctx.hidden_states + state["prev_residual"]
            self.stats["cached"] += 1
        else:
            # Cache miss: Run full transformer and cache the residual
            outputs = ctx.run_transformer_blocks()
            output = outputs[0] if isinstance(outputs, tuple) else outputs

            # Store residual (output - input) for next potential cache hit
            # For I2V: output may have fewer channels than input
            # Compute residual only on the latent channels
            if ctx.hidden_states.shape[1] != output.shape[1]:
                # Extract latent channels (match output channels)
                num_output_channels = output.shape[1]
                latent_channels = ctx.hidden_states[:, :num_output_channels]
                state["prev_residual"] = (output - latent_channels).detach()
            else:
                original = ctx.hidden_states.clone()
                state["prev_residual"] = (output - original).detach()

        # Update state for next iteration
        state["prev_input"] = ctx.modulated_input.detach()
        state["cnt"] += 1

        return ctx.postprocess(output)

    def _should_compute(self, state, modulated_inp):
        """Decide whether to compute transformer or use cached result.

        Returns True to compute, False to use cache.
        """
        # Warmup: Always compute first few steps to build stable cache
        if self.config.ret_steps and state["cnt"] < self.config.ret_steps:
            state["acc_dist"] = 0.0
            return True

        # Cooldown: Always compute last few steps for quality
        if self.config.cutoff_steps and state["cnt"] >= self.config.cutoff_steps:
            return True

        # First step: no previous input to compare
        if state["prev_input"] is None:
            return True

        # Compute relative change in timestep embedding
        curr, prev = modulated_inp, state["prev_input"]

        # For CFG (batch_size > 1), only compare conditional branch
        # Both branches move similarly, so one comparison is sufficient
        if modulated_inp.shape[0] > 1:
            curr, prev = modulated_inp.chunk(2)[1], prev.chunk(2)[1]

        # Calculate relative L1 distance (normalized by magnitude)
        rel_dist = ((curr - prev).abs().mean() / (prev.abs().mean() + 1e-8)).cpu().item()

        # Apply polynomial rescaling to adjust sensitivity
        # Accumulate distance (capped at 2x threshold to prevent overflow)
        rescaled = float(self.rescale_func(rel_dist))
        state["acc_dist"] = min(
            state["acc_dist"] + abs(rescaled), self.config.teacache_thresh * 2.0
        )

        logger.debug(
            f"TeaCache: step {state['cnt']} | dist {rel_dist:.2e} | acc {state['acc_dist']:.4f}"
        )

        # Cache decision based on accumulated distance
        if state["acc_dist"] < self.config.teacache_thresh:
            # Below threshold: use cache, apply decay to distance
            state["acc_dist"] *= 0.95
            return False
        else:
            # Above threshold: compute, reset accumulated distance
            state["acc_dist"] = 0.0
            return True


class TeaCacheBackend:
    """Manages TeaCache lifecycle."""

    def __init__(self, config):
        self.config = config
        self.hook = None

    def enable(self, module):
        if self.hook is None:
            logger.info(f"TeaCache: Enabling for {module.__class__.__name__}")
            self.hook = TeaCacheHook(self.config)
            self.hook.initialize(module)
            module._original_forward = module.forward
            module.forward = lambda *args, **kwargs: self.hook(module, *args, **kwargs)

    def disable(self, module):
        if self.hook and hasattr(module, "_original_forward"):
            module.forward = module._original_forward
            self.hook = None

    def refresh(self, num_inference_steps):
        """Reset TeaCache state for a new generation.

        Sets warmup/cutoff steps based on total inference steps:
        - Warmup steps: Always compute to build stable cache
        - Cutoff steps: Always compute for quality at the end
        - Middle steps: Use caching based on distance threshold

        Args:
            num_inference_steps: Total number of denoising steps
        """
        if not self.hook:
            return

        # Reset cache state (clears previous residuals and counters)
        self.hook.reset_state()

        # Configure warmup and cutoff based on mode
        if self.config.use_ret_steps:
            # Aggressive warmup: 5 steps to stabilize cache
            self.config.ret_steps = 5
            self.config.cutoff_steps = num_inference_steps  # No cutoff (cache until end)
        else:
            # Minimal warmup: 1 step
            self.config.ret_steps = 1
            self.config.cutoff_steps = num_inference_steps - 2  # Compute last 2 steps

        self.config.num_steps = num_inference_steps

        logger.info(
            f"TeaCache: {num_inference_steps} steps | "
            f"warmup: {self.config.ret_steps}, cutoff: {self.config.cutoff_steps}, "
            f"thresh: {self.config.teacache_thresh}"
        )

    def is_enabled(self):
        return self.hook is not None

    def get_stats(self):
        return self.hook.get_stats() if self.hook else {}
