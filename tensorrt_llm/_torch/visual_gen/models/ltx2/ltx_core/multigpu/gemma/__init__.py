"""Multi-GPU utilities for the Gemma text encoder."""

from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.multigpu.gemma.accelerate_wrapper import AccelerateGemmaWrapper

__all__ = ["AccelerateGemmaWrapper"]
