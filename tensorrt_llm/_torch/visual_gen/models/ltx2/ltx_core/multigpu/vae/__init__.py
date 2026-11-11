"""Multi-GPU utilities for VAE decoding."""

from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.multigpu.vae.distributed_decoder import DistributedVideoDecoder

__all__ = ["DistributedVideoDecoder"]
