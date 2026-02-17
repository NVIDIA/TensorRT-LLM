"""Output dataclass for visual generation models."""

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class MediaOutput:
    """Unified output for all visual generation models.

    Different models populate different fields:
    - FLUX2: image only
    - WAN: video only
    - LTX2: video + audio

    Attributes:
        image: Generated image as torch tensor with shape (height, width, channels) and dtype uint8.
               Populated by FLUX2 for text-to-image generation.
        video: Generated video frames as torch tensor with shape (num_frames, height, width, channels) and dtype uint8.
               Populated by WAN and LTX2 for text-to-video generation.
        audio: Generated audio as torch tensor with dtype float32.
               Populated by LTX2 for text-to-video-with-audio generation.
    """

    image: Optional[torch.Tensor] = None
    video: Optional[torch.Tensor] = None
    audio: Optional[torch.Tensor] = None
