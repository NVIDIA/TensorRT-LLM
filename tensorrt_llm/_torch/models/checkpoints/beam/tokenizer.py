# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Beam checkpoints' fast tokenizer for ``custom_tokenizer``.

``custom_tokenizer: tensorrt_llm._torch.models.checkpoints.beam.tokenizer.BeamFastTokenizer``
loads the checkpoint's ``tokenizer.json`` by its concrete class. ``AutoTokenizer`` would first
parse ``config.json`` as a generic transformers configuration, which rejects Beam's
per-layer-type RoPE parameters; the tokenizer itself needs none of it.
"""

from tensorrt_llm.llmapi.tokenizer import TransformersTokenizer
from tensorrt_llm.logger import logger


def load_beam_tokenizer(checkpoint: str, **kwargs):
    """The checkpoint's ``PreTrainedTokenizerFast``, loaded by its concrete class."""
    from transformers import PreTrainedTokenizerFast

    return PreTrainedTokenizerFast.from_pretrained(checkpoint, **kwargs)


class BeamFastTokenizer(TransformersTokenizer):
    """``TransformersTokenizer`` over the Beam checkpoint's fast tokenizer (``load_beam_tokenizer``)."""

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_dir: str,
        *,
        trust_remote_code: bool = False,
        use_fast: bool = True,
        **kwargs,
    ) -> "BeamFastTokenizer":
        # A concrete transformers class runs no checkpoint code, so trust_remote_code is moot.
        if not use_fast:
            raise ValueError(
                "the Beam checkpoint's tokenizer is a fast tokenizer (tokenizer.json); "
                "tokenizer_mode 'slow' is not supported"
            )
        tokenizer = load_beam_tokenizer(pretrained_model_dir, **kwargs)
        logger.info(f"custom_tokenizer {cls.__name__}: {type(tokenizer).__name__}")
        return cls(tokenizer)

    def __reduce__(self):
        # A copy stays this class, so a receiving process keeps loading it by concrete class.
        return (type(self), (self.tokenizer,))
