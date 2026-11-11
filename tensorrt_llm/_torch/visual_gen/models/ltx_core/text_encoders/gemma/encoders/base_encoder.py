from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

import torch
from transformers import PreTrainedModel, ProcessorMixin

from tensorrt_llm._torch.visual_gen.models.ltx_core.loader.module_ops import ModuleOps
from tensorrt_llm._torch.visual_gen.models.ltx_core.model.disposable import Disposable
from tensorrt_llm._torch.visual_gen.models.ltx_core.text_encoders.gemma.gemma_assets import (
    TOKENIZER_MAX_LENGTH,
    GemmaAssets,
    build_gemma_hf_tokenizer,
    build_gemma_processor,
)
from tensorrt_llm._torch.visual_gen.models.ltx_core.text_encoders.gemma.tokenizer import LTXGemmaTokenizer, PaddingSide

# Sampling decoding for the Gemma 3 self-enhance path.
GEMMA3_ENHANCE_GENERATION_KWARGS: dict[str, Any] = {"do_sample": True, "temperature": 0.7, "max_new_tokens": 512}
# Greedy decoding for Gemma 4 instruct enhancers.
GEMMA4_ENHANCE_GENERATION_KWARGS: dict[str, Any] = {
    "do_sample": False,
    "no_repeat_ngram_size": 5,
    "max_new_tokens": 600,
}


class LTXGemmaTextEncoder(torch.nn.Module, Disposable):
    """Pure Gemma text encoder — runs the LLM and returns raw hidden states.
    Prompt enhancement (generate) is also supported since the full Gemma
    multimodal-generation model (including lm_head) is loaded.
    """

    def __init__(
        self,
        model: PreTrainedModel | None = None,
        tokenizer: LTXGemmaTokenizer | None = None,
        processor: ProcessorMixin | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.processor = processor
        self._dtype = dtype

    def encode(self, prompts: list[str]) -> list[tuple[tuple[torch.Tensor, ...], torch.Tensor]]:
        """Run a single fused Gemma forward over a batch of prompts.
        Calls the inner model (self.model.model) to skip lm_head logits computation
        (~500 MiB saving). The tokenizer pads every prompt to ``max_length`` (1024),
        so the inputs stack into a single ``[N, 1024]`` batch with no further padding
        logic; per-prompt outputs are sliced back to ``[1, 1024, D]`` / ``[1, 1024]``
        in the original order.
        """
        if not prompts:
            return []
        tokenized = [self.tokenizer.tokenize_with_weights(t)["gemma"] for t in prompts]
        input_ids = torch.tensor(
            [[tok for tok, _ in pairs] for pairs in tokenized],
            device=self.model.device,
        )
        attention_mask = torch.tensor(
            [[w for _, w in pairs] for pairs in tokenized],
            device=self.model.device,
        )
        outputs = self.model.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        del outputs
        return [(tuple(h[i : i + 1] for h in hidden_states), attention_mask[i : i + 1]) for i in range(len(prompts))]

    def _model_type(self) -> str:
        return getattr(getattr(self.model, "config", None), "model_type", "") or ""

    def _default_generation_kwargs(self) -> dict[str, Any]:
        model_type = self._model_type()
        if model_type == "gemma3":
            return dict(GEMMA3_ENHANCE_GENERATION_KWARGS)
        if model_type == "gemma4":
            return dict(GEMMA4_ENHANCE_GENERATION_KWARGS)
        raise ValueError(f"Unsupported Gemma enhancer model_type={model_type!r}; expected 'gemma3' or 'gemma4'.")

    def _default_system_prompt(self, *, t2v: bool) -> str:
        model_type = self._model_type()
        if model_type == "gemma3":
            return default_gemma3_t2v_system_prompt() if t2v else default_gemma3_i2v_system_prompt()
        if model_type == "gemma4":
            return default_gemma4_t2v_system_prompt() if t2v else default_gemma4_i2v_system_prompt()
        raise ValueError(f"Unsupported Gemma enhancer model_type={model_type!r}; expected 'gemma3' or 'gemma4'.")

    def _enhance(
        self,
        messages: list[dict],
        image: torch.Tensor | None = None,
        max_new_tokens: int | None = None,
        seed: int = 10,
        static_cache: bool = False,
    ) -> str:
        text = self.processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        model_inputs = self.processor(
            text=text,
            images=image,
            return_tensors="pt",
        ).to(self.model.device)
        pad_token_id = self.processor.tokenizer.pad_token_id if self.processor.tokenizer.pad_token_id is not None else 0
        model_inputs = _pad_inputs_for_attention_alignment(model_inputs, pad_token_id=pad_token_id)

        gen_kwargs = self._default_generation_kwargs()
        if static_cache:
            gen_kwargs["cache_implementation"] = "static"
        if max_new_tokens is not None:
            gen_kwargs["max_new_tokens"] = max_new_tokens

        # fork_rng device pinning is only supported for CUDA; MPS/CPU fork CPU RNG only.
        fork_devices = [self.model.device] if self.model.device.type == "cuda" else []
        with torch.inference_mode(), torch.random.fork_rng(devices=fork_devices):
            torch.manual_seed(seed)
            outputs = self.model.generate(**model_inputs, **gen_kwargs)
            generated_ids = outputs[0][len(model_inputs.input_ids[0]) :]
            enhanced_prompt = self.processor.tokenizer.decode(generated_ids, skip_special_tokens=True)

        return enhanced_prompt

    def enhance_t2v(
        self,
        prompt: str,
        max_new_tokens: int | None = None,
        system_prompt: str | None = None,
        seed: int = 10,
        static_cache: bool = False,
    ) -> str:
        """Enhance a text prompt for T2V generation."""
        system_prompt = system_prompt or self._default_system_prompt(t2v=True)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"user prompt: {prompt}"},
        ]
        return self._enhance(messages, max_new_tokens=max_new_tokens, seed=seed, static_cache=static_cache)

    def enhance_i2v(
        self,
        prompt: str,
        image: torch.Tensor,
        max_new_tokens: int | None = None,
        system_prompt: str | None = None,
        seed: int = 10,
        static_cache: bool = False,
    ) -> str:
        """Enhance a text prompt for I2V generation using a reference image."""
        system_prompt = system_prompt or self._default_system_prompt(t2v=False)
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": f"User Raw Input Prompt: {prompt}."},
                ],
            },
        ]
        return self._enhance(messages, image=image, max_new_tokens=max_new_tokens, seed=seed, static_cache=static_cache)


# --- Standalone utility functions ---


@functools.lru_cache(maxsize=4)
def _load_system_prompt(prompt_name: str) -> str:
    with open(Path(__file__).parent / "prompts" / f"{prompt_name}", "r") as f:
        return f.read()


def default_gemma3_i2v_system_prompt() -> str:
    return _load_system_prompt("gemma3_i2v_system_prompt.txt")


def default_gemma3_t2v_system_prompt() -> str:
    return _load_system_prompt("gemma3_t2v_system_prompt.txt")


def default_gemma4_i2v_system_prompt() -> str:
    return _load_system_prompt("gemma4_i2v_system_prompt.txt")


def default_gemma4_t2v_system_prompt() -> str:
    return _load_system_prompt("gemma4_t2v_system_prompt.txt")


def _cat_with_padding(
    tensor: torch.Tensor,
    padding_length: int,
    value: int | float,
) -> torch.Tensor:
    """Left-pad a tensor (prepend the padding) with the given value.
    Decoder-only LLMs must be LEFT-padded for generation: with right-padding the trailing
    pad positions become the sequence end, so generate() continues from a pad position
    instead of the real last token. Left-padding keeps the real last token at the end
    while still aligning the sequence length to a multiple of 8.
    See https://huggingface.co/docs/transformers/en/llm_tutorial#padding-side
    """
    pad = torch.full(
        (1, padding_length),
        value,
        dtype=tensor.dtype,
        device=tensor.device,
    )
    return torch.cat([pad, tensor], dim=1)


def _pad_inputs_for_attention_alignment(
    model_inputs: dict[str, torch.Tensor],
    pad_token_id: int = 0,
    alignment: int = 8,
) -> dict[str, torch.Tensor]:
    """Pad sequence length to multiple of alignment for Flash Attention compatibility."""
    seq_len = model_inputs.input_ids.shape[1]
    padded_len = ((seq_len + alignment - 1) // alignment) * alignment
    padding_length = padded_len - seq_len

    if padding_length > 0:
        model_inputs["input_ids"] = _cat_with_padding(model_inputs.input_ids, padding_length, pad_token_id)
        model_inputs["attention_mask"] = _cat_with_padding(model_inputs.attention_mask, padding_length, 0)
        if "token_type_ids" in model_inputs and model_inputs["token_type_ids"] is not None:
            model_inputs["token_type_ids"] = _cat_with_padding(model_inputs["token_type_ids"], padding_length, 0)

    return model_inputs


def build_gemma_tokenizer(assets: GemmaAssets) -> LTXGemmaTokenizer:
    return LTXGemmaTokenizer(
        build_gemma_hf_tokenizer(assets),
        TOKENIZER_MAX_LENGTH,
        padding_side=PaddingSide.LEFT,
    )


def module_ops_from_gemma_root(gemma_root: str, *, assets: GemmaAssets | None = None) -> tuple[ModuleOps, ...]:
    """Return asset operations, reusing already-loaded assets when supplied."""
    if assets is None:
        assets = GemmaAssets.load(gemma_root)

    def load_tokenizer(module: LTXGemmaTextEncoder) -> LTXGemmaTextEncoder:
        module.tokenizer = build_gemma_tokenizer(assets)
        return module

    def load_processor(module: LTXGemmaTextEncoder) -> LTXGemmaTextEncoder:
        if not module.tokenizer:
            raise ValueError("Tokenizer model operation must be performed before processor model operation")
        module.processor = build_gemma_processor(assets, module.tokenizer.tokenizer)
        return module

    tokenizer_load_ops = ModuleOps(
        "TokenizerLoad",
        matcher=lambda module: isinstance(module, LTXGemmaTextEncoder) and module.tokenizer is None,
        mutator=load_tokenizer,
    )
    processor_load_ops = ModuleOps(
        "ProcessorLoad",
        matcher=lambda module: isinstance(module, LTXGemmaTextEncoder) and module.processor is None,
        mutator=load_processor,
    )
    return (tokenizer_load_ops, processor_load_ops)
