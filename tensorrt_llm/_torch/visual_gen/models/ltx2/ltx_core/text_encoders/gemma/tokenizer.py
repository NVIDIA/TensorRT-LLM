from enum import Enum

from transformers import PreTrainedTokenizerBase


class PaddingSide(str, Enum):
    LEFT = "left"
    RIGHT = "right"


class LTXGemmaTokenizer:
    """Wraps a HuggingFace tokenizer and normalizes padding / BOS for encode.
    Always ensures a leading ``<bos>``: Gemma 3 already emits it via post_processor;
    Gemma 4 does not, so we prepend. EOS is not appended on this path.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 256,
        padding_side: PaddingSide = PaddingSide.LEFT,
    ):
        self.tokenizer = tokenizer
        self.tokenizer.model_max_length = max_length
        self.tokenizer.padding_side = padding_side.value
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.max_length = max_length

    def tokenize_with_weights(self, text: str, return_word_ids: bool = False) -> dict[str, list[tuple[int, int]]]:
        """Return ``{"gemma": [(token_id, attn), ...]}``; with ``return_word_ids``, triples include index."""
        text = text.strip()
        bos_id = self.tokenizer.bos_token_id
        if bos_id is None:
            raise ValueError("Tokenizer is missing bos_token_id; encode path requires a leading BOS.")
        encoded = self.tokenizer(
            text,
            padding=False,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids = encoded.input_ids[0].tolist()
        if not input_ids or input_ids[0] != bos_id:
            input_ids = [bos_id, *input_ids][: self.max_length]

        padded = self.tokenizer.pad(
            {"input_ids": [input_ids]},
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
            return_attention_mask=True,
        )
        input_ids = padded.input_ids
        attention_mask = padded.attention_mask
        tuples = [
            (token_id, attn, i) for i, (token_id, attn) in enumerate(zip(input_ids[0], attention_mask[0], strict=True))
        ]
        out = {"gemma": tuples}
        if not return_word_ids:
            out = {k: [(t, w) for t, w, _ in v] for k, v in out.items()}
        return out
