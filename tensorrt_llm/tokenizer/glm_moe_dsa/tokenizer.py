"""GLM-Moe-Dsa tokenizer implementation.

Loads tokenizer from tokenizer.json and applies tokenizer_config.json manually
to work around incompatibilities when the checkpoint was saved with
transformers 5.x (TokenizersBackend / list-style extra_special_tokens).
"""

import json
from pathlib import Path
from typing import Any, Dict, Union

from tokenizers import Tokenizer as RustTokenizer
from transformers import PreTrainedTokenizerFast

from ..tokenizer import TransformersTokenizer

# Keys from tokenizer_config.json that are safe to pass to PreTrainedTokenizerFast.
# extra_special_tokens is not copied directly to the new config. It is renamed to additional_special_tokens
# for compatibility with older transformers.
_SAFE_CONFIG_KEYS = (
    "pad_token",
    "pad_token_id",
    "eos_token",
    "eos_token_id",
    "bos_token",
    "bos_token_id",
    "unk_token",
    "unk_token_id",
    "model_max_length",
    "padding_side",
    "truncation_side",
)


def _load_tokenizer_config(config_path: Path) -> Dict[str, Any]:
    """Load tokenizer_config.json and return a dict safe for older transformers."""
    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)
    out = {}
    for key in _SAFE_CONFIG_KEYS:
        if key in config:
            out[key] = config[key]

    if "extra_special_tokens" in config:
        out["additional_special_tokens"] = config["extra_special_tokens"]
    return out


class GlmMoeDsaTokenizer(TransformersTokenizer):
    """Tokenizer for GLM-Moe-Dsa / GLM-5 checkpoints that use TokenizersBackend.

    Loads from tokenizer.json only, then applies tokenizer_config.json manually
    so that checkpoints saved with transformers 5.x work on older transformers.
    """

    def __init__(self, tokenizer: PreTrainedTokenizerFast):
        self.tokenizer = tokenizer
        self._all_special_tokens_set = set(self.tokenizer.all_special_tokens)

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.vocab_size

    @classmethod
    def from_pretrained(
        cls,
        path_or_repo_id: Union[str, Path],
        *args,
        trust_remote_code: bool = False,
        revision: Union[str, None] = None,
        use_fast: bool = True,
        **kwargs,
    ) -> "GlmMoeDsaTokenizer":
        path = Path(path_or_repo_id)
        tokenizer_json = path / "tokenizer.json"
        if not tokenizer_json.exists():
            raise FileNotFoundError(
                f"Expected tokenizer.json at {tokenizer_json}. "
                "GlmMoeDsaTokenizer loads from tokenizer.json only."
            )

        rust_tok = RustTokenizer.from_file(str(tokenizer_json))
        init_kwargs: Dict[str, Any] = {}
        config_path = path / "tokenizer_config.json"
        if config_path.exists():
            init_kwargs = _load_tokenizer_config(config_path)
        hf_tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=rust_tok,
            **init_kwargs,
        )

        # Load chat template from chat_template.jinja
        chat_template_path = path / "chat_template.jinja"
        if chat_template_path.exists():
            with open(chat_template_path, encoding="utf-8") as f:
                hf_tokenizer.chat_template = f.read()

        return cls(hf_tokenizer)
