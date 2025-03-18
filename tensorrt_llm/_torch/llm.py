from pathlib import Path
from typing import Any, Literal, Optional, Union

from transformers import PreTrainedTokenizerBase

from ..llmapi.llm import LLM as BaseLLM
from ..llmapi.llm import TokenizerBase


class LLM(BaseLLM):

    def __init__(self,
                 model: str,
                 tokenizer: Optional[Union[str, Path, TokenizerBase,
                                           PreTrainedTokenizerBase]] = None,
                 tokenizer_mode: Literal['auto', 'slow'] = 'auto',
                 skip_tokenizer_init: bool = False,
                 trust_remote_code: bool = False,
                 tensor_parallel_size: int = 1,
                 pipeline_parallel_size: int = 1,
                 dtype: str = "auto",
                 revision: Optional[str] = None,
                 tokenizer_revision: Optional[str] = None,
                 **kwargs: Any):

        kwargs_dict = dict(kwargs)
        kwargs_dict['backend'] = 'pytorch'
        super().__init__(model, tokenizer, tokenizer_mode, skip_tokenizer_init,
                         trust_remote_code, tensor_parallel_size,
                         pipeline_parallel_size, dtype, revision,
                         tokenizer_revision, **kwargs_dict)
