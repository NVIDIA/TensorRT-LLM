from typing import Any, List

TokenIdsTy = List[int]


class TokenizerBase:
    ''' This is a protocol for the tokenizer. Users can implement their own tokenizer by inheriting this class.  '''

    @property
    def eos_token_id(self) -> int:
        ''' Return the id of the end of sentence token.  '''
        raise NotImplementedError()

    @property
    def pad_token_id(self) -> int:
        ''' Return the id of the padding token.  '''
        raise NotImplementedError()

    def encode(self, text: str, *args, **kwargs) -> TokenIdsTy:
        ''' Encode the text to token ids.  '''
        raise NotImplementedError()

    def decode(self, token_ids: TokenIdsTy, *args, **kwargs) -> str:
        ''' Decode the token ids to text.  '''
        raise NotImplementedError()

    def batch_encode_plus(self, texts: List[str]) -> dict:
        ''' Encode the batch of texts to token ids.  '''
        raise NotImplementedError()

    def tokenize(self, text, *args, **kwargs):
        return self.encode(text, *args, **kwargs)

    def __call__(self, text: str, *args, **kwargs) -> Any:
        ''' Encode the text to token ids.  '''
        raise NotImplementedError()


class TransformersTokenizer(TokenizerBase):
    ''' A wrapper for the Transformers' tokenizer.
    This is the default tokenizer for LLM. '''

    @classmethod
    def from_pretrained(self, pretrained_model_dir: str, **kwargs):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_dir,
                                                  **kwargs)
        return TransformersTokenizer(tokenizer)

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, text: str, *args, **kwargs) -> Any:
        return self.tokenizer(text, *args, **kwargs)

    @property
    def eos_token_id(self) -> int:
        return self.tokenizer.eos_token_id

    @property
    def pad_token_id(self) -> int:
        return self.tokenizer.pad_token_id

    def encode(self, text: str, *args, **kwargs) -> TokenIdsTy:
        return self.tokenizer.encode(text, *args, **kwargs)

    def decode(self, token_ids: TokenIdsTy, *args, **kwargs) -> str:
        return self.tokenizer.decode(token_ids, *args, **kwargs)

    def batch_encode_plus(self, texts: List[str], *args, **kwargs) -> dict:
        return self.tokenizer.batch_encode_plus(texts, *args, **kwargs)
