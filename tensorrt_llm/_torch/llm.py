from tensorrt_llm.llmapi.llm import _TorchLLM


class LLM(_TorchLLM):

    def __init__(self, *args, **kwargs):
        raise ImportError(
            "_torch.llm is deprecated, please use `from tensorrt_llm import LLM` directly"
        )


# Keep the LLM class to guide the users to use the default LLM class
__all__ = ['LLM']
