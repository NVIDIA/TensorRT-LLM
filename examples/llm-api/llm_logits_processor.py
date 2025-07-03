### :title Control generated text using logits processor
### :section Customization
### :order 1
from typing import List, Optional

import torch
from transformers import PreTrainedTokenizer

from tensorrt_llm import LLM
from tensorrt_llm.sampling_params import LogitsProcessor, SamplingParams


def text_to_token(tokenizer: PreTrainedTokenizer, text: str, last: bool):
    tokens = tokenizer.encode(text, add_special_tokens=False)

    max_token_count = 1
    bos_token_added = getattr(tokenizer, 'bos_token', None) and getattr(
        tokenizer, 'bos_token_id', None) in tokens
    prefix_token_added = getattr(tokenizer, 'add_prefix_space',
                                 None) is not False
    if bos_token_added or prefix_token_added:
        max_token_count = 2

    if not last and len(tokens) > max_token_count:
        raise Exception(
            f"Can't convert {text} to token. It has {len(tokens)} tokens.")

    return tokens[-1]


# The recommended way to create a customized logits processor:
#     * Subclass LogitsProcessor and implement the processing logics in the __call__ method.
#     * Create an instance and pass to SamplingParams.
# More LogitsProcessors references can be found at https://github.com/NVIDIA/logits-processor-zoo.
class GenLengthLogitsProcessor(LogitsProcessor):
    """
    A logits processor that adjusts the likelihood of the end-of-sequence (EOS) token
    based on the length of the generated sequence, encouraging or discouraging shorter answers.
    WARNING: Create a new object before every model.generate call since token_count is accumulated.

    Parameters
    ----------
    tokenizer: The tokenizer used by the LLM.
    boost_factor (float): A factor to boost the likelihood of the EOS token as the sequence length increases.
                        Suggested value range is [-1.0, 1.0]. Negative values are used for the opposite effect.
    p (int, optional): The power to which the token count is raised when computing the boost value. Default is 2.
    complete_sentences (bool, optional): If True, boosts EOS token likelihood only when the last token is a full stop
                                        or a new line. Default is False.

    """

    def __init__(self,
                 tokenizer,
                 boost_factor: float,
                 p: int = 2,
                 complete_sentences: bool = False):
        self.eos_token = tokenizer.eos_token_id
        self.boost_factor = boost_factor
        self.p = p
        self.token_count = 0
        self.full_stop_token = text_to_token(tokenizer,
                                             "It is a sentence.",
                                             last=True)
        self.new_line_token = text_to_token(tokenizer,
                                            "It is a new line\n",
                                            last=True)
        self.complete_sentences = complete_sentences

    def __call__(self, req_ids: int, logits: torch.Tensor, ids: List[List[int]],
                 stream_ptr, client_id: Optional[int]):
        boost_val = self.boost_factor * (self.token_count**self.p) / (10**
                                                                      self.p)

        stream = None if stream_ptr is None else torch.cuda.ExternalStream(
            stream_ptr)

        with torch.cuda.stream(stream):
            ids = torch.LongTensor(ids).to(logits.device, non_blocking=True)

            if self.complete_sentences:
                enabled = (ids[:, -1] == self.full_stop_token) | (
                    ids[:, -1] == self.new_line_token)
                logits[:, :, self.eos_token] += enabled * boost_val
            else:
                logits[:, :, self.eos_token] += boost_val

        self.token_count += 1


def main():

    llm = LLM(model="TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    # Sample prompts
    prompts = [
        "The future of AI is",
        "The future of AI is",
    ]

    # Generate text
    for prompt_id, prompt in enumerate(prompts):
        if prompt_id % 2 == 0:
            # Without logit processor
            sampling_params = SamplingParams(top_p=1, max_tokens=200)
        else:
            # Each prompt can be specified with a logits processor at runtime
            sampling_params = SamplingParams(
                temperature=0.8,
                top_p=0.95,
                logits_processor=GenLengthLogitsProcessor(
                    llm.tokenizer, boost_factor=1, complete_sentences=True))

        output = llm.generate(prompt, sampling_params)
        print(
            f"Prompt: {output.prompt!r}, Generated text: {output.outputs[0].text!r}"
        )

    # Got output like:
    # Prompt (original): "bright, and it's not just for big companies. Small businesses can also benefit from AI technology. Here are some ways:\n\n1. Improved customer service: AI can help businesses provide better customer service by analyzing customer data and providing personalized recommendations.
    #                    This can help businesses improve their customer experience and increase customer loyalty.\n\n2. Increased productivity: AI can help businesses automate repetitive tasks, freeing up employees to focus on more complex tasks. This can
    #                    help businesses increase productivity and reduce costs.\n\n3. Enhanced marketing: AI can help businesses create more personalized marketing campaigns by analyzing customer data and targeting specific audiences. This can help businesses
    #                    increase their marketing ROI and drive more sales.\n\n4. Improved supply chain management: AI can help businesses optimize their supply chain by analyzing data on demand,"'
    #
    # Prompt (with GenLenthLogitsProcesor): "bright, and it's not just for big companies. Small businesses can also benefit from AI technology."


if __name__ == '__main__':
    main()
