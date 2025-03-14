from typing import Any, Dict, List, Tuple

import torch.nn.functional as F
from build_and_run_ad import build_llm_from_config
from lm_eval.__main__ import cli_evaluate
from lm_eval.api.model import TemplateLM
from lm_eval.api.registry import register_model
from simple_config import SimpleConfig
from tqdm import tqdm

from tensorrt_llm._torch.auto_deploy.utils.logger import ad_logger
from tensorrt_llm.llmapi import RequestOutput
from tensorrt_llm.sampling_params import SamplingParams


@register_model("autodeploy")
class AutoDeployEval(TemplateLM):
    def __init__(self, chunk_size: int = 200, **kwargs):
        super().__init__()

        # some lm-eval specific default values
        kwargs["max_tokens"] = max(int(kwargs.get("max_tokens", 0)), 256)
        kwargs["max_seq_len"] = max(2048 + kwargs["max_tokens"], int(kwargs.get("max_seq_len", 0)))
        if "batch_size" in kwargs:
            kwargs["batch_size"] = int(kwargs["batch_size"])

        self.config = SimpleConfig(**kwargs)
        self.chunk_size = chunk_size

        ad_logger.info(f"AutoDeploy config: {self.config}")
        self.llm = build_llm_from_config(self.config)
        ad_logger.info("Loaded AutoDeploy model")

    @property
    def eot_token_id(self) -> int:
        return self.llm.tokenizer.eos_token_id

    def tok_encode(self, string, add_special_tokens=False, **kwargs):
        return self.llm.tokenizer.encode(string, add_special_tokens=add_special_tokens, **kwargs)

    def _loglikelihood_tokens(
        self, requests: List[Any], disable_tqdm: bool = False
    ) -> List[Tuple[float, bool]]:
        """Compute the log likelihood of the continuation given the context."""
        # some book-keeping...
        num_r = len(requests)
        desc = "Processing loglikelihood requests"
        sampling_params = SamplingParams(max_tokens=1, return_context_logits=True)

        # process requests
        futures: Dict[int, RequestOutput] = {}
        results = []
        for i, request in tqdm(enumerate(requests), desc=desc, total=num_r, disable=disable_tqdm):
            # asynchronously submit a chunk of requests ahead of time...
            if i % self.chunk_size == 0:
                for j in range(i, min(i + self.chunk_size, num_r)):
                    prompt_ids = requests[j][1] + requests[j][2]
                    futures[j] = self.llm.generate_async(prompt_ids, sampling_params)

            # process the output of the request i
            r_out: RequestOutput = futures.pop(i).result()

            # check continuation portion of the prompt
            # NOTE: context_logits are offset by 1 since they predict future token
            ctxlen = len(request[1])
            token_ids_cont = request[2]
            logits_cont = r_out.context_logits[ctxlen - 1 : -1]  # [sl, vocab]
            logprobs_cont = F.log_softmax(logits_cont, dim=-1)  # [sl, vocab]
            top_tokens_cont = logprobs_cont.argmax(dim=-1).tolist()  # [sl]

            # compute logprob and check for greedy
            logprob_sum = sum(logprobs_cont[list(range(len(logprobs_cont))), token_ids_cont]).item()
            is_greedy = top_tokens_cont == token_ids_cont

            results.append((logprob_sum, is_greedy))

            # clear response
            del r_out

        return results

    def loglikelihood_rolling(self, requests, disable_tqdm: bool = False):
        raise NotImplementedError

    def generate_until(self, requests: List[Any], disable_tqdm: bool = False) -> List[str]:
        # some book-keeping and parameters...
        num_r = len(requests)
        desc = "Processing generate requests"

        def _get_sp(gen_kwargs):
            k_mapping = {
                "temperature": "temperature",
                "top_p": "top_p",
                "until": "stop",
            }
            kwargs_mapped = {
                k_sp: gen_kwargs[k_gen] for k_gen, k_sp in k_mapping.items() if k_gen in gen_kwargs
            }
            return SamplingParams(max_tokens=self.config.max_tokens, **kwargs_mapped)

        # process requests
        futures: Dict[int, RequestOutput] = {}
        results = []
        for i, _ in tqdm(enumerate(requests), desc=desc, total=num_r, disable=disable_tqdm):
            # asynchronously submit a chunk of requests ahead of time...
            if i % self.chunk_size == 0:
                for j in range(i, min(i + self.chunk_size, num_r)):
                    context, gen_kwargs = requests[j].args
                    prompt_ids = self.tok_encode(context)
                    futures[j] = self.llm.generate_async(prompt_ids, _get_sp(gen_kwargs))

            # process the output of the request i
            r_out: RequestOutput = futures.pop(i).result()
            results.append(r_out.outputs[0].text)

        return results


if __name__ == "__main__":
    cli_evaluate()
