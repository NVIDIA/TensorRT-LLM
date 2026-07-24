import argparse
import datetime
import typing as _tp

import torch as _tor
from lmformatenforcer import (JsonSchemaParser, TokenEnforcer,
                              TokenEnforcerTokenizerData)
from pydantic import BaseModel
from transformers import AutoTokenizer

import tensorrt_llm.bindings.executor as trtllm


def _build_regular_tokens_list(
        tokenizer) -> _tp.List[_tp.Tuple[int, str, bool]]:
    token_0 = [tokenizer.encode("0")[-1]]
    regular_tokens = []
    vocab_size = tokenizer.vocab_size
    for token_idx in range(vocab_size):
        if token_idx in tokenizer.all_special_ids:
            continue
        # We prepend token 0 and skip the first letter of the result to get a space if the token is a start word.
        tensor_after_0 = _tor.tensor(token_0 + [token_idx], dtype=_tor.long)
        decoded_after_0 = tokenizer.decode(tensor_after_0)[1:]
        decoded_regular = tokenizer.decode(token_0)
        is_word_start_token = len(decoded_after_0) > len(decoded_regular)
        regular_tokens.append((token_idx, decoded_after_0, is_word_start_token))
    return regular_tokens


def build_token_enforcer(tokenizer, character_level_parser):
    """
    Build logits processor for feeding it into generate function (use_py_session should be True)
    """
    regular_tokens = _build_regular_tokens_list(tokenizer)

    def _decode(tokens: _tp.List[int]) -> str:
        tensor = _tor.tensor(tokens, dtype=_tor.long)
        return tokenizer.decode(tensor)

    tokenizer_data = TokenEnforcerTokenizerData(regular_tokens, _decode,
                                                tokenizer.eos_token_id)
    return TokenEnforcer(tokenizer_data, character_level_parser)


# Prepare and enqueue the requests
def enqueue_requests(args: argparse.Namespace,
                     executor: trtllm.Executor) -> None:

    sampling_config = trtllm.SamplingConfig(args.beam_width)

    request_ids = []
    for iter_id in range(args.batch_size):
        # Create the request.
        request = trtllm.Request(input_token_ids=prompt,
                                 max_tokens=25,
                                 end_id=tokenizer.eos_token_id,
                                 sampling_config=sampling_config,
                                 client_id=iter_id % 2)
        request.logits_post_processor_name = request.BATCHED_POST_PROCESSOR_NAME if args.lpp_batched else "my_logits_pp"

        # Enqueue the request.
        req_id = executor.enqueue_request(request)
        request_ids.append(req_id)

    return request_ids


# Wait for responses and store output tokens
def wait_for_responses(args: argparse.Namespace, request_ids: list[int],
                       executor: trtllm.Executor) -> dict[dict[list[int]]]:

    output_tokens = {
        req_id: {
            beam: []
            for beam in range(args.beam_width)
        }
        for req_id in request_ids
    }
    num_finished = 0
    iter = 0
    while (num_finished < len(request_ids) and iter < args.timeout_ms):
        responses = executor.await_responses(
            datetime.timedelta(milliseconds=args.timeout_ms))
        for response in responses:
            req_id = response.request_id
            if not response.has_error():
                result = response.result
                num_finished += 1 if result.is_final else 0
                for beam, outTokens in enumerate(result.output_token_ids):
                    output_tokens[req_id][beam].extend(outTokens)
            else:
                raise RuntimeError(
                    str(req_id) + " encountered error:" + response.error_msg)

    return output_tokens


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Executor Bindings Example")
    parser.add_argument("--tokenizer_path",
                        "-t",
                        type=str,
                        required=True,
                        help="Directory containing model tokenizer")
    parser.add_argument("--engine_path",
                        "-e",
                        type=str,
                        required=True,
                        help="Directory containing model engine")
    parser.add_argument("--beam_width",
                        type=int,
                        required=False,
                        default=1,
                        help="The beam width")
    parser.add_argument("--batch_size",
                        type=int,
                        required=False,
                        default=1,
                        help="The batch size")
    parser.add_argument(
        "--timeout_ms",
        type=int,
        required=False,
        default=10000,
        help="The maximum time to wait for all responses, in milliseconds")
    parser.add_argument("--lpp_batched",
                        action="store_true",
                        default=False,
                        help="Enable batched logits post processor")

    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    class AnswerFormat(BaseModel):
        last_name: str
        year_of_birth: int

    parser = JsonSchemaParser(AnswerFormat.model_json_schema())
    token_enforcer = build_token_enforcer(tokenizer, parser)

    def get_allowed_tokens(ids, client_id):
        if client_id is None or client_id == 0: return [42]

        def _trim(ids):
            return [x for x in ids if x != tokenizer.eos_token_id]

        allowed = token_enforcer.get_allowed_tokens(_trim(ids[0]))
        return allowed

    def logits_post_processor(req_id: int, logits: _tor.Tensor,
                              ids: _tp.List[_tp.List[int]], stream_ptr: int,
                              client_id: _tp.Optional[int]):
        mask = _tor.full_like(logits, fill_value=float("-inf"), device="cpu")
        allowed = get_allowed_tokens(ids, client_id)
        mask[:, :, allowed] = 0

        with _tor.cuda.stream(_tor.cuda.ExternalStream(stream_ptr)):
            mask = mask.to(logits.device, non_blocking=True)
            logits += mask

    def logits_post_processor_batched(
            req_ids_batch: _tp.List[int], logits_batch: _tp.List[_tor.Tensor],
            ids_batch: _tp.List[_tp.List[_tp.List[int]]], stream_ptr,
            client_ids_batch: _tp.List[_tp.Optional[int]]):
        masks = []
        for req_id, logits, ids, client_id in zip(req_ids_batch, logits_batch,
                                                  ids_batch, client_ids_batch):
            del req_id
            mask = _tor.full_like(logits,
                                  fill_value=float("-inf"),
                                  device="cpu")
            allowed = get_allowed_tokens(ids, client_id)
            mask[:, :, allowed] = 0
            masks.append(mask)

        with _tor.cuda.stream(_tor.cuda.ExternalStream(stream_ptr)):
            for logits, mask in zip(logits_batch, masks):
                logits += mask.to(logits.device, non_blocking=True)

    # Create the executor.
    executor_config = trtllm.ExecutorConfig(args.beam_width)
    logits_proc_config = trtllm.LogitsPostProcessorConfig()
    if not args.lpp_batched:
        logits_proc_config.processor_map = {
            "my_logits_pp": logits_post_processor
        }
    else:
        logits_proc_config.processor_batched = logits_post_processor_batched
    executor_config.logits_post_processor_config = logits_proc_config
    executor = trtllm.Executor(args.engine_path, trtllm.ModelType.DECODER_ONLY,
                               executor_config)

    input = "Please give me information about Michael Jordan. You MUST answer using the following json schema: "
    prompt = tokenizer.encode(input)
    print(f"Input text: {input}\n")

    if executor.can_enqueue_requests():
        request_ids = enqueue_requests(args, executor)
        output_tokens = wait_for_responses(args, request_ids, executor)

        # Print output
        for req_id in request_ids:
            for beam_id in range(args.beam_width):
                result = tokenizer.decode(
                    output_tokens[req_id][beam_id][len(prompt):])
                generated_tokens = len(
                    output_tokens[req_id][beam_id]) - len(prompt)
                print(
                    f"Request {req_id} Beam {beam_id} ({generated_tokens} tokens): {result}"
                )
