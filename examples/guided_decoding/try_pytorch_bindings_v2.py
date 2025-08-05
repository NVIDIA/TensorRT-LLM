import torch

from tensorrt_llm._torch.hostfunc import free_hostfunc_user_data
from tensorrt_llm._torch.pyexecutor.guided_decoder import GuidedDecoder
from tensorrt_llm.bindings.executor import (GuidedDecodingConfig,
                                            GuidedDecodingParams)
from tensorrt_llm.llmapi.tokenizer import (_xgrammar_tokenizer_info,
                                           load_hf_tokenizer)


def main():
    # stream = torch.cuda.current_stream()
    stream = torch.cuda.Stream()

    tokenizer = load_hf_tokenizer("nvidia/Llama-3.1-8B-Instruct-FP8")
    guided_decoding_config = GuidedDecodingConfig(
        backend=GuidedDecodingConfig.GuidedDecodingBackend.XGRAMMAR,
        # backend=GuidedDecodingConfig.GuidedDecodingBackend.LLGUIDANCE,
        **_xgrammar_tokenizer_info(tokenizer),
    )
    guided_decoder = GuidedDecoder(guided_decoding_config, 1024,
                                   len(tokenizer.tokenizer))
    guided_decoding_params = GuidedDecodingParams(
        guide_type=GuidedDecodingParams.GuideType.JSON)
    guided_decoder.grammar_matchers[
        0] = guided_decoder.grammar_matcher_factory.create(
            guided_decoding_params)

    torch.cuda.synchronize()
    with torch.cuda.stream(stream):
        guided_decoder.bitmask.zero_()
        guided_decoder.bitmask_host.zero_()
        handle = guided_decoder.inc_bitmask_host()
    torch.cuda.synchronize()
    print(guided_decoder.bitmask_host)
    # Free the hostfunc user data actively if CUDA graph is not used.
    free_hostfunc_user_data(handle)

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=stream):
        guided_decoder.bitmask.add_(1)
        guided_decoder.bitmask_host.copy_(guided_decoder.bitmask,
                                          non_blocking=True)
        handle = guided_decoder.inc_bitmask_host()
        guided_decoder.bitmask.copy_(guided_decoder.bitmask_host,
                                     non_blocking=True)
    torch.cuda.synchronize()
    print("Graph captured", flush=True)

    with torch.cuda.stream(stream):
        for i in range(10):
            g.replay()

    torch.cuda.synchronize()
    print(guided_decoder.bitmask)

    # Free the hostfunc user data after the CUDA graph is used.
    free_hostfunc_user_data(handle)


if __name__ == "__main__":
    main()
