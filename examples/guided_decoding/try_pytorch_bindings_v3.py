import torch
import xgrammar as xgr

from tensorrt_llm._torch.pyexecutor.guided_decoder import GuidedDecoder
from tensorrt_llm.bindings.executor import (GuidedDecodingConfig,
                                            GuidedDecodingParams)
from tensorrt_llm.llmapi.tokenizer import (_xgrammar_tokenizer_info,
                                           load_hf_tokenizer)
from tensorrt_llm.logger import logger


class DummyModel(torch.nn.Module):

    def __init__(self, vocab_size: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.a = torch.nn.Parameter(
            torch.randn(4096, 4096, dtype=torch.float32, device='cuda'))
        self.b = torch.nn.Parameter(
            torch.randn(4096, 4096, dtype=torch.float32, device='cuda'))

    def forward(self, x: torch.Tensor):
        # Simulate some GPU computation.
        for i in range(10):
            torch.matmul(self.a, self.b)
        return torch.randn(x.size(0),
                           self.vocab_size,
                           dtype=torch.float32,
                           device='cuda')


def main():
    logger.set_level("debug")
    # stream = torch.cuda.current_stream()
    stream = torch.cuda.Stream()

    tokenizer = load_hf_tokenizer("nvidia/Llama-3.1-8B-Instruct-FP8")
    vocab_size = len(tokenizer.tokenizer)
    guided_decoding_config = GuidedDecodingConfig(
        backend=GuidedDecodingConfig.GuidedDecodingBackend.XGRAMMAR,
        # backend=GuidedDecodingConfig.GuidedDecodingBackend.LLGUIDANCE,
        **_xgrammar_tokenizer_info(tokenizer),
    )
    guided_decoder = GuidedDecoder(guided_decoding_config, 1024, vocab_size)
    guided_decoding_params = GuidedDecodingParams(
        guide_type=GuidedDecodingParams.GuideType.REGEX,
        guide=r"\d{5} XGrammar \d{100}")
    guided_decoder.grammar_matchers[
        0] = guided_decoder.grammar_matcher_factory.create(
            guided_decoding_params)

    model = DummyModel(vocab_size)
    model.to('cuda')

    new_token_event = torch.cuda.Event()
    bitmask_event = torch.cuda.Event()

    token_ids_cuda = torch.tensor([5332], dtype=torch.int32, device='cuda')
    token_ids_cpu = torch.empty_like(token_ids_cuda,
                                     device='cpu',
                                     pin_memory=True)
    # Warm up.
    logits = model(token_ids_cuda)

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=stream):
        for i in range(10):
            token_ids_cpu.copy_(token_ids_cuda, non_blocking=True)
            new_token_event.record()
            logits = model(token_ids_cuda)

            with torch.cuda.stream(guided_decoder._stream):
                torch.cuda.current_stream().wait_event(new_token_event)
                guided_decoder.run(token_ids_cpu)
                guided_decoder.bitmask[0].copy_(guided_decoder.bitmask_host[0],
                                                non_blocking=True)
                bitmask_event.record()

            torch.cuda.current_stream().wait_event(bitmask_event)
            # Not compatible with CUDA graph because of host tensor creation.
            # torch.ops.trtllm.logits_bitmask([logits[0]], [guided_decoder.bitmask[0]])
            xgr.apply_token_bitmask_inplace(logits, guided_decoder.bitmask[:1])
            new_token_ids = logits.argmax(dim=-1)
            token_ids_cuda.copy_(new_token_ids, non_blocking=True)

    torch.cuda.synchronize()
    print("Graph captured", flush=True)

    with torch.cuda.stream(stream):
        g.replay()
        g.replay()

    torch.cuda.synchronize()
    print("Graph replayed", flush=True)

    assert len(guided_decoder.token_ids) == 20
    print(tokenizer.decode(guided_decoder.token_ids))


if __name__ == "__main__":
    main()
