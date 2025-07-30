import torch

from tensorrt_llm.bindings import CudaStream, DataType
from tensorrt_llm.bindings.executor import GuidedDecodingConfig
from tensorrt_llm.bindings.internal.batch_manager import GuidedDecoder
from tensorrt_llm.bindings.internal.runtime import BufferManager
from tensorrt_llm.llmapi.tokenizer import (_xgrammar_tokenizer_info,
                                           load_hf_tokenizer)

# stream = torch.cuda.current_stream()
stream = torch.cuda.Stream()

tokenizer = load_hf_tokenizer("nvidia/Llama-3.1-8B-Instruct-FP8")
guided_decoding_config = GuidedDecodingConfig(
    backend=GuidedDecodingConfig.GuidedDecodingBackend.XGRAMMAR,
    **_xgrammar_tokenizer_info(tokenizer),
)
buffer_manager = BufferManager(stream.cuda_stream)
guided_decoder = GuidedDecoder(guided_decoding_config, 1024,
                               tokenizer.tokenizer.vocab_size, DataType.FLOAT,
                               buffer_manager)

torch.cuda.synchronize()
with torch.cuda.stream(stream):
    guided_decoder.logits_bitmask.zero_()
    guided_decoder.logits_bitmask_host.zero_()
    guided_decoder.inc_logits_bitmask_host(CudaStream(stream.cuda_stream))
torch.cuda.synchronize()
print(guided_decoder.logits_bitmask_host)

g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g, stream=stream):
    guided_decoder.logits_bitmask.add_(1)
    guided_decoder.logits_bitmask_host.copy_(guided_decoder.logits_bitmask,
                                             non_blocking=True)
    guided_decoder.inc_logits_bitmask_host(CudaStream(stream.cuda_stream))
    guided_decoder.logits_bitmask.copy_(guided_decoder.logits_bitmask_host,
                                        non_blocking=True)
torch.cuda.synchronize()
print("Graph captured", flush=True)

with torch.cuda.stream(stream):
    for i in range(10):
        g.replay()

torch.cuda.synchronize()
print(guided_decoder.logits_bitmask)
