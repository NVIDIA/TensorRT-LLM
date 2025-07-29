import os

import tensorrt as trt
import torch
from cuda import cuda, cudart

import tensorrt_llm as tllm
from tensorrt_llm.functional import wan_attention

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # To use GPU 0


def check_cuda_err(err):
    if isinstance(err, cuda.CUresult):
        if err != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError("Cuda Error: {}".format(err))
    if isinstance(err, cudart.cudaError_t):
        if err != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError("Cuda Runtime Error: {}".format(err))
    else:
        raise RuntimeError("Unknown error type: {}".format(err))


def cuda_call(call):
    err, res = call[0], call[1:]
    check_cuda_err(err)
    if len(res) == 1:
        res = res[0]
    return res


logger = trt.Logger(trt.Logger.VERBOSE)

builder = trt.Builder(logger)
config = builder.create_builder_config()
config.set_flag(trt.BuilderFlag.FP16)
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 35)  # 1 MiB

trt_network = builder.create_network(
    1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
profile = builder.create_optimization_profile()

network = tllm.Network()
network._init(trt_network)

tllm._common.set_network(network)

query = tllm.Tensor(name="query",
                    dtype=trt.float16,
                    shape=(1, 79000, 40, 128),
                    network=network,
                    location=trt.TensorLocation.DEVICE)
kv_packed = tllm.Tensor(name="kv_packed",
                        dtype=trt.float16,
                        shape=(1, 79000, 2, 40, 128),
                        network=network,
                        location=trt.TensorLocation.DEVICE)
cu_seqlen_q = tllm.Tensor(name="cu_seqlen_q",
                          dtype=trt.int32,
                          shape=(2, ),
                          network=network,
                          location=trt.TensorLocation.DEVICE)
cu_seqlen_kv = tllm.Tensor(name="cu_seqlen_kv",
                           dtype=trt.int32,
                           shape=(2, ),
                           network=network,
                           location=trt.TensorLocation.DEVICE)

hidden_states = wan_attention(query,
                              kv_packed,
                              cu_seqlen_q,
                              cu_seqlen_kv,
                              num_heads=40,
                              head_size=128,
                              q_scaling=1.0)
hidden_states.mark_output('hidden_states', trt.float16)

serialized_engine = builder.build_serialized_network(network._trt_network,
                                                     config)

with open("wan_attention.engine", "wb") as f:
    f.write(serialized_engine)

with open("wan_attention.engine", "rb") as f, trt.Runtime(logger) as runtime:
    engine = runtime.deserialize_cuda_engine(f.read())
    print(engine)

context = engine.create_execution_context()
print(engine, context)

print(engine[0])
print(engine[1])
print(engine[2])
print(engine[3])
print(engine[4])

stream = torch.cuda.default_stream().cuda_stream
print(stream)

print(context.execute_async_v3)

tensor_dict = {
    'query': 0,
    'kv_packed': 1,
    'cu_seqlen_q': 2,
    'cu_seqlen_kv': 3,
    'hidden_states': 4,
}

query = torch.randn(1, 79000, 40, 128, dtype=torch.float16, device='cuda')
kv_packed = torch.randn(1,
                        79000,
                        2,
                        40,
                        128,
                        dtype=torch.float16,
                        device='cuda')
cu_seqlen_q = torch.tensor([0, 79000], dtype=torch.int32, device='cuda')
cu_seqlen_kv = torch.tensor([0, 79000], dtype=torch.int32, device='cuda')
hidden_states = torch.zeros(1,
                            79000,
                            40,
                            128,
                            dtype=torch.float16,
                            device='cuda')

bindings = [
    query.data_ptr(),
    kv_packed.data_ptr(),
    cu_seqlen_q.data_ptr(),
    cu_seqlen_kv.data_ptr(),
    hidden_states.data_ptr(),
]

print(bindings)

cuda_call(cudart.cudaStreamSynchronize(stream))

for k, v in tensor_dict.items():
    context.set_tensor_address(k, bindings[v])

context.execute_async_v3(stream)

hidden_states_true = torch.nn.functional.scaled_dot_product_attention(
    query.transpose(1, 2), kv_packed[:, :, 0, :, :].transpose(1, 2),
    kv_packed[:, :, 1, :, :].transpose(1, 2))

cuda_call(cudart.cudaStreamSynchronize(stream))
# print(hidden_states_true)
# print(hidden_states)

# print(torch.max(hidden_states_true), torch.min(hidden_states_true))
# print(torch.max(hidden_states), torch.min(hidden_states))

# torch.testing.assert_close(hidden_states.float().transpose(1,2), hidden_states_true.float(), rtol=0.001, atol=0.001)
