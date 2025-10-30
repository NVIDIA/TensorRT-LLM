# Torch Compile & Piecewise CUDA Graph

In this guide, we show how to enable torch.compile and Piecewise CUDA Graph in TensorRT LLM. TensorRT LLM uses torch.compile for lightweight vertical fusion and Piecewise CUDA Graph.

Piecewise CUDA Graph is a technique that runs cudagraph-unsupported components (primarily attention) in eager mode while capturing and replaying the supported parts with CUDA Graph to reduce context-phase launch overhead. We implement this on top of torch.compile because partitioning a model between CUDA Graph and eager execution—and managing graphs in pure eager mode—is cumbersome.

## Table of Contents

- [Torch Compile & Piecewise CUDA Graph](#torch-compile--piecewise-cuda-graph)
  - [Table of Contents](#table-of-contents)
  - [Usage](#usage)
  - [Tips for Piecewise CUDA Graph](#tips-for-piecewise-cuda-graph)
    - [Piecewise CUDA Graph & Generation Only CUDA Graph](#piecewise-cuda-graph--generation-only-cuda-graph)
    - [Piecewise CUDA Graph Padding](#piecewise-cuda-graph-padding)
    - [Performance Tuning](#performance-tuning)
  - [Known Issue](#known-issue)
  - [Development Guide](#development-guide)
    - [Background Knowledge](#background-knowledge)
      - [Custom Op](#custom-op)
      - [Current Status](#current-status)
    - [TensorRT LLM Custom Backend](#tensorrt-llm-custom-backend)
      - [Torch IR Optimization](#torch-ir-optimization)
      - [ATen IR Optimization](#aten-ir-optimization)
        - [Operation Fusion](#operation-fusion)
        - [Re-inplace Optimization](#re-inplace-optimization)
        - [Auto Multi-stream](#auto-multi-stream)
      - [Piecewise CUDA Graph](#piecewise-cuda-graph)
    - [Common Trace Failure](#common-trace-failure)
    - [Graph Break](#graph-break)
    - [Recompilation](#recompilation)

## Usage

To enable torch.compile and Piecewise CUDA Graph, add the following configuration to `extra_config.yml`. Typically, the `extra_config.yml` can be used by adding launching args `--extra_llm_api_options extra_config.yml` to `trtllm-serve` or `trtllm-bench`.

```yaml
... # Other extra config
torch_compile_config:
  capture_num_tokens: '${capture_num_tokens}' # List of num tokens to capture. e.g., [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, ..., 3072]
  enable_userbuffers: false
  enable_piecewise_cuda_graph: true
```

## Tips for Piecewise CUDA Graph

### Piecewise CUDA Graph & Generation Only CUDA Graph

Piecewise CUDA Graph only handles context-only and mixed context+generation iterations, while the generation-only CUDA Graph only handles pure generation iterations. Users need to specify the number of tokens to capture for each type of CUDA Graph separately in the extra config. Currently, the default value for `capture_num_tokens` is `[2**i for i in range(8)] + [i for i in range(256, 3073, 256)]`. However, this configuration should be tuned based on specific hardware, model, and parallel strategy. For guidance on tuning these values, see the [Performance Tuning](#performance-tuning) section below.

```yaml
cuda_graph_config:
  enable_padding: true
  max_batch_size: 1024 # Specify max capture batch size for generation only cuda graph. By default, TensorRT LLM will generate a capture list based on it. 

torch_compile_config:
  capture_num_tokens: '${capture_num_tokens}' # Specify capture_num_tokens for piecewise cuda graph
  enable_userbuffers: false
  enable_piecewise_cuda_graph: true
```

### Piecewise CUDA Graph Padding

Padding means that, at runtime, the token count is padded to the next captured token count. Unlike the generation-only CUDA Graph, padding is mandatory for Piecewise CUDA Graph because context-phase token counts vary widely, making it impractical to capture graphs for every possible length.

### Performance Tuning

Piecewise CUDA Graph uses a token-count–based capture strategy: it captures a CUDA graph for each user-specified token count and, at runtime, selects and replays the graph that matches the iteration’s token count(or can be padded to the next captured token count graph) in a single forward pass.

Piecewise CUDA Graph primarily benefit host-bound iterations in the context phase. Within a single iteration, larger token counts reduce exposure to host-side overhead. However, capturing a broader set of token counts increases GPU memory usage and can reduce achievable concurrency. We recommend manually tuning `capture_num_tokens` to balance latency, memory footprint, and concurrency for your workload.

Guidelines for `capture_num_tokens`:

- Define bounds:
  - Lower bound: base it on typical context lengths. In low-latency workflows with KV-cache reuse, it can be as small as <10 tokens.
  - Upper bound: set by hardware and model configuration—choose the largest token count that still provides a measurable benefit from Piecewise CUDA Graph even after padding. 
- Choose step size: Choose step sizes that balance coverage and memory overhead. Use denser steps in a smaller number of token ranges, and a fixed step (e.g., 256) for larger ranges.
- Manage trade-offs: more capture points reduce padding but increase memory use and can lower max concurrency; fewer points save memory but increase padding and compute cost.

Even with Piecewise CUDA Graph enabled, you may still observe bubbles in the context (prefill) phase, primarily due to the attention operator’s substantial host-side overhead.

## Known Issue

Torch compile cannot work with multi-ModelEngine config. 

1. Speculative Decoding in Two-Model Style

``` yaml
speculative_config:
  decoding_type: "MTP"
  mtp_eagle_one_model: False # Not supported

speculative_config:
  decoding_type: "Eagle"
  eagle3_one_model: False # Not supported
```

2. Multimodal Model Family

## Development Guide

### Background Knowledge

Currently, TRT-LLM mainly relies on torch.compile **fullgraph** mode to enable Piecewise CUDA Graph feature, which means all the operations in the model must be recognized by torch.compile.

#### Custom Op

For ops that cannot be represented by a torch native op, developers need to wrap them into a custom op so that they can work properly with torch.compile. A custom op mainly contains two parts: Op forward implementation & Fake kernel. 

1. Op forward implementation: Define how this op does forward calculation. Including custom CUDA kernel, etc. 
2. Fake kernel: Help torch.compile to do the output tensor dtype/shape inference.

After wrapping the op into a torch custom op, the implementation is a completely **black box** for torch compile. Instead, torch.compile will fully rely on a fake kernel to do the tracing. 

Below is a simple example of flashinfer op’s fake kernel. 

```python
@torch.library.custom_op("trtllm::flashinfer_silu_and_mul", mutates_args=())
def flashinfer_silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    return silu_and_mul(x, enable_pdl=ENABLE_PDL)

@flashinfer_silu_and_mul.register_fake
def _(x: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(x).chunk(2, dim=-1)[1].contiguous()
```

For more examples, please refer to `tensorrt_llm/_torch/custom_ops`.

#### Current Status

For hot models like deepseek/qwen/lllama, we’ve already wrapped some large modules into a custom op to avoid trace failure/graph breaks and exclude output projection & MTP from torch.compile's scope. 

This means developing the inside attention custom op part, the MoE routed export part, and the MPT part don’t need to worry about complex torch.compile constraints since they are treated as a black box for Torch compile. Developers should only make sure the fake kernels of attention custom op, and routed expert are aligned with the actual implementation. 


<div align="center">
<img src="https://github.com/NVIDIA/TensorRT-LLM/raw/main/docs/source/media/current_model_definition_ds.svg" alt="Current Model Status" width=50% height=50% />
</div>
<p align="center"><sub><em>Figure 1. The current model definition for DeepSeek</em></sub></p>

Reasons to wrap attention into a large custom op:

1. The C++ attention op interface is too complex. The argument number exceeds the torch custom op’s limitation
2. MLA has a slice to dispatch the MLA ctx & gen kernel. This introduces dynamic shapes, which may introduce recompilation in the real inference
3. Clear the boundary of attention so that it can be easily recognized by Piecewise CUDA Graph
4. Use some operators that will cause a graph break and are hard to avoid

Reasons to wrap MoE into a large custom op:

1. Use a lot of deepep ops that didn’t wrap into custom ops
2. Hard to support chunked MoE since it uses loops with data-dependent iteration counts, which forces Dynamo to unroll extensively and significantly slows compilation

For the op outside of attention and MLP, the developer should obey the torch.compile constraints. E.g., layernorm, allreduce, etc…

### TensorRT LLM Custom Backend

<div align="center">
<img src="https://github.com/NVIDIA/TensorRT-LLM/raw/main/docs/source/media/custom_backend_overview.svg" alt="Custom Backend Overview"/>
</div>
<p align="center"><sub><em>Figure 2. TensorRT LLM Custom torch.compile Backend Overview</em></sub></p>

Above is the overview of the TensorRT LLM custom backend for `torch.compile`. 

#### Torch IR Optimization

Torch IR is the Fx graph that is directly traced by Torch Dynamo. It has several important features for us to do some graph rewriting and get information:

1. Preserve the operations as is: We can easily find a specific operation and then transform it to arbitrary operations. No need to deal with `auto_functionalize`, etc.
2. Preserve original variable tensor name in the Fx graph: For Piecewise CUDA Graph, it needs to find the correct `SymInt` which represents the token number. Hence, we rely on the `input_ids`'s shape to make it find the `SymInt` correctly. 

#### ATen IR Optimization

We get ATen IR after explicitly calling `aot_module_simplified` on the Fx graph. ATen IR is

1. In SSA format (no input mutations)
2. Strict subset of aten op (<250): In Torch IR, Python native add op, `torch.Tensor().add()`, `torch.aten.add.Tensor` could be three different ops. After the transform, they will be the same op. 
3. Guaranteed metadata information, e.g., dtype and shape propagation

On this IR level, TensorRT LLM will do the following optimization

##### Operation Fusion

All fusions are located in `tensorrt_llm/_torch/compilation/patterns` and implemented using torch.compile’s [pattern matcher](https://docs.pytorch.org/tutorials/intermediate/torch_compile_conv_bn_fuser.html). Unlike the official approach, we write source patterns directly in a lower-level IR instead of relying on tracing. This avoids:

1. Inadequate handling of scalars and lists:
   - Scalars get specialized into the traced pattern, forcing one pattern per value—impractical and non-general.
   - Lists are flattened, turning elements into separate input arguments, making it impossible to match the original operation. 
2. Trace-driven pitfalls: Because it’s trace-based, the generated source patterns may not meet our needs and can introduce additional issues as we expand pattern coverage.

We mainly do the operation fusion for AllReduce & RMSNorm.

1. AllReduce related fusion: Fuse the following operations into one AllReduce op.
   + AllReduce + Residual + RMSNorm
   + AllReduce + Residual + RMSNorm + FP8 Quantization 
   + AllReduce + Residual + RMSNorm + FP4 Quantization
2. AllReduce with User Buffer: Converts AllReduce operations to use userbuffers to avoid extra copy overhead. 

We enable these fusions in torch.compile because they’re difficult to express in eager mode. For the AllReduce + RMSNorm fusion, which is cross-module, implementing it in eager mode would require moving code between modules, leading to redundant, complex, and hard-to-maintain logic.

For user buffers, torch.compile provides a global, flattened view of the model, making it easy for us to manage user buffers.

##### Re-inplace Optimization

Because ATen IR is SSA, in-place operations are rewritten as out-of-place via a mutation wrapper (`auto_functionalize` or `auto_functionalize_v2` ). That wrapper can introduce an extra tensor copy on mutates args. In a TorchInductor pipeline, later passes typically eliminate this copy, but TensorRT LLM relies on custom ops and does not use Inductor. To avoid the redundant overhead, we remove the wrapper ourselves and preserve the intended in-place update.

##### Auto Multi-stream

Currently torch.compile won't create a subgraph for user user-defined CUDA stream. Instead, it will convert it to `set_stream`. The set_stream op doesn't have any consumers, so it will be removed in the Torch IR to ATen IR transformation, thus losing all the multi-stream scheduling. 

To address this, we implemented an auto multi-stream scheduler:

1. Builds a DAG of the FX graph with explicit dependencies, including special handling for in-place ops

2. Computes a critical path using a rough cost model

3. Schedules nodes onto up to `max_num_streams` specified by user config

4. Insert multi-stream related custom op: since the Fx graph executes operators in list order, so we insert streaming-control operators directly into the graph. Moreover, as these operators have no users, we cannot perform dead-code elimination after multi-stream scheduling. Below is an example of multi-stream, which `trtllm.dsv3_router_gemm_op.default` and `trtllm.silu_and_mul.default` + `trtllm.fp4_quantize.default` execute in parallel. 

   ```
   call_function  record_event                             trtllm.record_event                          (1,)                                                                                   {}
   call_function  fp4_quantize_2                           trtllm.fp4_quantize.default                  (mm_1, arg18_1, 16)                                                                    {}
   call_function  getitem_9                                <built-in function getitem>                  (fp4_quantize_2, 0)                                                                    {}
   call_function  getitem_10                               <built-in function getitem>                  (fp4_quantize_2, 1)                                                                    {}
   call_function  nvfp4_gemm_2                             trtllm.nvfp4_gemm.default                    (getitem_9, arg19_1, getitem_10, arg20_1, arg21_1, torch.bfloat16)                     {}
   call_function  permute_2                                aten.permute.default                         (arg17_1, [1, 0])                                                                      {}
   call_function  record_event_1                           trtllm.record_event                          (0,)                                                                                   {}
   call_function  silu_and_mul_1                           trtllm.silu_and_mul.default                  (nvfp4_gemm_2,)                                                                        {}
   call_function  fp4_quantize_3                           trtllm.fp4_quantize.default                  (silu_and_mul_1, arg22_1, 16)                                                          {}
   call_function  getitem_11                               <built-in function getitem>                  (fp4_quantize_3, 0)                                                                    {}
   call_function  record_event_2                           trtllm.record_event                          (4,)                                                                                   {}
   call_function  getitem_12                               <built-in function getitem>                  (fp4_quantize_3, 1)                                                                    {}
   call_function  record_event_3                           trtllm.record_event                          (3,)                                                                                   {}
   call_function  set_stream                               trtllm.set_stream                            (1,)                                                                                   {}
   call_function  wait_event                               trtllm.wait_event                            (0,)                                                                                   {}
   call_function  wait_event_1                             trtllm.wait_event                            (1,)                                                                                   {}
   call_function  dsv3_router_gemm_op                      trtllm.dsv3_router_gemm_op.default           (mm_1, permute_2, None, torch.float32)                                                 {}
   call_function  record_stream                            trtllm.record_stream                         (permute_2, 1)                                                                         {}
   call_function  record_stream_1                          trtllm.record_stream                         (mm_1, 1)                                                                              {}
   call_function  record_event_4                           trtllm.record_event                          (2,)                                                                                   {}
   call_function  set_stream_1                             trtllm.set_stream                            (0,)                                                                                   {}
   call_function  wait_event_2                             trtllm.wait_event                            (2,)                        
   ```

#### Piecewise CUDA Graph

We implement Piecewise CUDA Graph execution on top of torch.compile: non-capturable regions run in eager mode, while the rest of the model is captured and replayed as CUDA Graph segments.

In the current design, we assume the attention block is the only non-capturable component. To maintain stable input pointers across segment boundaries, we convert attention to an in-place variant. Instead of allocating its own output, attention writes results into a tensor preallocated by the preceding CUDA Graph segment. This guarantees that each segment’s inputs are allocated by CUDA Graph and, therefore, stable for that segment’s capture.

<div align="center">
<img src="https://github.com/NVIDIA/TensorRT-LLM/raw/main/docs/source/media/piecewise_runner.svg" alt="Piecewise Runner" width=35% height=35% />
</div>
<p align="center"><sub><em>Figure 3. Piecewise Runner</em></sub></p>

Notes:

1. Attention **MUST NOT** have any output. The output tensor should be allocated by CUDA Graph. 
2. Each sub-cudagraph **MUST** have at least one input tensor that contains the number of tokens in the shape. 
3. Only allow dynamic shape for `num_of_tokens` dim. 

### Common Trace Failure

1. Custom op fake kernel: For every custom op, developers must implement a correct fake kernel. **Make sure to update the corresponding fake kernel when the custom op is changed**
2. Dynamic Iteration Number Loop: This is technically not a trace failure, but it will introduce long-time tracing that is generally not acceptable. When torch.compile tries to convert PyTorch modeling code to Fx graph, it will try to unroll the loop. For a loop that has a large and dynamic loop number with a large loop body, the tracing process will take a long time to do the unrolling. 
   1. If the IO of the loop can be easily written into a custom op format, try to replace it with a custom op
   2. If the loop num is unchanged during the whole inference service lifetime, then it is ok to leave the loop as is. (e.g., Model decoder layer loop)

### Graph Break

1. Use unsupported operators

   + python native operators: `print`, `sys.intern()`, etc.
   + pybind/nanobind operators
     + **Solution:** Wrap them to torch's custom op. For complex operators like attention that exceed the argument limit of PyTorch’s custom-op interface, wrap them in a higher-level module to reduce the argument count.
   + Some of the torch operators:
     + `torch.nonzeros()`: Produce data-dependent dynamic shape tensor
     + `torch.sym_min`: `SymInt` aware min
     + `torch.Tensor.tolist()`, `torch.Tensor.item()`
     + **Solution:** Use them inside a custom op if these operators don't get involved in producing the custom op's output tensor. 

2. Use a custom object’s method: For a class like mapping config, we cannot directly use its method like has_pp() in the model forward. 

   + **Solution**: We should convert it to a bool in the model init and use the bool. 

   ```python
   class Mapping(object):
       def __init__(self, ...):
           ...
         
       def has_pp(self): # Cannot use this method in torch.compile
           return self.pp_size > 1
   ```

3. Data Dependent Control(DDC) flow involved in code

   + **Solution**: Try to avoid DDC in the code. Try to pre-compute the result outside of torch.compile's scope. For the following example, try to pre-compute the `torch.sum(data)` at the data preparation stage, and pass the result to the `forward`. 

   ```python
   class TestCase(torch.nn.Module):
       def __init__(self):
           super().__init__()
   
    def forward(self, x, data):
        y = x ** 2
        if torch.sum(data) >= 4: # Data Dependent Control Here!
            t =  y
        else:
            t = y / 2
        t = t + 10
        return t
   
   test_case = TestCase()
   test_case = torch.compile(test_case, backend=Backend())
   x = torch.randn(5).cuda()
   data = torch.ones(2, dtype=torch.int32)
   data[0] = 2
   data[1] = 2
   test_case(x, data)
   ```

### Recompilation

1. Try not to use data-dependent dynamic shapes in the model forward. (e.g., slice the tensor based on input value). This will introduce 0/1 specialization to the model and will possibly introduce recompile. 

   1. **0/1 specialization**: torch.compile will recompile the model if a dynamic tensor’s dim equals 0 or 1. In the worst case, it will recompile 3 times for 1 dimension: 0,1, >2

2. For an int argument that would change during runtime, use `SymInt` rather than int in the C++ custom op definition. Otherwise, it will trigger a recompile when the value changes. 

   ```c++
   TORCH_LIBRARY_FRAGMENT(trtllm, m)
   {    
       m.def("allgather(Tensor input, SymInt[]? sizes, int[] group) -> Tensor");
       m.def("allgather_list(Tensor[] input_list, SymInt[]? sizes, int[] group) -> Tensor[]");
   }
   ```

3. Some recompiles that are hard to aware:

   1. python native `min(list)`, `max(list)`: it will recompile when the list elements changes

   2. Control Flow based on dynamic shape

   3. Next power of two: Previously, we used `bit_length()` to implement the next power of 2 function. However, it will cause a recompile for every int value. Now rewrite the code to be torch.compile-friendly. 

      ```python
      def next_positive_power_of_2(x: int) -> int:
          if x < 1:
              return 1
      
          # Following code is equivalent to 1 << (x - 1).bit_length()
          # But this impl does not contain bit_length(), so it can be used by torch compile.
          # It can correctly handle 64-bit numbers, which should be enough for now.
          n = x - 1
          n |= n >> 1
          n |= n >> 2
          n |= n >> 4
          n |= n >> 8
          n |= n >> 16
          n |= n >> 32
          return n + 1
      ```

      
