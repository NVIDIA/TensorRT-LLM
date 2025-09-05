# New XQA-kernel provides 2.4x more Llama-70B throughput within the same latency budget

XQA kernel provides optimization for [MQA](https://arxiv.org/abs/1911.02150) and [GQA](https://arxiv.org/abs/2305.13245v3) during generation phase. It also provides optimization for beam search. Using tensor cores for acceleration, reducing data loading and conversion, it delivers increased throughput within the same latency budget. Increased throughput allows serving greater number of user requests while providing the same experience.

Support matrix and usage flags are described in [docs/source/advanced/gpt_attention](/docs/source/advanced/gpt-attention.md#xqa-optimization).

**Increased Throughput:**
Looking at the Throughput-Latency curves below, we see that the enabling of XQA optimization increases throughput. Higher throughput equates to serving more users, and we can see that TPOT on the Y-axis flattens out when XQA gets enabled.


<img src="https://github.com/NVIDIA/TensorRT-LLM/blob/rel/docs/source/blogs/media/XQA_ThroughputvsLatency.png?raw=true" alt="XQA increased throughput within same latency budget" width="950" height="auto">

<sub>Preliminary measured Performance, subject to change. TPOT lower is better. FP8, 8xH100 GPUs, Single Engine, ISL/OSL: 512/2048, BS: 1 - 256, TensorRT-LLM v0.8a</sub>


## Llama-70B on H200 up to 2.4x increased throughput with XQA within same latency budget


**H200 2.4x with XQA**


|Model     |GPUs | Input Length | Output Length | Throughput w/o XQA (tok/s/GPU) | Throughput w/ XQA (tok/s/GPU) | Speedup |
|:---------|:----|:-------------|:--------------|:-------------------|:------------------|:--------|
|Llama-70B |   1 |          128 |          2048 |              1,227 |             2,941 | 2.4x
|          |   8 |          128 |          2048 |             13,232 |            25,300 | 1.9x


###### Closing

These improvements will be published in the `main` branch soon, and will be
included in the v0.8 releases.

For more information about H200, please see the [H200 announcement blog](./H200launch.md).

Throughput is calculated as output tokens per second per gpu.
`out_tps=output_seqlen*batch_size/total_latency/tp`

<sub> **Glossary:**
| DP  = Data Parallel
  ISL = Input Sequence Length
| PP  = Pipeline Parallel
| OSL = Output Sequence Length
| OOM = Out of Memory
| TP  = Tensor Parallel <sub/>
