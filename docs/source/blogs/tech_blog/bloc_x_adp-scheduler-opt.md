---
title: "ADP Scheduler Optimization"
---

# Motivation && Background

In DeepSeek MLA + MOE architectures under max-throughput scenarios, an Attention DP + MOE EP parallelization strategy is commonly used to avoid redundant KV Cache. However, in some cases, such as to avoid the system complexity and high maintenance costs of a dis-aggregated architecture, for scenarios with short input sequence length (ISL) and long output sequence length (OSL), or for offline inference, In-Flight Batching inference is still preferred. This can lead to load imbalance issues among different ranks in the Attention module, which slows down the Attention computation. For example, in a given iteration, some ranks might be processing context (prefill) while others are doing generation (decoding). This significantly impacts the overall system throughput.

Therefore, we need a balanced scheduler to dispatch different requests to different DP ranks.

# Theoretical Analysis and Modeling
Optimization Goals:
- Minimize the load difference between different GPU ranks.

# Theoretical Analysis
Here, we will model and quantify the negative impact of DP Attention. Since the load across ranks can be unequal, the time taken by the Attention part in any given iteration is determined by the rank with the longest execution time. The formula is as follows:
$$
    iter\_time_i = \max_{0 \leq m < N} iter\_time_{im}
$$
where $iter\_time_{im}$ represents the execution time of the m-th rank in the i-th iteration, and N is the `dp_size`.

To quantify load balance, we define two metrics: `balance_ratio` and `sol_tps` (speed-of-light throughput). The `balance_ratio` represents the load balance of the Attention part in the current iteration. (The load of the MoE part is unknown during the early scheduling phase, and is primarily handled by the EPLB module). With non-extremely-long context scenarios，we can assume that the overall latency is dominated by the non-MHA portion; hence the time complexity is effectively O(N).
Since scheduling is intended to resolve the imbalance across ranks, the theoretical upper-bound throughput (`sol_tps`) can be computed as follows:

$$
    balance\_ratio = avg\_tokens / max\_tokens
$$
where $avg\_tokens = {\sum_{i=1}^N tokens_i} / {N}$ and $max\_tokens = \max(tokens_i)$, with $tokens_i$ being the number of tokens on $rank_i$.

`sol_tps` represents the theoretical upper-bound throughput.
$$
    sol\_time = \sum_{i=0}^{\infty} (iter\_time_i / balance\_ratio)
$$

$$
    sol\_tps = elp\_time / sol\_time * tps
$$

Where:
- `iter_time_i` denotes the elapsed time of the i-th scheduling iteration.
- `elp_time` denotes the empirically measured end-to-end elapsed time.
- `tps` denotes the actual throughput in tokens per second.
- `sol_tps` represents the theoretical upper-bound throughput.


## System Modeling and Optimization Method
In any given iteration, the Attention module can face significant differences in the number of tokens among ranks. The total execution time for this part is bound by the rank with the longest processing time.

### Baseline
One approach is to sort the requests in the request queue by `num_tokens` and then dispatch them to different ranks in a round-robin fashion, as shown in the figure below. This method balances the total number of tokens across ranks from a global perspective, and only can reduce the disparity in token numbers between different contexts when all ranks are processing context requests.

<div align="center">
<figure>
  <img src="./../media/ADP Balance.png">
</figure>
</div>
<p align="center"><sub><em>Figure 1: Balance number of context request tokens across ranks through sort and round-robin </em></sub></p>

However, from a local (per-iteration) perspective, it cannot guarantee token balance. For example, in `iteration_i`, `rank_i` might be processing a new context, while other ranks are still in the generation phase. In this case, the Attention time will be completely dominated by the context processing time.

### ADP Balance Method
To address this issue, we propose an **ADP Balance strategy based on a waiting mechanism**. The intuition is that when some ranks have available slots for context requests, instead of scheduling them immediately, we wait for a certain number of iterations until other ranks also have a similar number of context requests to process. This reduces the overhead caused by imbalance.

Specifically, we introduce two parameters: `time_out_iters` and `batching_wait_iters`.
- `time_out_iters`: The maximum number of steps a rank will wait when it has a context request while others do not.
- `batching_wait_iters`: The maximum number of batch iterations to wait to equalize the number of context batches.

For example, assume all ranks have contexts of the same length, and each rank has M requests. Let the context phase duration be `time(ctx)` and the generation phase duration be `time(gen)`. Starting from iteration `i`, for N iterations, one new request arrives per iteration. With the original method, one rank processes a context in each iteration, making the per-iteration time `time(ctx)`. The total time would be `time(ctx) * N`. The `balance_ratio` would be `(ctx_len + (M-1) + M * (N - 1)) / (N * ctx_len)`. Here, `gij` represents the j-th generation request on rank i, and `*Cj*` represents the j-th context request.

```
iter_i:     [*C0*, g01, ..., g0M], [g10, g11, ..., g1M], ..., [gN0, gN1, ..., gnM]
iter_i+1:   [g00, g01, ..., g0M], [*C1*, g11, ..., g1M], ..., [gN0, gN1, ..., gnM]
...
iter_i+N-1: [g00, g01, ..., g0M], [g10, g11, ..., g1M], ..., [*CN*, gN1, ..., gNM]
```

If we use our ADP balance method, all ranks can wait for the first N-1 iterations. In the N-th iteration, when the number of context requests equals the number of ranks, they are scheduled together. In this scenario, the execution time for the first N-1 iterations is `time(gen)`, and the last iteration takes `time(ctx)`. The total time is `time(gen) * (N-1) + time(ctx)`. Here, the `balance_ratio=1`, achieving perfect balance in every step. The state of each rank per iteration is as follows:

```
iter_i:     [g00, g01, ..., g0N], [g10, g11, ..., g1N], ..., [gN0, gN1, ..., gnN]
iter_i+1:   [g00, g01, ..., g0N], [g10, g11, ..., g1N], ..., [gN0, gN1, ..., gnN]
...
iter_i+N-1: [*C0*, g01, ..., g0M], [*C1*, g11, ..., g1M], ..., [*CN*, gN1, ..., gNM]
```

Given that the context phase duration is much longer than the generation phase duration, the time saved is `(time(ctx) - time(gen)) * (N-1)`. This shows that optimizing for balance can significantly improve execution time and throughput.

Limitation: This method increases the time-to-first-token (TTFT) and is suitable for scenarios where TTFT is not a critical requirement.

# Experiments
## Dataset

The dataset contains 16,000 requests, with `avg_input_length=803.1` and `avg_output_length=3653.8`. The distribution of input and output token lengths is shown in the figure below.

<div align="center">
<figure>
  <img src="./../media/combined_token_distribution.png">
</figure>
</div>
<p align="center"><sub><em>Figure 2: Token distribution of input and output token numbers </em></sub></p>

The input and output token ranges are wide, and the output tokens have a clear long-tail distribution. This makes it difficult to co-schedule as many contexts as possible within the same step while avoiding additional bubbles, posing a significant challenge to our scheduling strategy.

## Hyperparameter Setting

model=DeepSeekV3-NVFP4,
TP=8, EP=8, WideEP=ON, max_num_tokens = 4678, max_batch_size=512, concurrency=5120, kv_cache_frac=0.85
streaming_interval=1024, context_chunking=OFF

## Experimental Results
The following experiments are based on the settings above. First is the baseline, without the ADP balance waiting mechanism.

### Baseline

tps: 25664, avg_balance_ratio=54.11%, sol_tps=39552. The curves for avg_token and balance_ratio over iterations are shown below.

<div align="center">
<figure>
  <img src="./../media/si2_ei30235_plot1_token_distribution.png">
</figure>
</div>
<p align="center"><sub><em>Figure 3: avg_tokens and balance_ratio by iterations </em></sub></p>

Most of the imbalance occurs within the first 12,000 iterations. After this, all requests have completed their context phase and are in the generation phase, so the imbalance here has little impact on throughput. The detailed graph for this interval is shown below. Within this range, the Theoretical Performance Improvement is 70.23%.

<div align="center">
<figure>
  <img src="./../media/si100_ei12000_plot1_token_distribution.png">
</figure>
</div>
<p align="center"><sub><em>Figure 4: avg_tokens and balance_ratio by iterations [100, 12000] </em></sub></p>

## ADP Balance
With ADP Balance, the output TPS reaches 34140.
This is achieved by adding `timeout_iters=50` and `batch_waiting_iters=10` in `config.yaml` to balance ADP tokens through waiting.

config.yaml
```
attention_dp_config:
    enable_balance: true
    batching_wait_iters: 10
    timeout_iters: 50
```

<div align="center">
<figure>
  <img src="./../media/balanced_si100_ei12000_plot1_token_distribution.png">
</figure>
</div>
<p align="center"><sub><em>Figure 5: avg_tokens and balance_ratio by iterations [100, 12000] with ADP Balance</em></sub></p>

Negative impact: The number of steps increases due to the waiting mechanism.

## Pareto Curve

The following Figure shows the pareto curve of impact of different `timeout_iters` and `batch_waiting_iters` values on TPS.

<div align="center">
<figure>
  <img src="./../media/combined_charts.png">
</figure>
</div>
<p align="center"><sub><em>Figure 6: Pareto Curve </em></sub></p>


[1] baseline: log_gb200_ifb/ifb_edp8_lbs512_mt4608_con5120_nreq16000_streamterval1024_wideep_balance_ep_20250810_764586
[2] ADP balance: to50bw10: log_gb200_ifb/ifb_edp8_lbs512_mt4608_con5120_nreq16000_streamterval1024_wideep_adpbalance_to50_bw10_20250812_766150
