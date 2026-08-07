# Evaluating Agentic Serving with Trace Replay and Job-Level Metrics

by NVIDIA TensorRT LLM team

## Overview

Agentic applications — coding assistants, deep-research pipelines, tree-structured reasoners — have become a major and fast-growing share of LLM serving traffic, and they stress an inference system in ways chatbot-style traffic never did. Yet the way we evaluate serving systems has largely not followed: performance is still reported on independent requests of fixed shape, while the workload being served is a long-running, multi-turn, tool-invoking, sometimes parallel agent task. This gap raises a practical question for anyone deploying an agent stack: **how do we measure whether a serving system is actually good at agentic workloads?**

Conventional benchmarks issue independent requests with fixed input and output lengths (ISL/OSL). A real agent task instead unfolds as a long-lived *job*: a shared system prompt is reused across many turns, the conversation grows as the agent reasons and invokes tools, sub-agents may run in parallel, and branches synchronize before the job completes. These behaviors — prefix reuse, tool-call gaps, parallel branching — are exactly what govern serving efficiency, and none of them is visible to a fixed-shape benchmark. Nor can such a benchmark answer the question practitioners actually ask: how many agent tasks does a GPU complete per hour?

We take a **trace-and-replay** approach: record each agent run once as a trace, then replay it structure-faithfully against an inference backend as many times as needed — without re-instantiating any tools — and evaluate the result with **job-level metrics** that complement conventional token-level ones. The framework code lives under [`tensorrt_llm/scaffolding/trace_replay/`](https://github.com/NVIDIA/TensorRT-LLM/tree/main/tensorrt_llm/scaffolding/trace_replay), with runnable examples under [`examples/scaffolding/trace_replay/`](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/scaffolding/trace_replay).

Benchmarking practice for agentic scenarios is still taking shape, with several efforts in the industry evolving alongside ours. What follows is what the TensorRT-LLM team has learned while building and using this evaluation pipeline, offered as one set of concrete choices and measurements that others can reuse or argue with.

## Methodology

Single-request benchmarks cannot capture prefix reuse or agent-level throughput; running live agent systems is realistic but hard to reproduce — the full tool environment (sandboxes, browsers, search services) must stay stable under high concurrency, and model nondeterminism makes every run take a different path. Trace replay sidesteps both problems, provided the trace records the prefix structure that drives cache reuse, the tool-call latency of each step, and the parallel-branch topology of the agent.

Figure 1 shows the pipeline. In the trace-collection phase (top), Scaffolding-based agents run agentic task benchmarks with their tools while trace hooks record the stepwise footprint of every run. In the replay-and-evaluation phase (bottom), the replay engine re-issues the recorded requests against the replay backend — the system under evaluation — and we compute agentic serving metrics from the run. The replay backend need not be the system that served the agents during tracing: a trace collected with one model can be replayed against another.

<div align="center">
    <img src="../media/tech_blog27_pipeline.png" alt="The scaffolding trace-replay evaluation pipeline" width="800px">
</div>
<p align="center"><sub><em>Figure 1. The trace-replay evaluation pipeline: a trace-collection phase (top) and a replay-and-evaluation phase (bottom).</em></sub></p>

### Design Choices

Three choices shape what such an evaluation can conclude.

**The unit of measurement is the job, not the request.** Conventional evaluation treats the service as independent, stateless generations summarized by TTFT/TBT, which drops the three properties that dominate agentic serving: execution dependency, so queueing accumulated along an agent's execution graph never surfaces as user-visible delay; tool gaps, so the idle periods that leave a batch underfilled are invisible; and shared prefixes across calls, so prefix-cache residency — the real ceiling for multi-turn serving — is never exercised. We report a Pareto frontier over completed jobs, keeping the token-level view alongside rather than replacing it.

**A trace records structure, not content.** Token counts, prefix identity, tool-call durations, and branch boundaries are enough, because content does not affect serving performance — and a content-free trace can be published, replayed against a different model, and scaled to any concurrency without re-instantiating sandboxes, browsers, or search services. Tool calls replay as timed sleeps of their recorded duration, reproducing request-readiness gaps. **Crucially, the execution graph is recorded too**, so fan-out and its join points replay concurrently instead of being flattened into a sequential chain. That is not a fidelity detail — it changes the answer: a fan-out workload wants a batch size two to four times the session concurrency, a single-branch workload wants it near the session count.

**Coverage must go beyond coding agents.** Coding agents are the highest-volume agentic workload and the easiest to collect, but they are also the most cacheable point in the space — one monotonically growing prefix per session — so calibrating on that shape alone is misleading. We traced four architectures with deliberately different execution patterns, and Table 1 shows how far apart they land: median ISL differs by roughly 8×, median OSL by more than 15×, and the optimal cache hit rate spans 96.5% down to 24.4%. Conclusions do not transfer between them — KV-cache-aware routing lifts the frontier substantially on the coding trace and does nothing for deep research.

### Job-Level Metrics

Inference systems are conventionally compared with token-level Pareto curves (tokens/s/GPU against tokens/s/user). For agentic workloads we complement that with a Pareto curve over whole **jobs**, for two reasons. First, users perceive end-to-end job latency — spanning many model calls, tool gaps, and synchronization points — not per-token rates. Second, token throughput is ambiguous under heavy prefix reuse: on our agentic traces, counting reused prefix tokens reports a per-GPU throughput roughly five times higher than counting only freshly computed tokens, and neither number alone compares systems fairly. A completed job carries no such ambiguity.

The two axes are:

- **Job-level interactivity — jobs/h/user**: 3600 s divided by the mean end-to-end job latency in seconds.
- **Job-level throughput — jobs/h/GPU**: completed jobs per hour, normalized by GPU count.

Because a single agent job runs for minutes, both are measured over a steady-state window: the shared system prompt is preloaded so it is a cache hit from the first call, session starts are staggered with a jittered ramp-up so identical copies do not stay phase-aligned, and a job is credited only if it completes inside the window.

## Trace Format

Each agent run produces one compact JSON file holding an ordered `events` list. Because token content does not affect serving performance, a trace records only structure and sizes, never the underlying text.

Every event is one of three kinds: a **`message`** (one conversation turn, with role, conversation membership, token counts, and — for assistant turns — prompt/completion/reasoning token splits and issued tool calls), a **`tool_call`** (tool name plus measured `duration_ms`), or a **`parallel_start`/`parallel_end`** boundary marking fan-out and synchronization of concurrent branches. A `system_prompt_id` marks which messages share a cacheable prefix, so replay reproduces prefix-cache behavior faithfully.

## Implementation

Tracing attaches to an existing Scaffolding agent through two decorators, with no change to the agent's own logic; the example agents already wire this up behind a flag, so collecting a trace is one CLI switch. Replay then follows a few fixed rules: each call keeps its recorded ISL/OSL but is filled with random token IDs; all replay copies share the same synthetic system-prompt prefix, so prefix caching is exercised rather than bypassed; tool calls become timed sleeps of the recorded duration; and concurrency is created by replaying many copies of the same trace at once. Internally, the `ReplayEngine` runs one queue per branch path so parallel sections and join points run concurrently rather than serialized.

A bundled example trace replays against any running `trtllm-serve` endpoint; the runnable scripts and their usage live in [`examples/scaffolding/trace_replay/`](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/scaffolding/trace_replay).

The framework also includes an offline analyzer that needs no GPU: it walks a trace against an idealized infinite cache and reports the **optimal (upper-bound) KV-cache hit rate**, which we compare against engine-measured hit rates below.

## Trace Dataset and Setup

We collect 730 traces from four Scaffolding agents chosen to cover distinct execution patterns: **Coder**, a single-thread ReAct loop over filesystem/shell tools (500 SWE-bench Verified tasks); **Open Deep Research**, a supervisor that fans out to parallel researcher subagents (100 Deep Research Bench tasks); **IterResearch**, an iterative researcher that keeps its context bounded through compaction (100 Deep Research Bench tasks); and **Tree-of-Thought Research**, a tree-structured reasoner that expands, scores, and prunes parallel branches (30 AIME 2026 problems). For replay we serve Qwen3-235B-A22B-Instruct-2507 through TensorRT-LLM on a single GB200 node (4 GPUs).

The agents occupy clearly different regions of the workload space:

| | **Coder** | **Open Deep Research** | **IterResearch** | **Tree-of-Thought** |
|---|---|---|---|---|
| ISL / request (median) | 14.4k | 6.2k | 6.3k | 1.7k |
| OSL / request (median) | 110 | 632 | 1.7k | 985 |
| Optimal cache hit rate (mean) | 96.5% | 47.8% | 24.4% | 28.8% |

<p align="center"><sub><em>Table 1. Trace-dataset summary. The optimal cache hit rate is the per-trace upper bound on prefix reuse, computed offline from the trace.</em></sub></p>

Coder is input-heavy and decode-light, and caches almost perfectly (96.5%): one shared prefix grows monotonically, so each turn re-reads what previous turns already cached. The research agents invert the shape — moderate inputs, long reasoning-heavy outputs — and cache far worse, because subagent fan-out, context rewriting, and branch exploration repeatedly introduce fresh context. The rest of this blog follows two representative traces at opposite ends of this space: a **Coder** trace (single ReAct thread, 23 requests, only ~3% fresh tokens per prompt) and an **Open Deep Research** trace (supervisor plus four parallel researcher branches, generation-bound, mostly fresh context).

## Experimental Findings

We replay the two representative traces while varying the server maximum batch size **B** and the user concurrency **C** (the number of agent sessions replayed at once).

### Token-Level and Job-Level Metrics Can Disagree

Figure 2 compares three ways of parallelizing the model across the four GPUs — TP4+EP4, DP4+EP4, and DP4+EP4 with KV-cache-aware routing — under the token-level view on fixed-shape inputs (top) and the job-level view on the agentic traces (bottom). On fixed shapes the three strategies nearly coincide; on the agentic traces TP4+EP4 leads along the entire frontier, and KV-cache-aware routing helps only where a large shared prefix exists to route toward: it lifts the frontier substantially on the highly cacheable Coder trace, but shows no visible gain on Open Deep Research, whose parallel researcher branches share little prefix, nor on fixed shapes, where independent requests share none at all. A fixed-shape benchmark would report the strategies as interchangeable and hide this difference entirely.

<div align="center">
    <img src="../media/tech_blog27_token_vs_job_pareto_strategies.png" alt="Token-level Pareto on fixed-shape inputs versus job-level Pareto on agentic traces" width="900px">
</div>
<p align="center"><sub><em>Figure 2. Token-level Pareto on fixed-shape inputs (top) versus job-level Pareto on the agentic traces (bottom), across three parallel strategies.</em></sub></p>

The two views also favor different batch sizes. Figure 3 holds C fixed and sweeps B: token-level interactivity (tokens/s/user) falls monotonically as B grows, because a larger decode batch lengthens each step — yet job-level interactivity (jobs/h/user) *rises* over most of the sweep, peaking at an intermediate B. The latency breakdown (panels c, d) explains why: a larger batch sharply reduces the queue wait that dominates end-to-end latency at small B, and that outweighs the longer decode, so the whole job finishes sooner. Only the job-level view follows the latency a user actually experiences.

<div align="center">
    <img src="../media/tech_blog27_batch_size_sweep_fixed_concurrency.png" alt="Sweeping server batch size at fixed user concurrency" width="900px">
</div>
<p align="center"><sub><em>Figure 3. Sweeping the server batch size at fixed user concurrency. Token-level (a) and job-level (b) interactivity move oppositely; the per-job latency breakdown (c, d) explains why.</em></sub></p>

### Prefix Caching Dominates Multi-Turn Serving

Figure 4 sweeps concurrency (with TP4+EP4 fixed) and annotates each job-level Pareto point with the engine-measured KV-cache hit rate. At low and moderate concurrency the measured rate matches the optimal upper bound computed offline from the trace — about 0.97 for Coder and 0.50 for Open Deep Research — confirming that the idealized per-trace rates are actually realized once the cache can hold the prefixes. As concurrency rises, the KV-cache pool overflows and the hit rate falls off a **KV-cache eviction cliff** (toward 0.73 and 0.19 respectively), and the job-level Pareto degrades in exactly that region. For Coder-style serving, the performance ceiling is set by prefix-cache residency, not raw compute.

<div align="center">
    <img src="../media/tech_blog27_hit_rate_eviction_cliff.png" alt="Measured versus optimal prefix-cache hit rate and the eviction cliff" width="900px">
</div>
<p align="center"><sub><em>Figure 4. Measured versus optimal prefix-cache hit rate, with the job-level Pareto alongside, as concurrency grows (Coder top, Open Deep Research bottom).</em></sub></p>

Host offloading pushes the cliff back. In TensorRT-LLM this is one line in the serving config:

```yaml
# extra-llm-api-config.yml
kv_cache_config:
  host_cache_size: 68719476736   # 64 GiB of host memory for offloaded KV blocks
```

Figure 5 repeats the sweep with host budgets of 0–128 GiB. Evicted prefixes are retained in host memory instead of discarded, so the hit rate stays near optimal at high concurrency — for Coder, from 0.73 back to nearly the 0.97 optimum at the largest budgets — and the job-level Pareto lifts accordingly. The benefit scales with workload reusability: largest for Coder, still clear for Open Deep Research.

<div align="center">
    <img src="../media/tech_blog27_host_offloading.png" alt="Effect of host KV-cache offloading on hit rate and job-level throughput" width="900px">
</div>
<p align="center"><sub><em>Figure 5. Effect of host KV-cache offloading (0 to 128 GiB) on hit rate and job-level throughput (Coder top, Open Deep Research bottom).</em></sub></p>

### Batch Size Should Follow the Agent's Branching Structure

Single-request benchmarks conventionally set C = B and fill every batch slot. Agentic sessions break that correspondence in two opposite ways:

- **Single-branch agents (Coder): stay near B = C.** Tool-call gaps mean fewer than C requests are on the server at any instant, so the batch runs underfilled at C = B (averaging ~51 of 64 slots in our sweep). But raising C above B to fill the batch adds more queuing delay than the underfill costs — especially since CUDA-graph batch padding makes a slightly underfilled batch nearly free. Figure 6 sweeps the full (B, C) grid for the Coder trace: the job-level frontier stays close to the B = C diagonal.
- **Fan-out agents (Open Deep Research): B well above C.** One session issues multiple concurrent requests during its fan-out phase, so the server sees more requests in flight than sessions. Figure 7 repeats the sweep for the Open Deep Research trace, and the contrast with Figure 6 is stark: the frontier sits not on the diagonal but mostly at B = 2C and B = 4C. The branch multiplicity, not the session count, sets the batch size the server needs.

Fan-out also makes the serving behavior harder to reason about in general. A single session no longer maps to a single in-flight request, so the load a server actually sees depends on how many branches are open at each moment — which varies within a job and across agent architectures. The best (B, C) point therefore leaves the diagonal and moves with the branching structure, and the configuration space to search grows accordingly. Finding a good configuration by intuition or by a fixed-shape benchmark is unlikely; it takes a reproducible replay of the real branching structure, which is what the trace-replay framework provides.

<div align="center">
    <img src="../media/tech_blog27_coder_bc_frontier.png" alt="Job-level Pareto frontier over a (B, C) sweep for the Coder trace" width="900px">
</div>
<p align="center"><sub><em>Figure 6. Job-level Pareto frontier over a full (B, C) sweep for the Coder trace, by parallel strategy. The frontier stays close to the B = C diagonal.</em></sub></p>

<div align="center">
    <img src="../media/tech_blog27_odr_bc_frontier.png" alt="Job-level Pareto frontier over a (B, C) sweep for the Open Deep Research trace" width="900px">
</div>
<p align="center"><sub><em>Figure 7. Job-level Pareto frontier over a full (B, C) sweep for the Open Deep Research trace, by parallel strategy. Subagent fan-out pushes the frontier to B = 2C and B = 4C.</em></sub></p>

## Key Takeaways

- **Trace-and-replay makes agentic serving measurable.** Recording each agent run once and replaying it structure-faithfully preserves the behaviors fixed-shape benchmarks miss — prefix reuse, tool-call gaps, and parallel branching — and job-level Pareto metrics report serving performance as completed work: jobs per hour per user and per GPU.
- **Token-level and job-level metrics can disagree** on both the best parallel strategy and the best batch size; only the job-level view follows the end-to-end latency a user actually experiences.
- **Prefix caching sets the ceiling for multi-turn serving.** The hit rate holds at its optimum until the cache overflows, then job-level throughput drops in lockstep with it; host offloading (`kv_cache_config.host_cache_size`) pushes the cliff back.
- **Configure batch size by branching structure**: near B = C for single-branch agents, B several times C for fan-out agents.
- **Fan-out makes serving performance harder to reason about.** A session no longer maps to one in-flight request, so the load the server sees swings with how many branches are open; the best configuration leaves the B = C diagonal and moves with the branching structure, widening the search space and making a good setting hard to find by intuition. This is precisely where reproducible replay pays off.

## Future Work

Agent workloads keep changing, and with them the shapes that reach the serving system: agent architectures evolve, context-management strategies such as compaction alter how much prefix survives across turns, and tool usage shifts the balance between waiting and computing. Any of these can move where the bottleneck sits, and a configuration tuned for today's traces may not hold for tomorrow's.

We will therefore continue to track the workload characteristics of real agentic scenarios — extending the trace dataset as new agent patterns appear and re-measuring the job-level behavior they produce — so that our performance optimizations are driven by what deployments actually run, and land where they matter in the real world.
