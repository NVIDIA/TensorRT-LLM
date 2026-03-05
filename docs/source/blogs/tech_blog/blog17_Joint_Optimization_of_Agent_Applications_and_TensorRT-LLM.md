# Joint Optimization of Agent Applications and TensorRT-LLM

## Overview

In our [previous tech blog](https://nvidia.github.io/TensorRT-LLM/latest/blogs/tech_blog/blog13_Inference_Time_Compute_Implementation_in_TensorRT-LLM.html), we introduced the Scaffolding framework, which provides powerful abstractions for various inference-time compute methods. Orthogonal to inference-time compute, (multi-)agents (e.g., Deep Research, Claude Code, Manus, etc.) have demonstrated tremendous potential to push the frontier of intelligence even further. At the same time, agents represent a new workload that, compared to traditional chatbots, poses new challenges while also opening up new opportunities for LLM serving engines, particularly in areas such as end-to-end performance, fairness, and quality of service (QoS).

In this tech blog, we introduce our latest efforts to apply Scaffolding to (multi-)agent systems. With Scaffolding, we build Open Deep Research, which is a representative multi-agent application for information collection and analysis. Scaffolding's decoupled frontend-backend architecture and modular design enable it to express Open Deep Research's logic concisely and accurately. More importantly, this architecture creates opportunities for joint optimization between agents and LLM inference engines. 

Currently, agents and inference engines operate independently: agents treat the inference engine as a black box, while inference engines remain unaware of the logical structure underlying generation requests from agents. Scaffolding addresses this gap by serving as an intermediate layer between agent application logic and the LLM inference engine. It embeds the semantic information of agents into generation requests, enabling the inference engine to make more informed decisions about request scheduling and KV cache management.

Although several research projects have explored this idea and demonstrated promising results, as a production-grade LLM inference engine, TensorRT-LLM aims to advance this line of work with a more robust, sustainable, and extensible architecture. This vision is guided by two goals from the perspectives of agent applications and the LLM inference engine, respectively:

1. Decouple information collection from the agent's core logic. In other words, ensure that this process remains transparent to agent applications.
2. Ensure that the corresponding optimization policies and mechanisms within the LLM inference engine are as universal and pluggable as possible.

With these two goals in mind, the following sections are organized as follows. First, we introduce how to build complex multi-agent systems using Scaffolding. Then, we describe Scaffolding's information collection mechanism, which addresses Goal 1. Next, we present two optimizations targeting Goal 2, both designed for scenarios where the LLM inference engine serves chatbots and agents simultaneously. The first optimization focuses on batch scheduling: by leveraging hierarchical information about requests, we enable SLO-aware control and performance optimization. The second optimization targets KV cache management: by allowing agents to explicitly release KV cache blocks with low (or zero) reuse probability, we achieve higher KV cache hit rates and reduced TTFT for chatbot applications. Finally, we discuss future directions and open questions.

## Building Multi-Agent Systems with Scaffolding

**Open Deep Research** is an open-source deep research agent built on a multi-agent Planner-Executor architecture. The **Supervisor** serves as the Planner: it accepts user input, generates a research brief, and delegates tasks to the **Researcher**, which functions as an Executor. The Researcher receives a research topic, conducts multiple rounds of interaction with external search tools, then summarizes and compresses the findings before returning results. Once the Supervisor determines that sufficient information has been gathered, it synthesizes everything into a final report.

<div align="center">
    <img src="https://github.com/NVIDIA/TensorRT-LLM/raw/main/docs/source/blogs/media/tech_blog17_open_deep_research_workflow.png" alt="Workflow of Open Deep Research" width="900px">
</div>
<p align="center"><sub><em>Figure 1. Workflow of Open Deep Research</em></sub></p>

The frontend-backend decoupling and modular architecture of Scaffolding support building multi-agent systems efficiently.

From the perspective of frontend-backend decoupling, the frontend encompasses the control flow of the aforementioned Planner-Executor architecture of the Deep Research agent, while the backends primarily consist of the LLM inference engine (for reasoning, content generation, and tool calling decisions) and the MCP server (which handles search tool requests). In terms of implementation, Scaffolding:

- **(Frontend)** implements agent control flow through `Controller`s. `Supervisor` serves as the entry controller for the entire agent and delegates nodes including **ResearchBrief**, **Planning**, and **FinalReport** to `NativeGenerationController`, which is a reusable controller provided by Scaffolding for text generation. The `Supervisor` delegates specific research topics to sub-agents, namely `Researcher`, which is also implemented as a `Controller`. `Researcher` uses a sub-controller called `ChatWithMCPController`, another reusable controller provided by Scaffolding for calling external tools. After multiple rounds of interaction with search tools, `Researcher` also uses `NativeGenerationController` to compress search results and model reflections.
- **(Backend)** serves LLM generation and tool call requests through two `Worker` instances. Scaffolding provides reusable implementations including `OpenaiWorker` (serves LLM generation requests via the OpenAI endpoint), `TRTLLMWorker` (serves LLM generation requests via the TensorRT-LLM API), and `MCPWorker` (serves tool calling requests via MCP server).

From the perspective of modularity, Scaffolding supports the evolution of individual components independent of other components in the multi-agent system. For example, if we want to use a more sophisticated sub-agent to write the final report, we only need to replace the corresponding Controller in that module. Additionally, we can support other LLM endpoints (e.g., Anthropic, Google) using an implementation similar to `OpenaiWorker`.

Below are simplified implementations of the Supervisor and Researcher with Scaffolding.

```python
class Supervisor(Controller):
    def process(self, tasks: List[Task], **kwargs):
        for _ in range(self.max_research_iter):
            
            # Plan the next steps. Note that spawning sub-agents is also
            # done through tool calls.
            yield from self.research_planning_controller.process([research_planning_task])
            
            if research_planning_task.finish_reason != "tool_calls":
                break
            
            # Reflect or spawn sub-agents
            research_tasks_list = [make_research_task(tool_call) for tool_call in research_planning_task.messages[-1].tool_calls:]
            
            yield ParallelProcess(researcher_controllers, research_tasks_list, kwargs_list)
```

```python
class Researcher(Controller):
    tools = [web_search_tool, reflection_tool]

    def process(self, research_tasks: List[ResearchTask], **kwargs):
        # Construct the chat task from the research topic
        chat_task = ChatTask.create_from_prompt(research_tasks[0].research_topic, self.system_prompt, self.tools)

        # Perform multiple rounds of tool calls
        yield from self.chat_with_tools_controller.process([chat_task])

        # Summarize the search results
        yield from self.compress_controller.process([chat_task])

        research_tasks[0].research_result = chat_task.output_str

```

## Capturing Agent Application Information

To jointly optimize agent applications and TensorRT-LLM, we need a way to capture application semantics that matter for performance. Scaffolding provides **Task Collection**, a mechanism designed to decouple information collection from agent logic.

Task Collection works by instrumenting a controller’s `yield` points, since `yield` is the controller’s direct interface for issuing `Task`s to backend workers (e.g., `ChatTask` to `OpenaiWorker`, `MCPTask` to `MCPWorker`). Concretely, a Task Collection is a `TaskCollection` subclass that is attached to a Controller via a decorator. The runtime calls the collection’s `before_yield()` and `after_yield()` hooks immediately before and after each `yield` in the decorated controller’s `process()` method. Each Task Collection can therefore define its own logic for extracting metrics, tracking state, or annotating tasks.

For example, if we want to track token usage within a controller, we can define a Task Collection like this:

```python
class GenerationTokenCounter(TaskCollection):
    def __init__(self):
        super().__init__()
        self.generation_token_count = 0
        self.pre_worker_token_sum = 0

    def before_yield(self, tasks: List[Task]):
        self.pre_worker_token_sum = 0
        for task in tasks:
            if isinstance(task, GenerationTask):
                if task.output_tokens:
                    self.pre_worker_token_sum += len(task.output_tokens)

    def after_yield(self, tasks: List[Task]):
        post_worker_token_sum = 0
        for task in tasks:
            if isinstance(task, GenerationTask):
                if task.output_tokens:
                    post_worker_token_sum += len(task.output_tokens)
        self.generation_token_count += (
            post_worker_token_sum - self.pre_worker_token_sum
        )
```

Next, we can use `GenerationTokenCounter` to implement majority voting that stops once a token budget is reached:

```python
@with_task_collection("token_counter", GenerationTokenCounter)
class TokenBudgetMajorityVoteController(Controller):
    def process(self, tasks: List[Task], **kwargs):
        candidates = []

        # Use GenerationTokenCounter to enforce a total token budget
        while (
            self.task_collections["token_counter"].generation_token_count
            < self.token_budget
        ):
            sample_num = self.sample_num_per_turn
            generation_controllers = [
                self.generation_controller.clone() for _ in range(sample_num)
            ]
            tasks_list = [copy.deepcopy(tasks) for _ in range(sample_num)]
            generation_kwargs_list = [
                copy.deepcopy(kwargs) for _ in range(sample_num)
            ]

            yield ParallelProcess(
                generation_controllers, tasks_list, generation_kwargs_list
            )

            for task_list in tasks_list:
                candidates.extend([task.output_str for task in task_list])

        result = self.majority_vote(candidates, **kwargs)
```

To attach a Task Collection, decorate the controller with `@with_task_collection(name, collection_cls)` and then access it via `self.task_collections[name]`. This not only makes the controller more aware of its own inference behavior, but also enables embedding application-level information into `Task`s so TensorRT-LLM can schedule, batch, and optimize requests more intelligently. Both case studies below use this capture-and-pass mechanism.

## Case Study 1: Proactive KV Cache Drop

Most LLM inference engines can reserve GPU memory to retain the KV cache from prior requests. This avoids repeated prefills when multiple requests share a common prefix, which is a common scenario in multi-turn conversations.

Agent applications are more KV cache-intensive due to several compounding factors. First, the fanout of sub-agents (e.g., Researcher in Open Deep Research) multiplies the KV cache footprint. Second, agent applications typically involve many rounds of tool use, result processing, and reflection (e.g., searching for information, digesting documents returned by tools, and reflecting on those results), making them far more cache-intensive than typical chatbot applications.

As a result, agent applications challenge current LLM inference engines by saturating prefix cache space and interfering with TTFT-sensitive chatbot applications. The root cause is that on-GPU prefix caches are effectively stateless: the inference engine typically does not know whether a prefix will be reused (e.g., whether a conversation has ended) or when it might be extended (e.g., when a user continues later). Consequently, KV cache blocks remain in GPU memory as long as possible and are evicted via LRU policy only when new sequences require space. This leads to suboptimal use of limited on-GPU prefix cache space.

In agentic applications, however, the context lifecycle is often deterministic and controlled by application logic. In other words, the application knows when a conversation or sub-task ends, pauses, or resumes. For example, in Open Deep Research, a "Researcher" sub-agent has a clearly defined exit point. Proactively dropping the KV cache when a conversation ends reduces pressure on both (1) the prefix cache for historical sequences and (2) KV cache blocks needed by incoming requests. This improves prefix-cache retention, particularly in colocated agent-chatbot deployments where agents can otherwise degrade the chatbots' KV cache hit rate.

Scaffolding provides a task collection called `drop_kv_cache_scope` that captures all ChatTasks issued by a controller during its process method. When the application logic completes, Scaffolding emits a corresponding `DropKVCacheTask` for each captured ChatTask to an underlying KV cache hint worker, triggering proactive release of the associated KV cache.

On the TensorRT-LLM side, the executor loop listens for KV cache drop events forwarded from Scaffolding and handles them by calling the KV cache manager's `truncate_blocks` method. The cache is typically truncated rather than fully removed because agentic workflows often include long system prompts that should be preserved. The prefix tree is traversed, and only the exact KV cache blocks specified by the event are dropped.

## Case Study 2: Batch Scheduling

Current LLM inference engines schedule requests in batches using First-Come-First-Serve (FCFS), selecting the earliest-arrived requests. However, this approach has limitations. More effective scheduling would benefit from knowing: 1) which user or agent a request belongs to, and 2) the request's position within the application's control flow graph.

This information enables performance isolation through traffic control across different service types (e.g., chatbots, various agents), ensuring that a burst from one service does not degrade quality of service for others.

To address performance isolation and critical path blocking when co-locating agent and chatbot services, we use Scaffolding to obtain hierarchical information about requests within an agent application's control flow graph. We then implement a workload-aware batching and scheduling strategy in TensorRT-LLM.

The hierarchical information can be represented as a tree. For example, in Open Deep Research, the tree has a depth of 3:

- **Layer 1**: The Supervisor node, which fans out to Layer 2
- **Layer 2**: Researcher nodes, or leaf nodes for requests sent directly by the Supervisor (e.g., Final Report)
- **Layer 3**: Leaf nodes from Researcher nodes (e.g., Search requests)

Each node has properties representing its hierarchical information, including its type, the types of its parent and ancestor nodes, and sequence numbers at each layer.

Scaffolding provides the `sub_request_node` decorator to capture this information:

```python
@sub_request_node("agent_deep_research", is_top_level=True)
class Supervisor(Controller):
```

The first parameter specifies the Controller's node type, while the second indicates whether it is a top-level Controller. The resulting annotation is stored in the `sub_request_markers` list of `ChatTask`, which contains the path from the root node to the leaf node in the hierarchical tree.

The `sub_request_markers` list is passed to TensorRT-LLM along with the chat request. TensorRT-LLM consumes this information in `MicroBatchScheduler` to reorder requests and achieve the desired scheduling behavior. Specifically, we regulate the relative ordering of agent and chatbot requests based on a user-defined ratio, ensuring that a burst from one service type does not affect the SLO of the other.

## Evaluation

### (I) Proactive KV Cache Drop

We evaluate a workload with concurrent Deep Research and multi-turn conversation jobs. Deep Research jobs typically go through an average of three Plan–Execute research iterations before the final report is written. Multi-turn conversations are generated with ISL, OSL, inter-round delay, and number of rounds configured to match the Qwen trace release. The arrival rates of agents and conversations are both 4 jobs/s.

Both the agent workload and the conversational workload are served by Qwen3-235B-A22B-Instruct-2507. Experiments run on 4×GB200 GPUs using TensorRT-LLM with TP=4 and EP=4, and `kv_cache_free_gpu_memory_fraction=0.8`.

Below is the GPU-resident prefix cache hit rate of different types of requests in the context stage before and after enabling proactive KV cache drop. After enabling proactive KV cache drop, the mean KV cache hit rate of **Chatbot** requests increases greatly by 18.76%. In the meantime, the KV cache hit rates of different agent tasks remain consistent. The increase in KV cache hit rate directly leads to the improvement in average TTFT of **Chatbots** from 15.9s to 6.65s. We also noticed the improvement in TTFT of Brief, which can be attributed to lowered KV cache pressure leading to higher allowed concurrency, which is determined by the scheduler based on resource availability.


| **Task Type**    | **Average KV Cache Hit Rate (TRT-LLM w/ Proactive KV Cache Drop)** | **Average KV Cache Hit Rate (TRT-LLM)** | **Average TTFT (TRT-LLM w/ Proactive KV Cache Drop)** | **Average TTFT (TRT-LLM)** |
| ---------------- | ------------------------------------------------------------------ | --------------------------------------- | ----------------------------------------------------- | -------------------------- |
| Brief            | 0.1969                                                             | 0.1969                                  | 6.95                                                  | 14.29                      |
| ResearchPlanning | 0.7224                                                             | 0.7053                                  | 0.75                                                  | 0.46                       |
| ChatWithMCP      | 0.8565                                                             | 0.8489                                  | 0.80                                                  | 0.49                       |
| Compressor       | 0.6811                                                             | 0.7887                                  | 0.72                                                  | 0.42                       |
| FinalReport      | 0.1747                                                             | 0.0476                                  | 0.76                                                  | 0.56                       |
| Chatbot          | **0.4866**                                                         | **0.2990**                              | **6.65**                                              | **15.90**                  |


### (II) Batch Schedule

We evaluate how Scaffolding with agent tree would assist batch scheduling decisions of TensorRT-LLM under colocated agent and chatbot services, to ensure performance isolation between different applications.

We consider a serving instance with 32 concurrent agent jobs and 32 chatbot requests when a burst of 32 agent jobs arrives. To avoid KV cache pressure posing constraints on batch scheduling decisions, we choose Qwen3-30B-A3B with TP/EP=4/4 on 4×GB200 GPUs, which allows abundant GPU memory space for requests.

The average E2E latencies of normal agents, burst agents and chatbots are presented in the table below. 


|              | **TRT-LLM**     | **TRT-LLM w/ AT** |
| ------------ | --------------- | ----------------- |
| Normal Agent | 298.68 ± 19.79s | 344.70 ± 12.93s   |
| Burst Agent  | 367.54 ± 24.61s | 407.12 ± 73.31s   |
| Chatbot      | 80.96 ± 4.49s   | 58.13 ± 0.95s     |


When agent tree is enabled, the E2E latencies of chatbots are reduced by 28.2% and 26.8% respectively. Burst agents will prevail over chatbots, in a joint effort with normal agents, by taking more scheduling opportunities from the LLM serving engine. Agent tree forces the portion of scheduling slots to be 0.5 for agents and chatbots. Therefore, the scheduling opportunities of chatbots remain consistent regardless of the burst traffic of agents. 

The effectiveness of agent tree in achieving performance isolation between chat tasks and agent tasks is demonstrated by the queuing delays during the agent burst period in the following figure. During the agent burst, agent tree reduces the average queuing delay of chat tasks from 18.3s to 1.3s, by 92.6%, to a similar level as prior to the agent burst (avg. 1.2s). Meanwhile, the average queuing delay of agents increases from 6.7s before the burst to 16.4s during the burst when agent tree is enabled, which is within our expectations since agent tree poses constraints on agent traffic in favor of performance isolation for chat tasks. 

<div align="center">
    <img src="https://github.com/NVIDIA/TensorRT-LLM/raw/main/docs/source/blogs/media/tech_blog17_queuing_delays.png" alt="Queuing Delays of Agent and Chatbot Tasks" width="900px">
</div>
<p align="center"><sub><em>Figure 2. Queuing Delays of Agent and Chatbot Tasks </em></sub></p>

We also find that [gang scheduling](https://arxiv.org/abs/2412.20993v2) shows promise for mitigating traffic spikes. By concentrating batch slot resources on requests from a subset of agent jobs, it reduces end-to-end completion times for burst jobs, thereby shortening the burst period. We will explore agent-aware scheduling policies in detail in our next blog post.

## Future Work

We have identified several directions for future work:

**Comprehensive Performance Metrics.** We plan to systematically define performance metrics for agents and, based on common agent patterns, integrate these with the inference engine, including the underlying kernel, to tackle more complex and challenging optimizations.

**Inference-Time Compute Integration.** We aim to explore the combination of agents with inference-time compute methods to identify new performance optimization opportunities.

**Intra-Parallelism Exploration.** We intend to investigate how agents can be combined with intra-parallelism techniques to unlock additional performance gains.

**Community Collaboration.** Finally, we invite partners across the community to join us in using Scaffolding to jointly optimize agent applications and inference engines. This remains a nascent area of research, and we believe many exciting opportunities have yet to be discovered.
