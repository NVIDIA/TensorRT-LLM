# Inference Time Compute Implementation in TensorRT LLM

By NVIDIA TensorRT LLM Team and UCSD Hao AI Lab

## Table of Contents
- [Inference-Time Compute Implementation in TensorRT LLM (Part 1: Design and Implementation）](#inference-time-compute-implementation-in-tensorrt-llm)
  - [Table of Content](#table-of-content)
  - [Background and Motivation](#background-and-motivation)
  - [Introduction for Scaffolding: A Framework for inference-time compute](#introduction-for-scaffolding)
    - [Core Features](#scaffolding-core-feature)
    - [Architecture](#scaffolding-architecture)
      - [Worker](#scaffolding-architecture-worker)
      - [Controller](#scaffolding-architecture-controller)
      - [ScaffoldingLlm](#scaffolding-architecture-scaffoldingllm)
  - [An Example: Implement Dynasor on Scaffolding](#example-for-scaffolding)
    - [Introduction for Dynasor](#dynasor-introduction)
    - [Implement Dynasor-CoT in Scaffolding](#dynasor-cot-implement-in-scaffolding)
    - [Implement Dynasor-CoT based Majority Voting in Scaffolding](#dynasor-cot-based-majority-vote-in-scaffolding)
    - [Acknowledgements](#dynasor-acknowledgements)
    - [Reference](#dynasor-reference)
  - [Feature List on Scaffolding](#scaffolding-feature-list)
  - [Future Work](#scaffolding-future-work)


## Background and Motivation
Inference-time compute, also known as test-time scaling, is increasingly important. Beyond simply increasing output length, workflows such as best-of-N and Monte Carlo Tree Search (MCTS) offer additional capabilities for optimizing inference. Further, most of the workflows of agentic or multi-agent are logically similar to these methods of inference-time compute, except that they use more complex tools and context engineering. However, how to conveniently define these methods while achieving excellent inference performance has become a new problem. Because good performance requires careful asynchronous scheduling, but writing asynchronous scheduling programs is not easy for algorithm engineers. When considering the use of external tools and token budget management, the problem becomes even more complex.


LLM inference frameworks such as TensorRT LLM,vLLM and SGLang provide high performance for inference of generation models or reward models, but they are only for single request inference. Popular Agent frameworks such as LangChain and Dify focus on enabling users to develop agents as simply as possible. But precisely because of this, they may have difficulty completing many inference-time compute methods that require precise definition and developments.


So we want to build a good framework to support users in exploring and deploying more inference-time compute methods. It should provide a modular infrastructure and fill the gap in balancing usability and performance for inference-time compute.


## Introduction for Scaffolding: A Framework for inference-time compute

`Scaffolding` is a framework for inference-time compute with high performance. It makes it easy for users to integrate various methods (CoT, majority vote, best of N, MCTS) and execution backends (TRTLLM/Openai API/Tools) and also allows users to develop customized features such as token budget. 


### Core Features
The core features including:


Decouple inference-time compute method and execution backend. Provides `Controller` concept for users to define the method, `Worker` concept to develop execution backend and `ScaffoldingLlm` to provide API for users to integrate `Controller` and `Worker` and run the request. 


Make the inference-time compute method modular and reusable. An inference time compute method can be
composed of multiple modules. In scaffolding, `Controller` can be constructed by a series of `Sub-Controllers`, then users can flexibly assemble and replace the `Sub-Controllers`.


Provides sufficient concurrency to achieve good performance while ease of use. Concurrency is the key for performance. `Scaffolding` provides three levels of concurrency. The first level is that the different requests to a `ScaffoldingLlm` instance can be concurrent. The second level is that the multiple `Sub-Controllers` can be concurrent.The third level is that the multiply Tasks which yielded from `Controller` can be concurrent.


### Architecture
`Scaffolding` consists of three core components. Let's first briefly introduce these components. The `Worker` class is the backend that execute a single task, such as sending an inference request to an LLM inference framework or service, or completing a call to an external tool. The `Controller` class focuses on defining the workflow of a inference-time compute method. The `ScaffoldingLlm` class is responsible for integrating the two and completing the entire task.


This is the call sequence diagram of `Scaffolding`:
<div align="center">
    <img src="https://github.com/NVIDIA/TensorRT-LLM/raw/main/docs/source/blogs/media/tech_blog13_scaffolding_sequence.png" alt="Scaffolding Sequence" width="900px">
</div>
<p align="center"><sub><em>Figure 1. Scaffolding Sequence</em></sub></p>

Here we can focus on two points. First, `ScaffoldingLlm` provides users with the interface. Second, the `Controller` does not directly call the Worker.


Next, we will introduce the code of the core components.


#### Worker
```python
class Worker(ABC):

    async def run_task(self, task: Task) -> TaskStatus:
        worker_cls = type(self)
        if type(task) not in worker_cls.task_handlers:
            return TaskStatus.WORKER_NOT_SUPPORTED
        return await worker_cls.task_handlers[type(task)](self, task)

    task_handlers = {}
```
The core interface of `Worker` is `run_task()`, which accepts a `Task`, executes it, and writes the result to the appropriate field. It should be noted that `run_task()` is an asynchronous function and it can be concurrently and asynchronously called with python asyncio.


#### Controller
```python
class Controller(ABC):

    def __init__(self):
        self.task_collections = {}

    def clone(self):
        return copy.deepcopy(self)

    def generate(self, prompt: str, **kwargs) -> GenerationResult:
        task = GenerationTask.create_from_prompt(prompt)

        yield from self.process([task], **kwargs)

        return task.create_scaffolding_output()

    def process(self, tasks: List[Task], **kwargs):
        raise NotImplementedError
```
Its two core interfaces are `generate()` and `process()`. `generate()` is the entry point for `ScaffoldingLlm` to invoke. In the default implementation of `generate()`, it produces a `Task` and then invokes `process()`. The `process()` is the most important part of every `Contronller` class, as it defines the implementation the workflow of this inference-time compute method.


Let's go into a specific subclass of `Controller` to see how `process()` is implemented. 
```python
class NativeGenerationController(Controller):

    class WorkerTag(Enum):
        GENERATION = "generation"

    def process(self, tasks: List[Task], **kwargs):
        for task in tasks:
            task.worker_tag = self.WorkerTag.GENERATION
            for key, value in self.sampling_params.items():
                if getattr(task, key) is None:
                    setattr(task, key, value)
            task.streaming = self.streaming

        yield tasks
```
Essentially, `process()` is an iterator in python that can return a list of tasks using yield statement. When the iterator is re-entered, that is, when the yield statement ends, the `Tasks` have been completed. That means the result of the `Task` has been written into its result field. Then the `process()` can proceed to the next steps.


From here we can see that the implement of the `Controller` can focus on the design of the workflow. It does not directly call the `Worker` and does not need to care about how these tasks are completed. And that is how `Scaffolding` decouple inference-time compute method and execution backend.


Also, `Controller` makes the inference-time compute method modular and reusable. It only requires the `sub-Controller` to be a member of class, and then the `process()` function of the `sub-Controller` is called using the “yield from” statement.
```python
yield from self.reward_controller.process(generation_tasks,
                                                **reward_kwargs)
```


For the concurrency with ease of use, `Controller` provides two ways. As the code above shows, the yield statement yield a list of `Task`. So the first one is that the multiple Tasks in a yield statement is executed in parallel. The second way is for the multiple `sub-Controller` which can be executed in parallel. `Controller` provides syntactic sugar called `ParallelProcess`.
```python
generation_controllers = [
            self.generation_controller for _ in range(sample_num)
        ]
        generation_kwargs_list = [generation_kwargs for _ in range(sample_num)]
        generation_tasks = [copy.deepcopy(task) for _ in range(sample_num)]

        yield ParallelProcess(generation_controllers,
                              [[t] for t in generation_tasks],
                              generation_kwargs_list)
```


#### ScaffoldingLlm
With `Controller` and `Worker`, we still need something that can combine them together, that is the `ScaffoldingLlm` class.
```python
llm_worker = TRTLLMWorker.init_with_new_llm(
    args.model_dir,
    backend="pytorch",
    max_batch_size=32,
    max_num_tokens=4096,
)

prototype_controller = NativeGenerationController(sampling_params={
    "temperature": 0.9,
    "max_tokens": 1024,
})

llm = ScaffoldingLlm(
    prototype_controller,
    {NativeGenerationController.WorkerTag.GENERATION: llm_worker},
)
results = llm.generate(prompts)
```
Users need to first create instances of `Worker` and `Controller`, and map them by `WorkerTag` to create the `ScaffoldingLlm` class. Then call the generate interface of `ScaffoldingLlm` to get the final result. 


`ScaffoldingLlm` also provides async interface.
```python
async for result in llm.generate_async(prompt):
    print(">>>", result.outputs[0].text)
```
Therefore, an instance of ScaffoldingLlm supports concurrent execution of multiple requests.


Let's make a summary of the overall implementation of `Scaffolding`. If users want to implement a new inference-time compute method, users can develop a new `Controller`. They can also call some existing `Controllers` as its `sub-Controller`. If users want to implement a new backend, users can either create a new `Worker` or add a new `Task` handler to an existing `Worker`.  As for `ScaffoldingLlM`, we have hidden many complex implementations, such as async scheduling within `ScaffoldingLlM`, and users do not need to modify the code of `ScaffoldingLlM`.


## An Example: Implement Dynasor-CoT on Scaffolding
Dynasor-CoT 
<a href="https://arxiv.org/abs/2412.20993">
  <img src="https://img.shields.io/badge/arXiv-2412.20993-b31b1b.svg?style=plastic" alt="arXiv" style="vertical-align: text-top;">
</a>
is a certainty-based, training-free approach to accelerate Chain-of-Thought (CoT) inference. This chapter discusses how inference-time compute methods can be smoothly integrated into the TRT-LLM Scaffolding framework, using Dynasor-CoT as an example.

<div align="center">
    <img src="https://github.com/NVIDIA/TensorRT-LLM/raw/main/docs/source/blogs/media/tech_blog13_dynasor_demo.gif" alt="Dynasor Demo" width="900px">
</div>
<p align="center"><sub><em>Figure 2. Demo of DeepSeek-R1-Distill-Qwen-7B achieving a 5.74x speedup compared to the baseline when using Dynasor-CoT on MATH500</em></sub></p>

### Introduction for Dynasor-CoT
#### Motivation of Dynasor-CoT
LLM reasoning is highly token-inefficient, often requiring far more tokens to achieve the same accuracy as non-reasoning models. A major source of this inefficiency is that reasoning models tend to **self-doubt**; they often reach the correct answer early but then engage in extended verification behaviors like double-checking and reassessment.

For instance, Figure 2 compares a traditional Qwen-7B model with a reasoning-focused, Deepseek-distilled Qwen-7B model on a simple question. While the traditional model reaches its answer in 180 tokens, the reasoning model expends 1,000 tokens on iterative verification, despite having already found the correct answer at token 340. This represents a significant waste of tokens for diminishing returns on accuracy.

<div align="center">
    <img src="https://github.com/NVIDIA/TensorRT-LLM/raw/main/docs/source/blogs/media/tech_blog13_dynasor_hesitation.png" alt="Motivation" width="900px">
</div>
<p align="center"><sub><em>Figure 2. An example answer from reasoning model (Deepseek-distilled Qwen-2.5 7B) vs traditional model (Qwen-2.5 7B) on one of the problem in MATH500 dataset.</em></sub></p>

#### The "Probe" technique
Dynasor-CoT uses a **"Probe-In-The-Middle"** (or "probe" for short) technique, which prompts reasoning models to output early-stage results during intermediate steps of reasoning. Imagine you're in a math exam working on a hard problem. When time is up, you're forced to write down your final answer, regardless of how confident you are.

More specifically, a probe is an extra generation request with an eliciting prompt appended to the intermediate reasoning tokens. One effective eliciting prompt is: `Oh, I suddenly got the answer to the whole problem, Final Answer: boxed{`. Figure 3 shows an analysis comparing the accuracy of directly asking versus probing the model. Taking AMC23 as an example, reasoning models frequently arrive at correct answers early (median: 830 tokens) but continue generating unnecessary tokens due to self-doubt (median: 2.7K tokens).


<div align="center">
    <img src="https://github.com/NVIDIA/TensorRT-LLM/raw/main/docs/source/blogs/media/tech_blog13_dynasor_pressure_testing.png" alt="Dynasor Demo" width="900px">
</div>
<p align="center"><sub><em>Figure 3. DeepSeek-R1's performance on AMC23 and AIME24 at varying token budgets. (Left) Standard reasoning with late answer outputs. (Right) Early answer extraction using the Probe-In-The-Middle technique, demonstrating equivalent accuracy with a 50% token reduction. The greener regions in the right panels suggest the model knows the answers much earlier than it reveals in standard reasoning.</em></sub></p>

#### How it speeds up inference
Instead of generating a fixed number of tokens or waiting for a stop token, Dynasor-CoT **probes the model regularly** (e.g., every 32, 64, or 128 tokens) and **terminates the process** early once a consistent answer is formed across recent probes. This avoids unnecessary computation, directly reducing latency.

Figure 4 provides an illustration:

* **Case 1**: All three probe requests yield the same answer, "3159.", indicating high certainty. The process can exit early.

* **Case 2**: Early-stage answers are inconsistent, indicating low confidence, so generation continues.

* **Case 3**: The model generates special tokens such as "wait" or "hmm," signaling hesitation; generation continues.

<div align="center">
    <img src="https://github.com/NVIDIA/TensorRT-LLM/raw/main/docs/source/blogs/media/tech_blog13_dynasor_illustration.jpg" alt="Dynasor Illustration" width="900px">
</div>
<p align="center"><sub><em>Figure 4. Illustration of Dynasor-CoT. Case 1: early exit due to consistent early-stage results. Case 2: continue generation due to inconsistent early-stage results. Case 3: responses containing hesitation words (e.g., wait) are discarded.</em></sub></p>

### Implement Dynasor-CoT in Scaffolding
A key difference between inference-time compute methods like Dynasor-CoT and a normal LLM generation request is that the generation process can consist of multiple smaller, user-defined tasks. The results of these tasks can dynamically control the overall logic—for example, by determining whether to expand the scope of subsequent generation or to terminate the process entirely. In a single Dynasor-CoT request, generation proceeds chunk by chunk, with additional "probe" tasks running in parallel with the main generation. Once a consistent answer is formed across recent probes, the process terminates early.

`Scaffolding` provides a good solution for customizing these kinds of data flows. Within a `Controller`, we can customize the data flow logic by defining how and when these smaller tasks are submitted. To implement Dynasor-CoT, we simply inherit from the base `Controller` class and override the `process()` function to customize how it yields tasks. We don't need to worry about how these tasks are executed because the inference-time compute methods and the execution backend are modularized and decoupled in Scaffolding. These tasks are submitted to `ScaffoldingLlm`, which then dispatches workers to complete them.

Let's start the implementation by inheriting the `Controller` class and adding the necessary parameters for Dynasor-CoT.
```python
class DynasorGenerationController(Controller):

    class WorkerTag(Enum):
        GENERATION = "generation_with_dynasor_cot"

    def __init__(
        self,
        generation_dir,
        max_tokens=8192,
        certainty_threshold=3,
        chunk_size=64,
        streaming=False,
    ):
        super().__init__()
        self.generation_dir = generation_dir
        self.max_tokens = max_tokens
        self.certainty_threshold = certainty_threshold
        self.chunk_size = chunk_size
        self.uncertain_words = ["wait", "hold", "but", "okay", "no", "hmm"]
        self.probe_suffix = "... Oh, I suddenly got the answer to the whole problem, **Final Answer**\n\n\\[ \\boxed{"
        self.answer_suffix = "\n\n... Oh, I have got the answer to the whole problem\n**Final Answer:**\n\\[\n \\boxed{"
        self.answer_suffix_with_marker = "\n\n...</think>\n Oh, I have got the answer to the whole problem\n**Final Answer:**\n\\[\n \\boxed{"
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.generation_dir,
            legacy=False,
            padding_side='left',
            truncation_side='left',
            trust_remote_code=False,
            use_fast=True,
        )
        self.streaming = streaming
```

The `process()` function, as mentioned before, is the core method within the `Controller` class. Here, we can customize our data flow by specifying the logic for yielding tasks. For Dynasor-CoT, we have two different kinds of tasks:

1. `proposer_task`: Handles the main content generation, producing self.chunk_size tokens based on the previous content.

2. `probe_task`: Elicits an early-stage answer by generating 20 tokens from the same content.

The code below creates these two types of tasks.

```python
    def process(self, tasks: List[GenerationTask], **kwargs):
        # Start with the initial prompt provided by the first task.
        initial_prompt = tasks[0].input_str

        proposer_task = GenerationTask()
        proposer_task.max_tokens = self.chunk_size
        proposer_task.temperature = 0.6
        proposer_task.top_p = 0.95
        proposer_task.worker_tag = self.WorkerTag.GENERATION

        probe_task = GenerationTask()
        probe_task.max_tokens = 20
        probe_task.temperature = 0.6
        probe_task.top_p = 0.95
        probe_task.worker_tag = self.WorkerTag.GENERATION

        probe_answers = []
        probe_responses = []

        initial_prompt_token_num = len(
            self.tokenizer.encode(initial_prompt, add_special_tokens=False))
        probe_suffix_token_num = len(
            self.tokenizer.encode(self.probe_suffix, add_special_tokens=False))

        current_prompt = initial_prompt
```

To prevent extra latency, the `proposer_task` should not be blocked by the `probe_task`. Scaffolding's task-level concurrency handles this perfectly. We can yield `proposer_task` and `probe_task` in a single list. Multiple tasks yielded together in the same list will be batched and executed in parallel.

```python
    yield[proposer_task, probe_task]
```

In the following `for` loop, each iteration performs these steps:

1. **Submit** both a proposer task and a probe task by yielding them. We don't need to worry about execution details, as they are handled by `ScaffoldingLlm`, which binds the `Controller` and `Workers` together behind the scenes.

2. **Evaluate** the probe response after the tasks return, checking for consistency over several rounds (using `certainty_threshold`).

3. **Finalize** the answer and return if it is consistent. Otherwise, append the new tokens from the proposer task and proceed to the next iteration.

```python
        # Iterate over generation rounds until the maximum tokens limit is reached.
        for _ in range(initial_prompt_token_num + probe_suffix_token_num,
                    self.max_tokens, self.chunk_size):
            proposer_task.input_str = current_prompt
            probe_task.input_str = current_prompt + self.probe_suffix

            # For the probe task, append the suffix to force a chain-of-thought leading to an answer.
            yield [proposer_task, probe_task]

            # Retrieve the output from the probe task.
            probe_text = probe_task.output_str

            # Extract the potential answer from the probe response.
            answer = self.obtain_answer(probe_text)
            probe_answers.append(answer)
            probe_responses.append(probe_text)

            if self.should_early_stop(probe_answers, probe_responses):
                tasks[0].result = probe_task.result
                # If the current prompt indicates the chain-of-thought phase has ended, use one type of suffix.
                if "</think>" in current_prompt:
                    tasks[0].output_str = (current_prompt + self.answer_suffix +
                                        probe_answers[-1] + "}\n\\]")
                    return
                else:
                    # Otherwise, use the suffix with marker to transition clearly.
                    tasks[0].output_str = (current_prompt +
                                        self.answer_suffix_with_marker +
                                        probe_answers[-1] + "}\n\\]")
                    return

            # If the answer is not deemed confident, perform another round of generation.
            # Append the newly generated text from the proposer to the current prompt for the next iteration.
            current_prompt += proposer_task.output_str

        # If the maximum token limit is reached without satisfying the certainty condition,
        # output the accumulated prompt as the final output.
        tasks[0].result = proposer_task.result
        tasks[0].output_str = current_prompt
        return
```
The `probe_task` can utilize prefix kvcache reuse to enhance inference performance. TensorRT LLM enables the kvcache of an in-progress request to be reused by other requests, so `probe_task` can `proposer_task`'s kvcache even though the `proposer_task` is in a continuous running state.

Now we have implemented a `Controller` for Dynasor-CoT. Here is an example of how to use it:
```python
dynasor_generation_controller = DynasorGenerationController(
    # Parameters for DynasorGenerationController
    )

llm = ScaffoldingLlm(
    prototype_controller=dynasor_generation_controller, 
    # other parameters for ScaffoldingLLM
    )
results = llm.generate(prompts)
```

### Implement Dynasor-CoT based Majority Voting in Scaffolding
Scaffolding is designed to be modular and reusable. We can assemble methods just like LEGO building blocks. For instance, to implement Dynasor-CoT-based Majority Voting, we can simply stack our `DynasorGenerationController` with a `MajorityVoteController`.

Once a controller for majority voting is built, no further implementation is needed. We can directly stack the two controllers as shown below.
```python
dynasor_generation_controller = DynasorGenerationController(
    # Parameters for DynasorGenerationController
    )

majority_vote_controller = MajorityVoteController(
    generation_controller=dynasor_generation_controller, # stack here
    # Other parameters for MajorityVoteController
    )

llm = ScaffoldingLlm(
    prototype_controller=majority_vote_controller, # Expose the outermost controller to ScaffoldingLlm
    # other parameters for ScaffoldingLLM
    )
results = llm.generate(prompts)
```


### Acknowledgements
This work demonstrates an outstanding example of cross-team collaboration between the TensorRT LLM and UCSD Hao AI Lab. We sincerely appreciate the support from everyone who contributed to making this happen.


### Reference
[1] Y. Fu*, J. Chen*, Y. Zhuang, Z. Fu, I. Stoica, and H. Zhang, "Dynasor: More Efficient Chain-of-Thought Through Certainty Probing," Hao-AI-Lab Blog, Feb. 16, 2025. [Online]. Available: https://hao-ai-lab.github.io/blogs/dynasor-cot/


## Feature List on Scaffolding
You can customize your own `Controller`, `Worker` and `Task`, however, we have provided a foundational set with commonly used functionality that you can use.


`Worker`: TensorRT LLM, OpenaiAPI, MCP;


`Task`: Generation, Reward, ToolCall;


`Controller`: MajorityVote, PRMReward, BestOfN, MCTS;


## Future Work
The future work is divided into two parts.


The first part is to enable `Scaffolding` to support more inference-time compute methods, especially the methods of agentic and multi-agent system. 


The second part is that we hope to find more opportunities to optimize TensorRT LLM based on `Scaffolding` workloads. For examples, in terms of kvcache prefix reuse, `Scaffolding` can identify which parts are system prompts, which parts are likely to be reused in the subsequent requests of the agent task, and which parts cannot be reused and can be evicted immediately.


Finally, what we want to emphasize is that we welcome and look forward to more people joining our open source community. You can find these issues in the [TensorRT LLM GitHub issues with Scaffolding tag](https://github.com/NVIDIA/TensorRT-LLM/issues?q=state%3Aopen%20label%3AScaffolding).
