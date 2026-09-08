# Scaffolding

## Introduction

Scaffolding is a framework for inference time compute(aka test-time scaling) with high performance. It makes users easily to integrate various methods(CoT, majority vote, best of N, MCTS) and execution backends(TRTLLM/Openai API/Tools) and also allows users to develop customized feature such as token budget. The core features including:

- Decouple inference time compute method and execution backend. Provides `Controller` concept for users to define the method, `Worker` concept to develop execution backend and `ScaffoldingLlm` to provide API for users to integrate `Controller` and `Worker` and run the request. `Controller` focus on the logic of method and just the yields the `Task`. `Worker` responses to complete the `Task`.

- Make the inference time compute method modular and reusable. An inference time compute method can be composed by multiply modules. For an example, majority vote can sample based on the simple Cot method or the token-budget Cot method such as Dynasor Cot. In scaffolding, `Controller` can be constructed by a series of `Sub-Controllers`, then users can flexibly assemble and replace the `Sub-Controllers`. Back to the example, the majority vote `Controller` can work with any `Sub-Controller` for sampling.

- Provides sufficient concurrency to achieve good performance while ease of use. The key to obtaining good performance is to achieve adequate utilization of resources with concurrency. For examples, LLM inference engine can handle the multiply samples from majority vote `Controller` at the same time, and generation model and reward model can simultaneously run in best-of-n method. However, the asynchronous scheduling required for concurrency often brings difficulties to development. Scaffolding provides three level of concurrency with ease of use. The first level is that the different requests to a `ScaffoldingLlm` instance can be concurrent. The second level is that the multiply `Sub-Controllers` can be concurrent. The third level is that the multiply `Task`s which yielded from `Controller` can be concurrent.

In summary, Scaffolding is not a specific inference time compute method or workflow but a framework that enables more inference time compute methods and backends to be better developed and managed.


## Getting Started

### Install Scaffolding
Now Scaffolding is a module in TensorRT-LLM, so users just need to install TensorRT-LLM.

### Examples
[The first example](../../examples/scaffolding/run_basic_generation.py)
``` bash
python examples/scaffolding/run_basic_generation.py --model_dir PATH/TO/MODEL
```
This example run the generation with TensorRT LLM backend. It shows the step of using Scaffolding. Users firstly need to create `Controller` and `Worker` instance, then map the worker tag to the worker instance, finally create the `ScaffoldingLlm` instance and run the request. It also shows how to run scaffolding on asyncio and run the batched request.

[More examples](../../examples/scaffolding)
These examples shows how to run more complex methods including majority voting and best-of-n, how to static the output tokens with the decorator, how to run the dataset on concurrency and static the results.

### Building a Controller from a config file
A `Controller` can also be described declaratively instead of being constructed in code, which makes a
method easier to share and to migrate between projects.

``` bash
python examples/scaffolding/run_from_config.py \
    --config examples/scaffolding/configs/majority_vote.json \
    --model_dir PATH/TO/MODEL
```

Controllers compose recursively, so a config is a tree. Any object with a `type` key is a controller node
and its `args` are passed to the constructor; nested controller arguments are built first.

``` json
{
  "controller": {
    "type": "MajorityVoteController",
    "args": {
      "default_sample_num": 3,
      "generation_controller": {
        "type": "NativeGenerationController",
        "args": {"sampling_params": {"max_tokens": 1024, "temperature": 0.9}}
      }
    }
  }
}
```

Both JSON and YAML are accepted. `load_controller_config(path)` returns the built `Controller`, and
`build_controller(spec)` does the same for a config already in memory. Controllers that need a live object
rather than a literal, such as the tokenizer of `PRMController`, take it through `{"$ref": "name"}` with the
object passed in as `build_controller(spec, objects={"name": obj})`. Controllers under `contrib` can make
themselves available by calling `register_controller(name, cls)`.

### [Contribute Guide](contrib)


## Future Work
Future work includes the following aspects:

- Support more inference time compute methods and backends. As we mentioned above, Scaffolding is a framework platform for various methods and backend. Now the core feature of Scaffolding has now been completed, community welcome contributors to propose and implement any work that you find valuable including complex reward methods, agentic-based methods, tree-search-based methods, dynamo backend, pytorch backend and so on.

- Provide information for combined performance optimization with backends. Scaffolding can provide some information which is helpful for LLM inference engine. For a example, Controller may aware the prefix relation between generation requests, that would be helpful for kvcache reuse.

- Develop auxiliary components to support generic requirements. Now we have developed some interesting auxiliary components. For an [example](examples/scaffolding/token_budget_majority_vote.py), we developed `GenerationTokenCounter` as a task collection decorator so that Controller could get the output tokens count for itself and its Sub-Controller. There are still many such works waiting for us to do.

You can see more specific work in this [link](https://github.com/NVIDIA/TensorRT-LLM/issues/3706#issuecomment-2820015957).
