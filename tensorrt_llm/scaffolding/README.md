# Scaffolding

## Introduction

Scaffolding is a framework to run various inference time compute methods such as CoT, majority vote, best of N, MCTS, etc.

Scaffolding is built around three key principles:
- Ease of Use: Users can easily run inference time compute methods and enginners can easily customize the framework to add new methods and execution backends.
- Modularity: Scaffolding is designed to be modular. Engineers can fully reuse existing modules when defining new methods.
- Performance: Scaffolding is designed to be performant. It considers the design of concurrent schedule and provides more information to the backend to help optimize performance.

## Architecture
There are following key components in Scaffolding:

- `Controller`: The class that defines the workflow of inference time compute methods.
- `Worker`: The class that handles the operations such as generation and reward scoring.
- `ScaffoldingLlm`: The interface class to run inference time compute methods.

Workflow of Scaffolding:
1. Users instantiate a `ScaffoldingLlm` instance by assembling a `Controller` and some `Worker`s.
2. Users call `ScaffoldingLlm`'s API to run inference time compute methods.
3. `ScaffoldingLlm` instantiate a `Controller` instance and get `Task` from `Controller`.
4. `ScaffoldingLlm` dispatch the `Task` to `Worker` and return the completed `Task` back to `Controller`.
5. `Controller` create new `Task` until the inference time compute method is finished.
6. `ScaffoldingLlm` return the result to users.


## Usage

See [example/scaffolding](example/scaffolding).

## Future Work
- support openai api worker（on the way）
- support reward model （on the way）
- performance benchmark （on the way）
- support best of N
- support MCTS
- support sandbox
