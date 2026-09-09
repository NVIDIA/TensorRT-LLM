# VisualGen CUDA Graphs

```{note}
This page is an unindexed draft until the VisualGen documentation hub is introduced.
```

- [Overview](#overview)
- [Configuration](#configuration)
- [Capture Scope](#capture-scope)
- [Graph Keys](#graph-keys)
- [Extra Keys](#extra-keys)
- [Capture and Replay](#capture-and-replay)
- [Implementation Notes](#implementation-notes)

## Overview

VisualGen CUDA graphs reduce Python and CUDA launch overhead during the denoising loop by capturing transformer forward calls and replaying them for later denoising steps with compatible inputs.

This is separate from LLM generation-only CUDA graphs and from the PyTorch backend's torch.compile / Piecewise CUDA Graph flow. VisualGen pipelines have their own `CUDAGraphRunner` because diffusion inference has a different execution shape: a fixed denoising loop, model-specific transformer inputs, optional cache acceleration, and optional multi-component pipelines.

## Configuration

CUDA graph capture is controlled by `VisualGenArgs.cuda_graph_config`:

```yaml
cuda_graph_config:
  enable: true
```

The public option is intentionally small. Warmup shapes are configured through `CompilationConfig`, which is shared with torch.compile warmup.

## Capture Scope

VisualGen captures transformer component forward calls, not the full pipeline.

Captured:

- `transformer.forward` for the primary diffusion transformer.
- Additional transformer components such as `transformer_2` when a pipeline has multiple transformer stages.

Not captured:

- Prompt encoding.
- Scheduler step logic.
- VAE decode.
- Python-side denoising-loop control flow.
- TeaCache cache-hit decisions.

TeaCache wraps on top of the CUDA graph wrapper. When TeaCache decides a step must compute, it calls the graphed transformer forward. When TeaCache skips a step, no transformer graph replay is needed for that step.

## Graph Keys

Each captured graph is stored under a graph key. The default key contains tensor shapes from the wrapped `model.forward` arguments and keyword arguments.

A graph key is a tuple of named key parts. Shape key parts are derived by the runner, for example `("hidden_states", (1, 4096, 3072))`. Extra key parts can be registered by the model when a non-shape forward input changes graph-visible behavior.

This separation matters because tensor shapes alone are not always enough. If two forwards have the same shapes but different graph-visible behavior, they must not share a captured graph.

## Extra Keys

Models add non-shape graph-key contributors through `BaseDiffusionModel.register_cuda_graph_extra_key_fns()`. The hook receives the model's `CUDAGraphRunner`; implementations call `runner.register_extra_key_fn(name, fn)` to register a named callback.

During key construction, the runner calls each registered callback with the same `*args` and `**kwargs` passed to the wrapped `model.forward`. If the callback returns a non-`None` hashable value, the runner appends `(name, value)` to the graph key. If it returns `None`, that key part is omitted for the current call.

Use an extra key when a forward input affects captured kernels or control flow but is not already represented by tensor shapes. For example, [Skip Softmax Attention](sparse-attention.md) can require separate CUDA graphs across its dense and sparse phases even when tensor shapes are identical. The base `BaseDiffusionModel` implementation registers a callback through `runner.register_extra_key_fn(...)` for this case.

Subclasses can override `register_cuda_graph_extra_key_fns()` to add model-specific contributors. They should call `super()` unless they intentionally replace the shared registrations. This changes graph capture partitioning without exposing internal CUDA graph keys as public model-forward arguments.

## Capture and Replay

Capture is lazy:

1. On the first call with a new graph key, VisualGen runs a short warmup, captures a CUDA graph, and stores static input buffers for that key.
2. On later calls with the same key, VisualGen copies new tensor values into the static buffers in place and replays the captured graph.
3. On a call with a different key, VisualGen captures a separate graph.

This means the first denoising step for a new shape or phase pays capture cost. Later matching steps replay the graph.

When a pipeline has multiple transformer components that do not execute concurrently, VisualGen can share a CUDA graph memory pool across their runners so graph memory can be reused between components.

## Implementation Notes

The base runner is implemented in `tensorrt_llm/_torch/visual_gen/cuda_graph_runner.py`. Model components can participate by overriding or extending `BaseDiffusionModel.register_cuda_graph_extra_key_fns()` when they need model-specific graph-key behavior.

Most VisualGen models use the base runner, which understands flat tensor arguments. LTX-2 uses a model-specific runner because its transformer forward accepts `Modality` dataclasses and other structured inputs that need custom keying, cloning, and replay copying.

Model authors should add an extra key function or a custom runner whenever non-shape forward inputs affect graph-visible execution. Otherwise, CUDA graph replay may reuse a graph captured for incompatible behavior.
