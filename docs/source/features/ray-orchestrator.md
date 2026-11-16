# Ray Orchestrator (Prototype)

```{note}
This project is under active development and currently in a prototype stage. The current focus is on core functionality, with performance optimization coming soon. While we strive for correctness, there are currently no guarantees regarding functionality, stability, or reliability.
```

## Motivation
The **Ray orchestrator** uses [Ray](https://docs.ray.io/en/latest/index.html) instead of MPI to manage workers for single- and multi-node inference. It’s a first step toward making TensorRT-LLM a better fit for Reinforcement Learning from Human Feedback (RLHF) workflows. For RLHF, Ray can dynamically spawn and reconnect distributed inference actors, each with its own parallelism strategy. This feature is a prototype and under active development. MPI remains the default in TensorRT-LLM.


## Basic Usage
To use Ray orchestrator, you need to first install Ray.
```shell
cd examples/ray_orchestrator
pip install -r requirements.txt
```

To run a simple `TP=2` example with a Hugging Face model:

```shell
python llm_inference_distributed_ray.py
```

This example is the same as in `/examples/llm-api`, with the only change being `orchestrator_type="ray"` on `LLM()`. Other examples can be adapted similarly by toggling this flag.


## Features
Currently available:
- Generate text asynchronously (refer to [llm_inference_async_ray.py](source:examples/ray_orchestrator/llm_inference_async_ray.py))
- Multi-node inference (refer to [multi-node README](source:examples/ray_orchestrator/multi_nodes/README.md))
- Disaggregated serving (refer to [disagg README](source:examples/ray_orchestrator/disaggregated/README.md))

*Initial testing has been focused on LLaMA and DeepSeek variants. Please open an Issue if you encounter problems with other models so we can prioritize support.*

## Roadmap
- Performance optimization
- Integration with RLHF frameworks, such as [Verl](https://github.com/volcengine/verl) and [NVIDIA NeMo-RL](https://github.com/NVIDIA-NeMo/RL).

## Architecture
This feature introduces new classes such as [RayExecutor](source:tensorrt_llm/executor/ray/executor.py) and [RayGPUWorker](source:tensorrt_llm/executor/ray/gpu_worker.py) for Ray actor lifecycle management and distributed inference. In Ray mode, collective ops run on [torch.distributed](https://docs.pytorch.org/tutorials/beginner/dist_overview.html) without MPI. We welcome contributions to improve and extend this support.

![Ray orchestrator architecture](https://github.com/NVIDIA/TensorRT-LLM/raw/main/docs/source/media/ray_orchestrator_architecture.jpg)

## Environment Variables

| Variable | Default | Meaning |
|---|---|---|
| `TLLM_DISABLE_MPI` | unset | `1` runs without MPI. The Ray orchestrator sets it on its workers. |
| `TLLM_RAY_FORCE_LOCAL_CLUSTER` | `0` | `1` always starts a new local Ray cluster instead of attaching to an existing one. |
| `TRTLLM_RAY_PER_WORKER_GPUS` | `1.0` | GPUs reserved for each worker actor. A fraction lets several actors share one GPU (colocated use). `ray_placement_config.per_worker_gpu_share` takes precedence. |
| `TRTLLM_RAY_BUNDLE_INDICES` | unset | Comma-separated placement-group bundle indices, one per rank (unique, in range). Must be set from inside a Ray placement group. `ray_placement_config` takes precedence. |
| `RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES`, `RAY_DEDUP_LOGS` | set by the orchestrator (`1`, `0`) | Keep Ray from restricting `CUDA_VISIBLE_DEVICES` per worker and from collapsing duplicate log lines across ranks. Export them yourself before `ray start` on a manually launched cluster, as the [multi-node example](source:examples/ray_orchestrator/multi_nodes/run_cluster.sh) does. |
| `TLLM_CUDA_OBJECT_COLLECTIVES` | `1` | Without MPI, small Python-object collectives (attention-DP votes, response gathers) are staged through CUDA tensors on dedicated NCCL communicators. `0` falls back to the gloo CPU backend. Must be identical on every rank. |
| `TLLM_REFIT_DEDUP_MODULES` | `1` | During RL weight refit, each module's refit hook runs once even if a `torch.compile` wrapper exposes it twice. `0` disables this, for bisecting. |
