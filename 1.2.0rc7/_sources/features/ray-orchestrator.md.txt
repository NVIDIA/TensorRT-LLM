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
- Generate text asynchronously (refer to [llm_inference_async_ray.py](/examples/ray_orchestrator/llm_inference_async_ray.py))
- Multi-node inference (refer to [multi-node README](/examples/ray_orchestrator/multi_nodes/README.md))
- Disaggregated serving (refer to [disagg README](/examples/ray_orchestrator/disaggregated/README.md))

*Initial testing has been focused on LLaMA and DeepSeek variants. Please open an Issue if you encounter problems with other models so we can prioritize support.*

## Roadmap
- Performance optimization
- Integration with RLHF frameworks, such as [Verl](https://github.com/volcengine/verl) and [NVIDIA NeMo-RL](https://github.com/NVIDIA-NeMo/RL).

## Architecture
This feature introduces new classes such as [RayExecutor](/tensorrt_llm/executor/ray_executor.py) and [RayGPUWorker](/tensorrt_llm/executor/ray_gpu_worker.py) for Ray actor lifecycle management and distributed inference. In Ray mode, collective ops run on [torch.distributed](https://docs.pytorch.org/tutorials/beginner/dist_overview.html) without MPI. We welcome contributions to improve and extend this support.

![Ray orchestrator architecture](/docs/source/media/ray_orchestrator_architecture.jpg)
