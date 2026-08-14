<div align="center">

# TensorRT-LLM with Ray orchestrator

</div>

<div align="left">

This folder contains examples for a prototype **Ray orchestrator** that supports on-demand LLM instance spin-up and flexible GPU placement across single- and multi-node inference. It’s a first step toward making TensorRT-LLM a better fit for Reinforcement learning from human feedback (RLHF) workflows. For RLHF, [Ray](https://docs.ray.io/en/latest/index.html) — unlike MPI’s fixed world size and placement — can dynamically spawn and reconnect distributed inference actors, each with its own parallelism strategy.

This feature is a prototype and under active development. MPI remains the default.


## Quick Start
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
### Available
- Generate text asynchronously (refer to [llm_inference_async_ray.py](llm_inference_async_ray.py))
- Multi-node inference (refer to [multi-node README](./multi_nodes/README.md))
- Disaggregated serving (refer to [disagg README](./disaggregated/README.md))

*Initial testing has been focused on LLaMA and DeepSeek variants. Please open an Issue if you encounter problems with other models so we can prioritize support.*

### Upcoming
- Performance optimization
- Integration with RLHF frameworks, such as [Verl](https://github.com/volcengine/verl) and [NVIDIA Nemo-RL](https://github.com/NVIDIA-NeMo/RL).

## Architecture
This feature introduces new classes such as [RayExecutor](/tensorrt_llm/executor/ray_executor.py) and [RayGPUWorker](/tensorrt_llm/executor/ray_gpu_worker.py) for Ray actor lifecycle management and distributed inference. In Ray mode, collective ops run on [torch.distributed](https://docs.pytorch.org/tutorials/beginner/dist_overview.html) without MPI. We welcome contributions to improve and extend this support.

![Ray orchestrator architecture](/docs/source/media/ray_orchestrator_architecture.jpg)


## Disclaimer
The code a prototype and subject to change. Currently, there are no guarantees regarding functionality, performance, or stability.

</div>
