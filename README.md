<div align="center">

TensorRT-LLM
===========================
<h4> A TensorRT Toolbox for Optimized Large Language Model Inference</h4>

[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg?style=flat)](https://nvidia.github.io/TensorRT-LLM/)
[![python](https://img.shields.io/badge/python-3.12-green)](https://www.python.org/downloads/release/python-3123/)
[![python](https://img.shields.io/badge/python-3.10-green)](https://www.python.org/downloads/release/python-31012/)
[![cuda](https://img.shields.io/badge/cuda-12.8.0-green)](https://developer.nvidia.com/cuda-downloads)
[![trt](https://img.shields.io/badge/TRT-10.8.0-green)](https://developer.nvidia.com/tensorrt)
[![version](https://img.shields.io/badge/release-0.19.0.dev-green)](./tensorrt_llm/version.py)
[![license](https://img.shields.io/badge/license-Apache%202-blue)](./LICENSE)

[Architecture](./docs/source/torch/arch_overview.md)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[Performance](./docs/source/performance/perf-overview.md)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[Examples](https://nvidia.github.io/TensorRT-LLM/quick-start-guide.html)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[Documentation](./docs/source/)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[Roadmap](https://docs.google.com/presentation/d/1gycPmtdh7uUcH6laOvW65Dbp9F1McUkGDIcAyjicBZs/edit?usp=sharing)

---
<div align="left">

## Latest News
* [03/22] TensorRT-LLM is now fully open-source, with developments moved to GitHub!
* [03/18]  🚀🚀 NVIDIA Blackwell Delivers World-Record DeepSeek-R1 Inference Performance with TensorRT-LLM [➡️ Link](https://developer.nvidia.com/blog/nvidia-blackwell-delivers-world-record-deepseek-r1-inference-performance/)
* [02/28] 🌟 NAVER Place Optimizes SLM-Based Vertical Services with TensorRT-LLM [➡️ Link](https://developer.nvidia.com/blog/spotlight-naver-place-optimizes-slm-based-vertical-services-with-nvidia-tensorrt-llm/)

* [02/25] 🌟 DeepSeek-R1 performance now optimized for Blackwell [➡️ Link](https://huggingface.co/nvidia/DeepSeek-R1-FP4)

* [02/20] Explore the complete guide to achieve great accuracy, high throughput, and low latency at the lowest cost for your business [here](https://www.nvidia.com/en-us/solutions/ai/inference/balancing-cost-latency-and-performance-ebook/?ncid=so-twit-348956&linkId=100000341423615).

* [02/18] Unlock #LLM inference with auto-scaling on @AWS EKS ✨ [➡️ link](https://aws.amazon.com/blogs/hpc/scaling-your-llm-inference-workloads-multi-node-deployment-with-tensorrt-llm-and-triton-on-amazon-eks/)

* [02/12] 🦸⚡ Automating GPU Kernel Generation with DeepSeek-R1 and Inference Time Scaling
[➡️ link](https://developer.nvidia.com/blog/automating-gpu-kernel-generation-with-deepseek-r1-and-inference-time-scaling/?ncid=so-twit-997075&linkId=100000338909937)

* [02/12] 🌟 How Scaling Laws Drive Smarter, More Powerful AI
[➡️ link](https://blogs.nvidia.com/blog/ai-scaling-laws/?ncid=so-link-889273&linkId=100000338837832)

* [01/25] Nvidia moves AI focus to inference cost, efficiency [➡️ link](https://www.fierceelectronics.com/ai/nvidia-moves-ai-focus-inference-cost-efficiency?linkId=100000332985606)

* [01/24] 🏎️ Optimize AI Inference Performance with NVIDIA Full-Stack Solutions [➡️ link](https://developer.nvidia.com/blog/optimize-ai-inference-performance-with-nvidia-full-stack-solutions/?ncid=so-twit-400810&linkId=100000332621049)

* [01/23] 🚀 Fast, Low-Cost Inference Offers Key to Profitable AI [➡️ link](https://blogs.nvidia.com/blog/ai-inference-platform/?ncid=so-twit-693236-vt04&linkId=100000332307804)

* [01/16] Introducing New KV Cache Reuse Optimizations in TensorRT-LLM [➡️ link](https://developer.nvidia.com/blog/introducing-new-kv-cache-reuse-optimizations-in-nvidia-tensorrt-llm/?ncid=so-twit-363876&linkId=100000330323229)

* [01/14] 📣 Bing's Transition to LLM/SLM Models: Optimizing Search with TensorRT-LLM [➡️ link](https://blogs.bing.com/search-quality-insights/December-2024/Bing-s-Transition-to-LLM-SLM-Models-Optimizing-Search-with-TensorRT-LLM)

* [01/04] ⚡Boost Llama 3.3 70B Inference Throughput 3x with TensorRT-LLM Speculative Decoding
[➡️ link](https://developer.nvidia.com/blog/boost-llama-3-3-70b-inference-throughput-3x-with-nvidia-tensorrt-llm-speculative-decoding/)

<details close>
<summary>Previous News</summary>

* [2024/12/10] ⚡ Llama 3.3 70B from AI at Meta is accelerated by TensorRT-LLM. 🌟 State-of-the-art model on par with Llama 3.1 405B for reasoning, math, instruction following and tool use. Explore the preview
[➡️ link](https://build.nvidia.com/meta/llama-3_3-70b-instruct)

* [2024/12/03] 🌟 Boost your AI inference throughput by up to 3.6x.  We now support speculative decoding and tripling token throughput with our NVIDIA TensorRT-LLM. Perfect for your generative AI apps.  ⚡Learn how in this technical deep dive
[➡️ link](https://nvda.ws/3ZCZTzD)

* [2024/12/02] Working on deploying ONNX models for performance-critical applications? Try our NVIDIA Nsight Deep Learning Designer ⚡ A user-friendly GUI and tight integration with NVIDIA TensorRT that offers:
✅ Intuitive visualization of ONNX model graphs
✅ Quick tweaking of model architecture and parameters
✅ Detailed performance profiling with either ORT or TensorRT
✅ Easy building of TensorRT engines
[➡️ link](https://developer.nvidia.com/nsight-dl-designer?ncid=so-link-485689&linkId=100000315016072)

* [2024/11/26] 📣 Introducing TensorRT-LLM for Jetson AGX Orin, making it even easier to deploy on Jetson AGX Orin with initial support in JetPack 6.1 via the v0.12.0-jetson branch of the TensorRT-LLM repo. ✅ Pre-compiled TensorRT-LLM wheels & containers for easy integration ✅ Comprehensive guides & docs to get you started
[➡️ link](https://forums.developer.nvidia.com/t/tensorrt-llm-for-jetson/313227?linkId=100000312718869)

* [2024/11/21] NVIDIA TensorRT-LLM Multiblock Attention Boosts Throughput by More Than 3x for Long Sequence Lengths on NVIDIA HGX H200
[➡️ link](https://developer.nvidia.com/blog/nvidia-tensorrt-llm-multiblock-attention-boosts-throughput-by-more-than-3x-for-long-sequence-lengths-on-nvidia-hgx-h200/)

* [2024/11/19] Llama 3.2 Full-Stack Optimizations Unlock High Performance on NVIDIA GPUs
[➡️ link](https://developer.nvidia.com/blog/llama-3-2-full-stack-optimizations-unlock-high-performance-on-nvidia-gpus/?ncid=so-link-721194)

* [2024/11/09] 🚀🚀🚀 3x Faster AllReduce with NVSwitch and TensorRT-LLM MultiShot
[➡️ link](https://developer.nvidia.com/blog/3x-faster-allreduce-with-nvswitch-and-tensorrt-llm-multishot/)

* [2024/11/09] ✨ NVIDIA advances the AI ecosystem with the AI model of LG AI Research 🙌
[➡️ link](https://blogs.nvidia.co.kr/blog/nvidia-lg-ai-research/)

* [2024/11/02] 🌟🌟🌟 NVIDIA and LlamaIndex Developer Contest
🙌 Enter for a chance to win prizes including an NVIDIA® GeForce RTX™ 4080 SUPER GPU, DLI credits, and more🙌
[➡️ link](https://developer.nvidia.com/llamaindex-developer-contest)

* [2024/10/28] 🏎️🏎️🏎️ NVIDIA GH200 Superchip Accelerates Inference by 2x in Multiturn Interactions with Llama Models
[➡️ link](https://developer.nvidia.com/blog/nvidia-gh200-superchip-accelerates-inference-by-2x-in-multiturn-interactions-with-llama-models/)

* [2024/10/22] New 📝 Step-by-step instructions on how to
✅ Optimize LLMs with NVIDIA TensorRT-LLM,
✅ Deploy the optimized models with Triton Inference Server,
✅ Autoscale LLMs deployment in a Kubernetes environment.
🙌 Technical Deep Dive:
[➡️ link](https://nvda.ws/3YgI8UT)

* [2024/10/07] 🚀🚀🚀Optimizing Microsoft Bing Visual Search with NVIDIA Accelerated Libraries
[➡️ link](https://developer.nvidia.com/blog/optimizing-microsoft-bing-visual-search-with-nvidia-accelerated-libraries/)

* [2024/09/29] 🌟 AI at Meta PyTorch + TensorRT v2.4 🌟 ⚡TensorRT 10.1 ⚡PyTorch 2.4 ⚡CUDA 12.4 ⚡Python 3.12
[➡️ link](https://github.com/pytorch/TensorRT/releases/tag/v2.4.0)

* [2024/09/17] ✨ NVIDIA TensorRT-LLM Meetup
[➡️ link](https://drive.google.com/file/d/1RR8GqC-QbuaKuHj82rZcXb3MS20SWo6F/view?usp=share_link)

* [2024/09/17] ✨ Accelerating LLM Inference at Databricks with TensorRT-LLM
[➡️ link](https://drive.google.com/file/d/1NeSmrLaWRJAY1rxD9lJmzpB9rzr38j8j/view?usp=sharing)

* [2024/09/17] ✨ TensorRT-LLM @ Baseten
[➡️ link](https://drive.google.com/file/d/1Y7L2jqW-aRmt31mCdqhwvGMmCSOzBUjG/view?usp=share_link)

* [2024/09/04] 🏎️🏎️🏎️ Best Practices for Tuning TensorRT-LLM for Optimal Serving with BentoML
[➡️ link](https://www.bentoml.com/blog/tuning-tensor-rt-llm-for-optimal-serving-with-bentoml)


* [2024/08/20] 🏎️SDXL with #TensorRT Model Optimizer ⏱️⚡ 🏁 cache diffusion 🏁 quantization aware training 🏁 QLoRA 🏁 #Python 3.12
[➡️ link](https://developer.nvidia.com/blog/nvidia-tensorrt-model-optimizer-v0-15-boosts-inference-performance-and-expands-model-support/)

* [2024/08/13] 🐍 DIY Code Completion with #Mamba ⚡ #TensorRT #LLM for speed 🤖 NIM for ease ☁️ deploy anywhere
[➡️ link](https://developer.nvidia.com/blog/revolutionizing-code-completion-with-codestral-mamba-the-next-gen-coding-llm/)

* [2024/08/06] 🗫 Multilingual Challenge Accepted 🗫
🤖 #TensorRT #LLM boosts low-resource languages like Hebrew, Indonesian and Vietnamese ⚡[➡️ link](https://developer.nvidia.com/blog/accelerating-hebrew-llm-performance-with-nvidia-tensorrt-llm/?linkId=100000278659647)

* [2024/07/30] Introducing🍊 @SliceXAI ELM Turbo 🤖 train ELM once ⚡ #TensorRT #LLM optimize ☁️ deploy anywhere
[➡️ link](https://developer.nvidia.com/blog/supercharging-llama-3-1-across-nvidia-platforms)

* [2024/07/23] 👀 @AIatMeta Llama 3.1 405B trained on 16K NVIDIA H100s - inference is #TensorRT #LLM optimized ⚡
🦙 400 tok/s - per node
🦙 37 tok/s - per user
🦙 1 node inference
[➡️ link](https://developer.nvidia.com/blog/supercharging-llama-3-1-across-nvidia-platforms)

* [2024/07/09] Checklist to maximize multi-language performance of @meta #Llama3 with #TensorRT #LLM inference:
✅ MultiLingual
✅ NIM
✅ LoRA tuned adaptors[➡️ Tech blog](https://developer.nvidia.com/blog/deploy-multilingual-llms-with-nvidia-nim/)

* [2024/07/02] Let the @MistralAI MoE tokens fly 📈 🚀 #Mixtral 8x7B with NVIDIA #TensorRT #LLM on #H100.
[➡️ Tech blog](https://developer.nvidia.com/blog/achieving-high-mixtral-8x7b-performance-with-nvidia-h100-tensor-core-gpus-and-tensorrt-llm?ncid=so-twit-928467)

* [2024/06/24] Enhanced with NVIDIA #TensorRT #LLM, @upstage.ai’s solar-10.7B-instruct is ready to power your developer projects through our API catalog 🏎️. ✨[➡️ link](https://build.nvidia.com/upstage/solar-10_7b-instruct?snippet_tab=Try )

* [2024/06/18] CYMI: 🤩 Stable Diffusion 3 dropped last week 🎊 🏎️ Speed up your SD3 with #TensorRT INT8 Quantization[➡️ link](https://build.nvidia.com/upstage/solar-10_7b-instruct?snippet_tab=Try )

* [2024/06/18] 🧰Deploying ComfyUI with TensorRT?  Here’s your setup guide [➡️ link](https://github.com/comfyanonymous/ComfyUI_TensorRT)

* [2024/06/11] ✨#TensorRT Weight-Stripped Engines ✨
Technical Deep Dive for serious coders ✅+99% compression ✅1 set of weights → ** GPUs ✅0 performance loss ✅** models…LLM, CNN, etc.[➡️ link](https://developer.nvidia.com/blog/maximum-performance-and-minimum-footprint-for-ai-apps-with-nvidia-tensorrt-weight-stripped-engines/)

* [2024/06/04] ✨ #TensorRT and GeForce #RTX unlock ComfyUI SD superhero powers 🦸⚡ 🎥 Demo: [➡️ link](https://youtu.be/64QEVfbPHyg)
📗 DIY notebook: [➡️ link](https://console.brev.dev/launchable/deploy?userID=2x2sil999&orgID=ktj33l4xj&name=ComfyUI_TensorRT&instance=L4%40g2-standard-4%3Anvidia-l4%3A1&diskStorage=500&cloudID=GCP&baseImage=docker.io%2Fpytorch%2Fpytorch%3A2.2.0-cuda12.1-cudnn8-runtime&ports=ComfUI%3A8188&file=https%3A%2F%2Fgithub.com%2Fbrevdev%2Fnotebooks%2Fblob%2Fmain%2Ftensorrt-comfyui.ipynb&launchableID=env-2hQX3n7ae5mq3NjNZ32DfAG0tJf)

* [2024/05/28] ✨#TensorRT weight stripping for ResNet-50 ✨ ✅+99% compression
✅1 set of weights → ** GPUs\ ✅0 performance loss ✅** models…LLM, CNN, etc
👀 📚 DIY [➡️ link](https://console.brev.dev/launchable/deploy?userID=2x2sil999&orgID=ktj33l4xj&launchableID=env-2h6bym7h5GFNho3vpWQQeUYMwTM&instance=L4%40g6.xlarge&diskStorage=500&cloudID=devplane-brev-1&baseImage=nvcr.io%2Fnvidia%2Ftensorrt%3A24.05-py3&file=https%3A%2F%2Fgithub.com%2FNVIDIA%2FTensorRT%2Fblob%2Frelease%2F10.0%2Fsamples%2Fpython%2Fsample_weight_stripping%2Fnotebooks%2Fweight_stripping.ipynb&name=tensorrt_weight_stripping_resnet50)

* [2024/05/21] ✨@modal_labs has the codes for serverless @AIatMeta Llama 3 on #TensorRT #LLM ✨👀 📚 Marvelous Modal Manual:
Serverless TensorRT-LLM (LLaMA 3 8B) | Modal Docs [➡️ link](https://modal.com/docs/examples/trtllm_llama)

* [2024/05/08] NVIDIA TensorRT Model Optimizer -- the newest member of the #TensorRT ecosystem is a library of post-training and training-in-the-loop model optimization techniques ✅quantization ✅sparsity ✅QAT [➡️ blog](https://developer.nvidia.com/blog/accelerate-generative-ai-inference-performance-with-nvidia-tensorrt-model-optimizer-now-publicly-available/)

* [2024/05/07] 🦙🦙🦙 24,000 tokens per second 🛫Meta Llama 3 takes off with #TensorRT #LLM 📚[➡️ link](https://blogs.nvidia.com/blog/meta-llama3-inference-acceleration/)

* [2024/02/06] [🚀 Speed up inference with SOTA quantization techniques in TRT-LLM](./docs/source/blogs/quantization-in-TRT-LLM.md)
* [2024/01/30] [ New XQA-kernel provides 2.4x more Llama-70B throughput within the same latency budget](./docs/source/blogs/XQA-kernel.md)
* [2023/12/04] [Falcon-180B on a single H200 GPU with INT4 AWQ, and 6.7x faster Llama-70B over A100](./docs/source/blogs/Falcon180B-H200.md)
* [2023/11/27] [SageMaker LMI now supports TensorRT-LLM - improves throughput by 60%, compared to previous version](https://aws.amazon.com/blogs/machine-learning/boost-inference-performance-for-llms-with-new-amazon-sagemaker-containers/)
* [2023/11/13] [H200 achieves nearly 12,000 tok/sec on Llama2-13B](./docs/source/blogs/H200launch.md)
* [2023/10/22] [🚀 RAG on Windows using TensorRT-LLM and LlamaIndex 🦙](https://github.com/NVIDIA/trt-llm-rag-windows#readme)
* [2023/10/19] Getting Started Guide - [Optimizing Inference on Large Language Models with NVIDIA TensorRT-LLM, Now Publicly Available
](https://developer.nvidia.com/blog/optimizing-inference-on-llms-with-tensorrt-llm-now-publicly-available/)
* [2023/10/17] [Large Language Models up to 4x Faster on RTX With TensorRT-LLM for Windows
](https://blogs.nvidia.com/blog/2023/10/17/tensorrt-llm-windows-stable-diffusion-rtx/)

</details>

## TensorRT-LLM Overview

TensorRT-LLM is an open-sourced library for optimizing Large Language Model (LLM) inference. It provides state-of-the-art optimizations, including custom attention kernels, inflight batching, paged KV caching, quantization (FP8, [FP4](https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/), INT4 [AWQ](https://arxiv.org/abs/2306.00978), INT8 [SmoothQuant](https://arxiv.org/abs/2211.10438), ...), speculative decoding, and much more, to perform inference efficiently on NVIDIA GPUs.

Recently [re-architected with a **PyTorch backend**](https://nvidia.github.io/TensorRT-LLM/torch.html), TensorRT-LLM now combines peak performance with a more flexible and developer-friendly workflow. The original [TensorRT](https://developer.nvidia.com/tensorrt)-based backend remains supported and continues to provide an ahead-of-time compilation path for building highly optimized "[Engines](https://docs.nvidia.com/deeplearning/tensorrt/quick-start-guide/index.html#ecosystem)" for deployment. The PyTorch backend complements this by enabling faster development iteration and rapid experimentation.

TensorRT-LLM provides a flexible [**LLM API**](https://nvidia.github.io/TensorRT-LLM/quick-start-guide.html#llm-api) to simplify model setup and inference across both PyTorch and TensorRT backends. It supports a wide range of inference use cases from a single GPU to multiple nodes with multiple GPUs using [Tensor Parallelism](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/features/parallelisms.html#tensor-parallelism) and/or [Pipeline Parallelism](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/features/parallelisms.html#pipeline-parallelism). It also includes a [backend](https://github.com/triton-inference-server/tensorrtllm_backend) for integration with the [NVIDIA Triton Inference Server](https://developer.nvidia.com/nvidia-triton-inference-server).

Several popular models are pre-defined and can be easily customized or extended using [native PyTorch code](./tensorrt_llm/_torch/models/modeling_deepseekv3.py) (for the PyTorch backend) or a [PyTorch-style Python API](./tensorrt_llm/models/llama/model.py) (for the TensorRT backend).


## Getting Started

To get started with TensorRT-LLM, visit our documentation:

- [Quick Start Guide](https://nvidia.github.io/TensorRT-LLM/quick-start-guide.html)
    - [Running DeepSeek](./examples/deepseek_v3)
- [Installation Guide for Linux](https://nvidia.github.io/TensorRT-LLM/installation/linux.html)
- [Installation Guide for Grace Hopper](https://nvidia.github.io/TensorRT-LLM/installation/grace-hopper.html)
- [Supported Hardware, Models, and other Software](https://nvidia.github.io/TensorRT-LLM/reference/support-matrix.html)
- [Benchmarking Performance](https://nvidia.github.io/TensorRT-LLM/performance/performance-tuning-guide/benchmarking-default-performance.html#benchmarking-with-trtllm-bench)
- [Release Notes](https://nvidia.github.io/TensorRT-LLM/release-notes.html)

## Useful Links
- [Quantized models on Hugging Face](https://huggingface.co/collections/nvidia/model-optimizer-66aa84f7966b3150262481a4): A growing collection of quantized (e.g., FP8, FP4) and optimized LLMs, including [DeepSeek FP4](https://huggingface.co/nvidia/DeepSeek-R1-FP4), ready for fast inference with TensorRT-LLM.
- [NVIDIA Dynamo](https://github.com/ai-dynamo/dynamo): A datacenter scale distributed inference serving framework that works seamlessly with TensorRT-LLM.
- [AutoDeploy](./examples/auto_deploy/README.md): An experimental backend for TensorRT-LLM to simplify and accelerate the deployment of PyTorch models.
