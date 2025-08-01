(expert-parallelism)=

# Expert Parallelism in TensorRT-LLM

## Mixture of Experts (MoE)

Mixture of Experts (MoE) architectures have become widespread, with models such as [Mistral Mixtral 8×7B](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1). Specifically, MoE’s structure supports multiple parallel feed-forward neural-network (FFN) layers (called experts) in place of the single FFN layer in a dense model. When tokens arrive, the router layer selects the top-k experts for each token, and the corresponding hidden state of each token is dispatched to those experts. As a result, there are multiple tokens’ hidden states that are dispatched to each expert.

<img src="https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/blogs/media/moe_structure.png?raw=true" alt="moe_structure" width="500" height="auto">

<sub>the MOE structure in Switch Transformer: [https://arxiv.org/pdf/2101.03961.pdf](https://arxiv.org/pdf/2101.03961.pdf) </sub>

## Tensor Parallel vs Expert Parallel

Parallelism on multi-GPUs is necessary if the MoE model can not be accommodated by a single GPU’s memory.  We have supported two kinds of parallel patterns for MoE structure, Tensor Parallel (default pattern), Expert Parallel, and a hybrid of the two.

<img src="https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/blogs/media/tp_ep.png?raw=true" alt="tensor parallel vs expert parallel" width="500" height="auto">

Tensor Parallel evenly splits each expert’s weight and distributes them to different GPUs, which means each GPU holds partial weight of all experts, While Expert Parallel evenly distributes some of the experts’ full weight to different GPUs, which means each GPU holds part of the experts’ full weight. As a result, each GPU rank in the Tensor Parallel group receives all tokens’ hidden states for all experts, then computes using the partial weights, while for Expert Parallel, each GPU rank only receives part of tokens’ hidden states for experts on this rank, then computes using the full weights.

When both Tensor Parallel and Expert Parallel are enabled, each GPU handles a portion of the expert weights matrices (as in EP mode) and these weights are further sliced across multiple GPUs (as in TP mode). This hybrid approach aims to balance the workload more evenly across GPUs, enhancing efficiency and reducing the likelihood of bottlenecks associated with EP mode alone.


## How to Enable

The default parallel pattern is Tensor Parallel. You can enable Expert Parallel or hybrid parallel by setting `--moe_tp_size` and `--moe_ep_size` when calling `convert_checkpoint.py`. If only `--moe_tp_size` is provided, TRT-LLM will use Tensor Parallel for the MoE model; if only `--moe_ep_size` is provided, TRT-LLM will use Expert Parallel; if both are provided, the hybrid parallel will be used.

Ensure the product of `moe_tp_size` and `moe_ep_size` is equal to `tp_size`, since the total number of MoE parallelism across all GPUs must match the total number of parallelism in other parts of the model.

The other parameters related to the MoE structure, such as `num_experts_per_tok` (TopK in previous context) and `num_local_experts,` can be found in the model’s configuration file, such as the one for [Mixtral 8x7B model](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1/blob/main/config.json).
