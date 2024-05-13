(expert-parallelism)=

# Expert Parallelism in TensorRT-LLM

## Mixture of Experts (MoE)

Mixture of Experts (MoE) architectures have been used widely recently, such as [Mistral Mixtral 8x7B](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1/blob/main/config.json). Specifically, MOE’s structure supports multiple parallel Feedforward Neural Network (FFN) layers (called experts) to replace the single FFN layer in the dense model. When tokens arrive, the router layer selects the TopK experts for each token. The corresponding hidden state of the token is then dispatched to the selected TopK experts, respectively. As a result, there are multiple tokens’ hidden states that are dispatched to each expert.

<img src="https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/blogs/media/moe_structure.png?raw=true" alt="moe_structure" width="500" height="auto">

<sub>the MOE structure in Switch Transformer: [https://arxiv.org/pdf/2101.03961.pdf](https://arxiv.org/pdf/2101.03961.pdf) </sub>

## Tensor Parallel vs Expert Parallel

Parallelism on multi-GPUs is necessary if the MoE model can not be accommodated by a single GPU’s memory.  We have supported two kinds of parallel patterns for MoE structure, Tensor Parallel (default pattern) and Expert Parallel.

<img src="https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/blogs/media/tp_ep.png?raw=true" alt="tensor parallel vs expert parallel" width="500" height="auto">

Tensor Parallel evenly splits each expert’s weight and distributes them to different GPUs, which means each GPU holds partial weight of all experts, While Expert Parallel evenly distributes some of the experts’ full weight to different GPUs, which means each GPU holds part of the experts’ full weight. As a result, each GPU rank in the Tensor Parallel group receives all tokens’ hidden states for all experts, then computes using the partial weights, while for Expert Parallel, each GPU rank only receives part of tokens’ hidden states for experts on this rank, then computes using the full weights.


## How to Enable

The default parallel pattern is Tensor Parallel. You can enable Expert Parallel by setting `--moe_tp_mode 1` when calling `convert_coneckpoint.py`, and `--tp_size` is used to set the Expert Parallel size.

The other parameters related to MoE structure, such as `num_experts_per_tok` (TopK in previous context), and `num_local_experts`, can be find in the model’s configuration file, such as the one for [Mixtral 8x7B model](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1/blob/main/config.json).
