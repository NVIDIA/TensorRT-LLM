# DeepSeek R1 MTP Implementation and Optimization
by NVIDIA TensorRT LLM team
## Table of Contents
- [DeepSeek R1 MTP Implementation and Optimization](#deepseek-r1-mtp-implementation-and-optimization)
  - [Table of Contents](#table-of-contents)
  - [MTP for inference](#mtp-for-inference)
    - [Background](#background)
    - [MTP Vanilla](#mtp-vanilla)
    - [MTP Eagle](#mtp-eagle)
  - [MTP implementation in TensorRT LLM](#mtp-implementation-in-tensorrt-llm)
    - [Basic Implementation](#basic-implementation)
    - [MTP Modules](#mtp-modules)
    - [Attention for MTP](#attention-for-mtp)
    - [How to run DeepSeek models with MTP](#how-to-run-deepseek-models-with-mtp)
  - [MTP optimization - Relaxed Acceptance](#mtp-optimization---relaxed-acceptance)
    - [Relaxed Acceptance](#relaxed-acceptance)
    - [How to run the DeepSeek-R1 model with Relaxed Acceptance](#how-to-run-the-deepseek-r1-model-with-relaxed-acceptance)
  - [Evaluation](#evaluation)
    - [Achieving speedup with MTP speculative decoding](#achieving-speedup-with-mtp-speculative-decoding)
    - [Accuracy studies for Relaxed Acceptance](#accuracy-studies-for-relaxed-acceptance)
  - [Future Works](#future-works)
    - [Tree-based speculative decoding support](#tree-based-speculative-decoding-support)
    - [Eagle3 support](#eagle3-support)
    - [Fix known issues](#fix-known-issues)
  - [Acknowledgment](#acknowledgment)


TensorRT LLM achieves world-record inference performance for DeepSeek-R1 on NVIDIA Blackwell GPUs, where Multi-Token Prediction (MTP) delivers a significant speedup. In our [previous blog post](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/blogs/tech_blog/blog1_Pushing_Latency_Boundaries_Optimizing_DeepSeek-R1_Performance_on_NVIDIA_B200_GPUs.md), we discussed the key optimizations that enable the outstanding inference latency of the DeepSeek-R1 model. This article dives deeper into the implementation and optimization of MTP in TensorRT LLM.

## MTP for inference
Inspired by a previous [research work](https://arxiv.org/pdf/2404.19737), MTP is designed to help the DeepSeek-V3 training. It adds additional MTP modules at the end of the main model and uses them to predict additional tokens. In this way, MTP can extend the prediction scope to multiple future tokens at each position to achieve better model accuracy. During inference, those MTP modules can also be used for speculative decoding to improve the generation latency further. In this section, we will introduce the MTP speculative decoding algorithm for LLM inference.

### Background
Speculative decoding is a popular technique for faster and cost-effective LLM inference. It’s based on the premise that generating multiple future tokens(especially for decode phase which is less compute bound) is more efficient than processing a single token. Speculative decoding techniques usually divide the process into a low-cost draft stage and a parallelized verification stage. The draft stage predicts draft tokens by using a small model or a subset of layers in the main model. And the verification stage uses the main model to determine how many of these draft tokens to accept, which is far more efficient than generating one token per iteration.

<div align="center">
<figure>
  <img src="https://github.com/NVIDIA/TensorRT-LLM/raw/main/docs/source/blogs/media/tech_blog2_verify_and_accept.png" alt="tech_blog2_verify_and_accept" width="1280" height="auto">
</figure>
</div>
<p align="center"><sub><em>Figure 1. Verification example</em></sub></p>

Figure 1 shows an example of how to verify and accept those draft tokens. Assuming there are a total of 5 draft tokens “ABCDE”, we will extend them to the input token “G”, and input a total of 6 tokens to the main model. After sampling, we can get six different expected tokens, then compare the expected tokens with the draft tokens and accept the longest prefix matched tokens. In this example, the tokens “ABC” are matched. Because “H” is predicted by the main model and the corresponding input token “C” is already accepted, “H” will also be accepted. In this way, we can accept four tokens in a single iteration. MTP also uses this method to verify and accept draft tokens.
For the draft stage in MTP, there are two different MTP methods, MTP vanilla and MTP eagle. They can be used for different inference cases.

### MTP Vanilla

<div align="center">
<figure>
  <img src="https://github.com/NVIDIA/TensorRT-LLM/raw/main/docs/source/blogs/media/tech_blog2_mtp_vanilla.png" alt="tech_blog2_mtp_vanilla" width="640" height="auto">
</figure>
</div>
<p align="left"><sub><em>Figure 2. MTP Vanilla, where t<sub>i</sub> is the input token, d<sub>i</sub> is the predicted draft token, K is the number of MTP modules, and h<sub>i</sub><sup>n</sup> is the hidden state of the n-th MTP module. Note that h<sub>0</sub> means the hidden states of the main model.  (Disclaimer: the figures adapted from the original DeepSeek V3 tech report)</em></sub></p>


MTP Vanilla method is more similar to the MTP training, and it sequentially uses different MTP modules to predict multiple draft tokens. This method can support model checkpoints with weights of multiple different MTP modules. And each MTP module will have its own KV cache.

Figure 2 illustrates the MTP vanilla inference. In the context phase, assuming there are a total of four input tokens, we will get the output token $t_5$ and the hidden states after the main model forward. The output token will be appended to the input tokens, then we shift out the first token to get tokens from $t_2$ to $t_5$ as the input tokens of the first MTP module. The hidden states from the main model will be directly used as the input of the first MTP module to predict the first draft token. For the next few MTP modules, we'll append the newly generated draft token and the hidden states corresponding to the last input token to the input tokens and hidden states. Then we'll shift out the first token to prepare the inputs for the next MTP module. In this way, we can retain as much information as possible from the main model, which helps the draft layer make more accurate predictions.

In the generation phase, there will be a little difference. The predicted token $t_5$ and the draft tokens will be used as inputs for the main model. After the main model forward, we will do the verification to get the accepted tokens. In this example, assuming $j$ draft tokens $d_6$~$d_{j+5}$ are accepted. Then prepare the MTP module inputs.  Different from the context phase, we will prepare input IDs and hidden states of a total of $K$ tokens before the last accepted token. In this example, the last accepted token is $t_{j+6}$. Then we can get the first draft token after the first MTP module forward. For the sequential MTP modules, we can prepare their inputs in a similar way to the MTP modules in the context phase, so all of those MTP modules have the same input sequence length. After predicting all of the draft tokens, we need to evict the keys/values of those rejected draft tokens from the main model's KV cache to ensure the subsequent calculation is correct.

### MTP Eagle

<div align="center">
<figure>
  <img src="https://github.com/NVIDIA/TensorRT-LLM/raw/main/docs/source/blogs/media/tech_blog2_mtp_eagle.png" alt="tech_blog2_mtp_eagle" width="640" height="auto">
</figure>
</div>
<p align="center"><sub><em>Figure 3. MTP Eagle, using the same notation as Figure 2</em></sub></p>

MTP Eagle can be viewed as a variant of [Eagle](https://arxiv.org/pdf/2401.15077) speculative decoding method, but only supports chain decoding now. It reuses the same MTP module and repeats multiple times to predict draft tokens. MTP Eagle supports the model checkpoint with only one MTP module. The official DeepSeek-V3 and DeepSeek-R1 have only one MTP module in their checkpoints. Another difference with MTP vanilla is the KV cache. In the MTP Eagle method, the MTP module reuses the same KV cache when predicting multiple draft tokens.

Figure 3 gives an MTP Eagle example. In the context phase, the inputs of the first MTP module forward are the same as the MTP Vanilla. However, for the sequential MTP module forward, the first difference is that MTP Eagle uses the same MTP module to predict draft tokens and reuses the same KV cache. Another difference is that we only need to input the token ID and the hidden state of one token. The token is the last predicted draft token, while the hidden state is the corresponding hidden state in the last MTP module forward. In this way, we can predict total K draft tokens by using only one MTP module.

In the generation phase, the verification stage is the same as MTP Vanilla. Once we get the accepted tokens, we use all of them along with their corresponding hidden states as inputs for the first MTP module forward. Unlike MTP Vanilla, which needs to store past tokens and hidden states, this approach is much easier to implement. Subsequent MTP module forwards follow the same input preparation method as the context phase. After predicting all draft tokens, we need to evict the key/value pairs of any rejected draft tokens from the main model’s KV cache.

## MTP implementation in TensorRT LLM
### Basic Implementation
TensorRT LLM has two different paths for MTP, one for [MTP Vanilla](https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/_torch/speculative/mtp.py#L1047) and another for [MTP Eagle](https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/_torch/speculative/mtp.py#L1047). MTP Eagle is the default path for DeepSeek-V3 and DeepSeek-R1 models.

<div align="center">
<figure>
  <img src="https://github.com/NVIDIA/TensorRT-LLM/raw/main/docs/source/blogs/media/tech_blog2_overall_workflow.png" alt="tech_blog2_overall_workflow" width="800" height="auto">
</figure>
</div>
<p align="center"><sub><em>Figure 4. MTP workflow in TensorRT LLM</em></sub></p>

Figure 4 shows the overall workflow of MTP in TensorRT LLM. Both paths share the runtime workflow, and the differences are in the MTP modules forward. In the context phase, there is no draft token in the inputs. TensorRT LLM model engine fetches the input IDs from the requests and inputs to the model engine forward to get the next token and the hidden state. Then we prepare the MTP module inputs, and the MTP modules forward the inputs to predict the draft tokens.

The generation workflow is more complicated. We need to do both the verification and draft stages. The predicted new token and draft tokens are the inputs for the main model. After the main model forward, we can sample from the output logits and get the following new tokens. Then compare them with the input draft tokens to get the final accepted tokens. The verification stage will be finished here. We will use the accepted tokens and hidden states to start a new draft stage, which uses the MTP layers to predict new draft tokens for the next iteration. Finally, we need to rewind the KV cache to evict keys/values corresponding to those rejected tokens.

Except for the Rewind KV Cache, all of those processes are inside the model engine forward function. In this way, we can use one model engine to support MTP inference, and it would be easier for MTP to be compatible with other features, such as CUDA graph and overlap scheduler. When enabling CUDA graph, both the verification and draft stages can be captured in one graph, significantly reducing CPU overhead.

### MTP Modules

<div align="center">
<figure>
  <img src="https://github.com/NVIDIA/TensorRT-LLM/raw/main/docs/source/blogs/media/tech_blog2_mtp_modules.png" alt="tech_blog2_mtp_modules" width="640" height="auto">
</figure>
</div>
<p align="center"><sub><em>Figure 5. MTP model architecture</em></sub></p>

Figure 5 introduces the basic model architecture of [MTP Vanilla](https://github.com/NVIDIA/TensorRT-LLM/blob/338744fba6a91147b739b7f02d19b37bc19aa17a/tensorrt_llm/_torch/speculative/mtp.py#L326), [MTP Eagle](https://github.com/NVIDIA/TensorRT-LLM/blob/338744fba6a91147b739b7f02d19b37bc19aa17a/tensorrt_llm/_torch/speculative/mtp.py#L1047), and the basic [MTP module](https://github.com/NVIDIA/TensorRT-LLM/blob/338744fba6a91147b739b7f02d19b37bc19aa17a/tensorrt_llm/_torch/models/modeling_deepseekv3.py#L829) design. Because MTP vanilla needs $K$ input tokens, if the number of accepted tokens is less than the number of input tokens, i.e. $j<K$, we need to use the old token IDs and hidden states as the input of the first MTP module. To avoid bringing much additional computation overhead, we add two tensors for each request to save the past $K$ input IDs and the hidden states of past $K$ tokens, and update them by using the accepted tokens and corresponding hidden states each iteration. In this way, we can read these tensors when preparing inputs for the first MTP module. MTP Eagle implementation is much easier and straightforward, just call the same MTP module forward $K$ times to get $K$ new draft tokens.

The MTP module follows the design in DeepSeek-V3. The embedding layer and output head in MTP modules are shared with the main model, which can save GPU memory consumption.


### Attention for MTP

Attention is also a very important component in supporting MTP inference. The changes are mainly in the attention kernels for the generation phase. For the normal request, there will be only one input token in the generation phase, but for MTP, there will be $K+1$ input tokens. Since MTP sequentially predicts additional tokens, the predicted draft tokens are chained. Though we have an MTP Eagle path, currently, we only have the chain-based support for MTP Eagle. So, a causal mask is enough for the attention kernel to support MTP. In our implementation, TensorRT LLM will use the fp8 flashMLA generation kernel on Hopper GPU, while using TRTLLM customized attention kernels on Blackwell for better performance.

### How to run DeepSeek models with MTP
Run DeepSeek-V3/R1 models with MTP, use [examples/llm-api/quickstart_advanced.py](https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/llm-api/quickstart_advanced.py) with additional options:

```bash
cd examples/llm-api
python quickstart_advanced.py --model_dir <YOUR_MODEL_DIR> --spec_decode_algo MTP --spec_decode_nextn N
```

To benchmark min-latency performance with MTP, you need to follow [this document](https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/models/core/deepseek_v3/README.md#6-dataset-preparation) to prepare your dataset, then follow the steps below:

```bash
YOUR_DATA_PATH=<your dataset file following the format>

cat >./extra-llm-api-config.yml<<EOF
cuda_graph_config: {}
moe_config:
  backend: TRTLLM
speculative_config:
    decoding_type: MTP
    num_nextn_predict_layers: 3
EOF

export TRTLLM_ENABLE_PDL=1

trtllm-bench --model nvidia/DeepSeek-R1-FP4 \
    throughput \
    --dataset $YOUR_DATA_PATH \
    --backend pytorch \
    --num_requests 10 \
    --concurrency 1 \
    --max_batch_size 1 \
    --tp 8 \
    --ep 2 \
    --extra_llm_api_options ./extra-llm-api-config.yml
```

## MTP optimization - Relaxed Acceptance
DeepSeek-R1 is a reasoning model that first outputs some thinking tokens, after which the user can get the actual outputs. The thinking process usually takes up a lot of tokens, and the quality of the outputs of the thinking process may have a limited impact on the final answer. So we want to use a more aggressive acceptance strategy, called [relaxed acceptance](https://github.com/NVIDIA/TensorRT-LLM/pull/3865), for the thinking process to speed up the thinking decoding phase. This will be a tradeoff between speedup and output quality. From the experimental results, the impact of relaxed acceptance on output quality is limited.

### Relaxed Acceptance

<div align="center">
<figure>
  <img src="https://github.com/NVIDIA/TensorRT-LLM/raw/main/docs/source/blogs/media/tech_blog2_relaxed_acceptance.png" alt="tech_blog2_relaxed_acceptance" width="1024" height="auto">
</figure>
</div>
<p align="center"><sub><em>Figure 6. Relaxed Acceptance example. Use MTP nextn=4 and top-3 in this example.</em></sub></p>

In previous verification and acceptance, we will use a top-1 to sample from the logits the main model to get the “expected” tokens as shown in Figure 1. There will be only one choice to compare with the draft tokens, which we call “Strict Acceptance”.

As for the Relaxed Acceptance, we first get the top-N tokens sampled from the logits, so more candidates will be compared with the input draft tokens. To make sure the accepted tokens are as accurate as possible, we also added a probability threshold, i.e., delta. We can get the token probabilities by applying a softmax to the logits. After getting the top-N tokens, we will remove tokens from the candidate list if their probability is smaller than the (top-1 probability - delta). In this way, we may get more than one token candidate, and all of those tokens are with a high probability. Then we can compare the input draft tokens with those candidates. If one of them matches, we can accept this draft token, so the acceptance rate will be increased. Figure 6 shows an example of a comparison between Strict Acceptance and Relaxed Acceptance.

Note that the Relaxed Acceptance will only be used during the thinking phase, while the Strict Acceptance will still be used during the non-thinking phase. And the Relaxed Acceptance only supports the DeepSeek-R1 model now.


### How to run the DeepSeek-R1 model with Relaxed Acceptance

Run DeepSeek-R1 models with MTP Relaxed Acceptance, use [examples/llm-api/quickstart_advanced.py](https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/llm-api/quickstart_advanced.py) with additional options:

```bash
cd examples/llm-api
python quickstart_advanced.py --model_dir <YOUR_MODEL_DIR> --spec_decode_algo MTP --spec_decode_nextn N --use_relaxed_acceptance_for_thinking --relaxed_topk 10 --relaxed_delta 0.6
```

To benchmark min-latency performance with MTP Relaxed Acceptance, you need to follow [this document](https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/models/core/deepseek_v3/README.md#6-dataset-preparation) to prepare your dataset, then follow the steps below:

```bash
YOUR_DATA_PATH=<your dataset file following the format>

cat >./extra-llm-api-config.yml<<EOF
cuda_graph_config: {}
moe_config:
  backend: TRTLLM
speculative_config:
    decoding_type: MTP
    num_nextn_predict_layers: 3
    use_relaxed_acceptance_for_thinking: true
    relaxed_topk: 10
    relaxed_delta: 0.6
EOF

export TRTLLM_ENABLE_PDL=1

trtllm-bench --model nvidia/DeepSeek-R1-FP4 \
    throughput \
    --dataset $YOUR_DATA_PATH \
    --backend pytorch \
    --num_requests 10 \
    --concurrency 1 \
    --max_batch_size 1 \
    --tp 8 \
    --ep 2 \
    --extra_llm_api_options ./extra-llm-api-config.yml
```

## Evaluation
### Achieving speedup with MTP speculative decoding

<div align="center">
<figure>
  <img src="https://github.com/NVIDIA/TensorRT-LLM/raw/main/docs/source/blogs/media/tech_blog2_perf_and_ar.png" alt="tech_blog2_perf_and_ar" width="1280" height="auto">
</figure>
</div>
<p align="center"><sub><em>Figure 7. DeepSeek-R1-FP4 671B min-latency performance with different MTP next-n</em></sub></p>

We tested the min-latency (batch size = 1) performance of the DeepSeek-R1-FP4 model with different MTP next-n on a B200 node. The MLA runs with TP=8, and the MoE runs with EP=2. And there are ten different requests with ISL/OSL=1K/2K. From Figure 7, we can see that MTP=3 can help get the best min-latency performance on 8 B200 GPUs, which can bring 2.16x speedup compared with the baseline nextn=0. And with the help of the relaxed acceptance, the min-latency performance can be further improved to achieve a 2.33x speedup. We also evaluated the CUDA graph and overlap scheduler benefits. For such a min-latency case, CUDA graph can achieve a 7.22x average speedup, while the overlap scheduler can achieve 1.03x average latency.

### Accuracy studies for Relaxed Acceptance

<div align="center">
<figure>
  <img src="https://github.com/NVIDIA/TensorRT-LLM/raw/main/docs/source/blogs/media/tech_blog2_acc_relaxed_acceptance.png" alt="tech_blog2_acc_relaxed_acceptance" width="800" height="auto">
</figure>
</div>
<p align="center"><sub><em>Figure 8. Ablation results for the Relaxed Acceptance. Using MTP nextn=3, top-10, and delta=0.6.</em></sub></p>

We validated the Relaxed Acceptance on different datasets. In Figure 8, we show the ablation results for Relaxed Acceptance by using the DeepSeek-R1-FP4 model. Compared with Strict Acceptance, the impact of Relaxed Acceptance on output quality is limited, resulting in only a slight accuracy drop.

## Future Works
### Tree-based speculative decoding support

<div align="center">
<figure>
  <img src="https://github.com/NVIDIA/TensorRT-LLM/raw/main/docs/source/blogs/media/tech_blog2_tree_spec_decoding.png" alt="tech_blog2_tree_spec_decoding" width="800" height="auto">
</figure>
</div>
<p align="center"><sub><em>Figure 9. Comparison between the chain-based and tree-based speculative decoding</em></sub></p>

TensorRT LLM PyTorch backend can only support chain-based speculative decoding now, both MTP Vanilla and MTP Eagle. However, the tree-based speculative decoding technique is widely used in previous advanced methods, such as Ealge2 and Eagle3, to increase the acceptance rate. MTPs in TensorRT LLM can also be extended to support the tree-based technique. Figure 9 compares the chain-based method with the tree-based method. Both full tree and dynamic tree methods can help expand the candidate combinations, so that we can have more choices for the draft tokens.

### Eagle3 support

Another important method is Eagle3. From the [Eagle3 paper](https://arxiv.org/pdf/2503.01840), the promising results show that it can help greatly increase the acceptance rate by leveraging different levels’ hidden states to predict draft tokens. Since TensorRT LLM already has [Eagle-3 support](https://github.com/NVIDIA/TensorRT-LLM/pull/3035) now, in the future, we also want to train an Eagle3 head to support DeepSeek-V3/R1+Eagle3 to achieve better speedup.

## Acknowledgment

This was a remarkable cross-team effort to support and optimize MTP in TensorRT LLM. We would like to extend our gratitude to everyone who contributed to making this possible, as it involved a typical system/algorithm co-design approach spanning multiple technical layers—including kernel optimization, runtime enhancements, algorithmic improvements, and performance measurement & analysis. And a special thanks goes to the DeepSeek team for developing the MTP method, which lays down the foundation of this blog.
