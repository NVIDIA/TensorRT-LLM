# Expert Parallelism Load Balancer (EPLB)

Effective load balancing is crucial when leveraging large-scale expert parallelism. As described in the [DeepSeek-V3 paper](https://arxiv.org/abs/2412.19437), redundant experts can be introduced to rebalance the workload across GPUs. This mechanism is known as the Expert Parallelism Load Balancer ([EPLB](https://github.com/deepseek-ai/EPLB)).

> **Note:** Currently, only the offline EP load balancer is supported.

## Offline EP Load Balancer

### Step 1: Run Inference and Collect Statistics

To generate the necessary statistics for load balancing, run your model on a target dataset (e.g., GSM8K) while counting the routed expert IDs during inference. Once counting is complete, the statistics will be saved for further processing.

Set up some environment variables:

```bash
export MODEL_PATH=<YOUR_MODEL_PATH>
# Set the expert statistic data path
export EXPERT_STATISTIC_PATH=./expert_statistic
# Enable counting of routed expert IDs from iteration 100 to iteration 200
export EXPERT_STATISTIC_ITER_RANGE=100-200
```

Prepare a configuration file and run inference on GSM8K:

```bash
cat > ./extra_llm_api_options.yaml <<EOF
enable_attention_dp: true
use_cuda_graph: true
EOF

trtllm-eval --model $MODEL_PATH \
    --tp_size 8 \
    --ep_size 8 \
    --extra_llm_api_options ./extra_llm_api_options.yaml \
    --backend pytorch gsm8k
```

After inference, review the dumped statistic files in `$EXPERT_STATISTIC_PATH`.

### Step 2: Generate the EPLB Configuration

Use the provided [`generate_eplb_config.py`](./generate_eplb_config.py) script to convert the collected statistics into an EPLB configuration file. Specify the expert parallelism size (`--ep_size`) and the total number of slots (`--num_slots`) that will be used for deployment:

```bash
python generate_eplb_config.py \
    --ep_size 8 \
    --num_slots 320 \
    --expert_statistic_path $EXPERT_STATISTIC_PATH \
    --output_path ./moe_load_balancer.yaml
```

### Step 3: Run Inference with the EPLB Configuration

Disable the expert ID counting by unsetting the environment variable:

```bash
unset EXPERT_STATISTIC_ITER_RANGE
```

Prepare a new configuration file that incorporates the generated EPLB configuration, then run inference on GSM8K:

```bash
cat > ./extra_llm_api_options_eplb.yaml <<EOF
enable_attention_dp: true
use_cuda_graph: true
moe_load_balancer: ./moe_load_balancer.yaml
EOF

trtllm-eval --model $MODEL_PATH \
    --tp_size 8 \
    --ep_size 8 \
    --extra_llm_api_options ./extra_llm_api_options_eplb.yaml \
    --backend pytorch gsm8k
```
