# Expert Parallelism Load Balancer

## Offline (Static) EP Load Balancer

Run the model on a target dataset, e.g., GSM8K. At the same time, enable counting the routed expert ids during the model inference. The statistic data will be dumped when the counting finishes.

```bash
export MODEL_PATH=<YOUR_MODEL_PATH>
# Set the expert statistic data path
export EXPERT_STATISTIC_PATH=./expert_statistic
# Enable counting routed expert ids from iteration 50 to iteration 100
export EXPERT_STATISTIC_ITER_RANGE=50-100

mkdir -p $EXPERT_STATISTIC_PATH

# Prepare a config file and run inference on GSM8K
cat > ./extra_llm_api_options.yaml <<EOF
enable_attention_dp: true
pytorch_backend_config:
    use_cuda_graph: true
EOF

trtllm-eval --model $MODEL_PATH \
    --tp_size 8 \
    --ep_size 8 \
    --extra_llm_api_options ./extra_llm_api_options.yaml \
    --backend pytorch gsm8k
```

You may check the dumped statstic files in `$EXPERT_STATISTIC_PATH`. Then, use [`generate_eplb_config.py`](./generate_eplb_config.py) to read the dumped statistics and convert to an EPLB config file. Please also specify the expert parallelism size (`--ep_size`) and the number of total slots (`--num_slots`) that will be used in EPLB deployment.

```bash
python generate_eplb_config.py \
    --ep_size 8 \
    --num_slots 320 \
    --expert_statistic_path $EXPERT_STATISTIC_PATH \
    --output_path ./moe_load_balancer.yaml
```

Disable counting the routed expert ids, and run inference with EPLB configuration.

```bash
# Disable counting routed expert ids
unset EXPERT_STATISTIC_ITER_RANGE

# Prepare a config file with EPLB config and run inference on GSM8K
cat > ./extra_llm_api_options_eplb.yaml <<EOF
enable_attention_dp: true
pytorch_backend_config:
    use_cuda_graph: true
    moe_load_balancer: ./moe_load_balancer.yaml
EOF

trtllm-eval --model $MODEL_PATH \
    --tp_size 2 \
    --ep_size 2 \
    --extra_llm_api_options ./extra_llm_api_options_eplb.yaml \
    --backend pytorch gsm8k
```
