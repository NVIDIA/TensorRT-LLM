# Expert Parallelism Load Balancer

## Offline (Static) EP Load Balancer

Run

```bash
export MODEL_PATH=<YOUR_MODEL_PATH>

# Enable doing statistic on routed expert ids
export EXPERT_STATISTIC_ITER_RANGE=50-100
export EXPERT_STATISTIC_PATH=expert_statistic

mkdir $EXPERT_STATISTIC_PATH

cat > ./extra_llm_api_options.yaml <<EOF
enable_attention_dp: true
pytorch_backend_config:
    use_cuda_graph: true
EOF

trtllm-eval --model $MODEL_PATH \
    --tp_size 2 \
    --ep_size 2 \
    --extra_llm_api_options ./extra_llm_api_options.yaml \
    --backend pytorch gsm8k
```


Read the statistics and convert to a

```bash
python generate_eplb_config.py \
    --num_experts_per_token 6 \
    --ep_size 2 \
    --num_slots 80 \
    --expert_statistic_path $EXPERT_STATISTIC_PATH \
    --output_path ./moe_load_balancer.yaml
```



```bash
# Disable doing statistic on routed expert ids
unset EXPERT_STATISTIC_ITER_RANGE

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
