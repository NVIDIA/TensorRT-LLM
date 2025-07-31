# Model Evaluation with LM Evaluation Harness

lm-evaluation-harness is supported. To run the evaluation, please use the following command:

```bash
# model is defined the same as above. Other config args can also be specified in the model_args (comma separated).
# You can specify any tasks supported with lm-evaluation-harness.
cd examples/auto_deploy
python lm_eval_ad.py \
--model autodeploy --model_args model=meta-llama/Meta-Llama-3.1-8B-Instruct,world_size=2 --tasks mmlu
```
