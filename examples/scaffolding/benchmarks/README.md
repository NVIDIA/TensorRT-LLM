```
wget https://www.gutenberg.org/ebooks/1184.txt.utf-8
mv 1184.txt.utf-8 pg1184.txt
```

```
python -m examples.scaffolding.benchmarks.benchmark_agent_chat \
    --model_dir /home/scratch.trt_llm_data/llm-models/gpt_oss/gpt-oss-20b \
    --enable_multiround_chatbot \
    --multiround_data_source json_config \
    --multiround_json_config_file /home/scratch.rysun_gpu/TensorRT-LLM/examples/scaffolding/benchmarks/generate_multi_turn.json
```
