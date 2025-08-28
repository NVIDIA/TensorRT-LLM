# Multimodal Feature Support Matrix (PyTorch Backend)

| Model Architecture/Feature         | Overlap Scheduler | CUDA Graph | Chunked Prefill | Torch Sampler | TLLM C++ Sampler | KV Cache Reuse | Logits Post Processor | EPD Disaggregated Serving |
| ---------------------------------- | ----------------- | ---------- | --------------- | ------------- | ---------------- | -------------- | --------------------- | ------------------------- |
| Gemma3ForConditionalGeneration     | Yes               | Yes        | N/A             | Yes           | Yes              | N/A            | Yes                   | No                        |
| HCXVisionForCausalLM               | Yes               | Yes        | No              | Yes           | Yes              | No             | Yes                   | No                        |
| LlavaLlamaModel (VILA)             | Yes               | Yes        | No              | Yes           | Yes              | No             | Yes                   | No                        |
| LlavaNextForConditionalGeneration  | Yes               | Yes        | No              | Yes           | Yes              | Yes            | Yes                   | No                        |
| Llama4ForConditionalGeneration     | Yes               | Yes        | No              | Yes           | Yes              | No             | Yes                   | No                        |
| Mistral3ForConditionalGeneration   | Yes               | Yes        | No              | Yes           | Yes              | No             | Yes                   | No                        |
| Phi4MMForCausalLM                  | Yes               | Yes        | No              | Yes           | Yes              | No             | Yes                   | No                        |
| Qwen2VLForConditionalGeneration    | Yes               | Yes        | No              | Yes           | Yes              | Yes            | Yes                   | No                        |
| Qwen2_5_VLForConditionalGeneration | Yes               | Yes        | No              | Yes           | Yes              | Yes            | Yes                   | No                        |
