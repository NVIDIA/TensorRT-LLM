# Multimodal Feature Support Matrix (PyTorch Backend)

| Model              | CUDA Graph | Encoder IFB         | KV Cache Reuse | Chunked Prefill |
| :----------------- | :--------- | :------------------ | :------------- | :-------------- |
| Gemma 3            | Yes        | Yes                 | No             | No              |
| HyperCLOVA         | Yes        | Yes                 | No             | No              |
| VILA               | Yes        | No                  | No             | No              |
| LLaVA-NeXT         | Yes        | Yes                 | No             | No              |
| Llama 4            | Yes        | No                  | No             | No              |
| Mistral-Small-3.1  | Yes        | Yes                 | No             | No              |
| Phi-4-multimodal   | Yes        | Yes                 | No             | No              |
| Qwen2-VL           | Yes        | Yes                 | Yes            | No              |
| Qwen2.5-VL         | Yes        | Yes                 | Yes            | No              |
