# Multimodal Feature Support Matrix (PyTorch Backend)

| Model              | CUDA Graph | IFB w/ SharedTensor | KV Cache Reuse | Chunked Prefill |
| :----------------- | :--------- | :------------------ | :------------- | :-------------- |
| Gemma 3            | Yes        | Yes                 | No             | No              |
| HyperCLOVA         | Yes        | Yes                 | No             | No              |
| VILA               | Yes        | No                  | No             | No              |
| LLaVA-NeXT         | Yes        | No                  | No             | No              |
| Llama 4            | Yes        | No                  | No             | No              |
| Mistral            | Yes        | Yes                 | No             | No              |
| Phi-4-multimodal   | Yes        | Yes                 | No             | No              |
| Qwen2-VL           | Yes        | Yes                 | Yes            | No              |
| Qwen2.5-VL         | Yes        | Yes                 | Yes            | No              |
