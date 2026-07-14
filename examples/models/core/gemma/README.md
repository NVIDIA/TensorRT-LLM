# Gemma (PyTorch Backend)

Gemma 4 runs on the TensorRT LLM **PyTorch backend** — HuggingFace checkpoints are
loaded directly. The legacy TensorRT engine flow (`convert_checkpoint.py` /
`trtllm-build`) is no longer required.

## Run Gemma 4

Gemma 4 runs on the **PyTorch backend** — HuggingFace checkpoints are loaded directly. The legacy TensorRT engine flow (`convert_checkpoint.py` / `trtllm-build`) is not required and is not covered here.

| HuggingFace checkpoint        | Modalities                       | Notes                                  |
|-------------------------------|----------------------------------|----------------------------------------|
| `google/gemma-4-E2B-it`       | text + image + video + audio     | Single-GPU friendly                    |
| `google/gemma-4-E4B-it`       | text + image + video + audio     | Single-GPU friendly                    |
| `google/gemma-4-26B-A4B-it`   | text + image + video (MoE)       | Multi-GPU recommended; no audio tower  |
| `google/gemma-4-31B-it`       | text + image + video             | Multi-GPU recommended; no audio tower  |

All four variants ship the vision tower (image + video). The audio tower is only present on `E2B` / `E4B`. The examples below use `google/gemma-4-E4B-it` (small, full multimodal) — swap the model name for the other variants and bump `--tp_size` (e.g. `4` or `8`) for the larger checkpoints.

### Serve with `trtllm-serve` (OpenAI-compatible API)

Launch the server:

```bash
trtllm-serve \
    google/gemma-4-E4B-it \
    --host 0.0.0.0 \
    --port 8000 \
    --tp_size 1 \
    --max_batch_size 16
```

Query it with `curl`:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "google/gemma-4-E4B-it",
    "messages": [
      {"role": "user", "content": "Explain quantum tunneling in one paragraph."}
    ],
    "max_tokens": 256,
    "temperature": 0
  }'
```

The `/v1/chat/completions` endpoint applies the Gemma 4 chat template automatically.

### Accuracy evaluation with `trtllm-eval`

`trtllm-eval` is the canonical entry point for accuracy benchmarks. Two tasks relevant to Gemma 4:

```bash
# MMMU (vision multiple-choice; works on E2B / E4B / 26B-A4B / 31B)
trtllm-eval \
    --model google/gemma-4-E4B-it \
    --tp_size 1 \
    --max_batch_size 64 \
    mmmu \
    --num_samples 900

# CoVoST 2 BLEU (English → Chinese speech translation; E2B / E4B only)
trtllm-eval \
    --model google/gemma-4-E4B-it \
    --tp_size 1 \
    --max_batch_size 64 \
    covost2 \
    --lang_pair en_zh-CN \
    --num_samples 500
```

Internally, `trtllm-eval mmmu` forces `apply_chat_template=True` because it is a multimodal benchmark, while `trtllm-eval covost2` sets `--apply_chat_template` to `True` by default. Both configurations align with the chat template that Gemma 4 was trained on.
