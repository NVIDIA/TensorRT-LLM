# Suffix Automaton Speculative Decoding

This example demonstrates how to use the suffix automaton (SA) speculative decoding
implementation with TensorRT-LLM's MTP (Multi-Token Prediction) to boost acceptance rates.

## Overview

The suffix automaton is a compact state machine that recognizes all suffixes of a string.
It's used to find longest patterns in previously generated tokens to predict future tokens.
When combined with MTP, this hybrid approach can boost acceptance rates by up to 40%,
particularly for sequences with repetitive patterns.

## How It Works

1. **SA State Building**: When a new request is added, the suffix automaton is built
   from the context tokens on the host (CPU).

2. **Pattern Matching**: During generation, after accepting tokens from verification,
   the SA looks up the longest suffix in the current text that appeared earlier.

3. **Draft Token Selection**: If the match length exceeds a threshold, the SA draft
   tokens (tokens that followed the earlier occurrence) are used instead of MTP drafts.

## Usage

### Basic Usage

```bash
# Run with MTP only (baseline)
python run_sa_spec.py --model /path/to/deepseek-v3

# Run with SA+MTP (hybrid)
python run_sa_spec.py --model /path/to/deepseek-v3 --use_sa_spec
```

### Configuration Options

```bash
python run_sa_spec.py \
    --model /path/to/deepseek-v3 \
    --use_sa_spec \
    --sa_spec_threshold 4 \
    --num_nextn_predict_layers 1 \
    --max_new_tokens 256 \
    --batch_size 4 \
    --tp_size 8
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model` | Required | Path to model checkpoint |
| `--use_sa_spec` | False | Enable SA speculative decoding |
| `--sa_spec_threshold` | 4 | Min match length to use SA drafts |
| `--num_nextn_predict_layers` | 1 | Number of MTP layers |
| `--max_new_tokens` | 256 | Max tokens to generate |
| `--batch_size` | 1 | Batch size for generation |
| `--tp_size` | 1 | Tensor parallelism size |

## Expected Results

With repetitive or pattern-heavy text (code, structured data, etc.), you should see:

- **Without SA**: Baseline MTP acceptance rate (~60-70%)
- **With SA**: Improved acceptance rate (~80-90%) for repetitive content

The improvement is most noticeable when generating:
- Code with repeated patterns (loops, similar functions)
- Structured text (JSON, XML, Markdown)
- Text with repeated phrases or patterns

## Implementation Details

The suffix automaton implementation is integrated natively into TensorRT-LLM:

- **C++ Core**: `cpp/tensorrt_llm/kernels/speculativeDecoding/suffixAutomaton/`
  - `suffixAutomaton.h` - Suffix automaton data structure
  - `suffixAutomatonKernels.cu` - CUDA kernel for batch extension

- **Python Integration**: `tensorrt_llm/_torch/speculative/suffix_automaton.py`
  - `SuffixAutomatonState` - Single request SA state
  - `SuffixAutomatonManager` - Multi-request management
  - `SAResourceManager` - TRT-LLM resource manager integration

- **MTP Integration**: Modified `tensorrt_llm/_torch/speculative/mtp.py`
  - Calls SA extend after accepting tokens
  - Selects between SA and MTP drafts based on threshold

## References

- [Baseten SA Spec Blog Post](https://www.baseten.co/blog/suffix-automaton-speculative-decoding/)
- [Original sa_spec Implementation](https://github.com/basetenlabs/sa_spec)
