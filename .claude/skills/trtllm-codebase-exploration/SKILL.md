---
name: trtllm-codebase-exploration
tags: [tensorrt-llm, workflow, exploration]
description: >
  Systematic approach to exploring the TensorRT-LLM codebase before implementing
  new features or optimizations. Teaches how to discover existing infrastructure,
  trace code paths, and avoid reimplementing what already exists. Derived from
  real mistakes where ~250 lines of code were written and deleted because
  existing forward methods weren't discovered upfront.
  Use when starting any new feature, optimization, or code modification in TRT-LLM.
license: Apache-2.0
metadata:
  author: NVIDIA Corporation
---

# TensorRT-LLM Codebase Exploration Guide

## Why This Matters

TRT-LLM is a large codebase (~500K lines) with many reusable abstractions. The most common source of wasted effort is reimplementing something that already exists. On the short-seq MHA branch, ~250 lines were written across 4 iterations before discovering that a 10-line dispatch to an existing method (`forward_context_default`) was the right solution.

**Rule of thumb**: Spend 30 minutes reading existing code before writing 1 line of new code.

## MANDATORY: Ignore the TensorRT backend, focus on the PyTorch backend

## Step-by-Step Exploration Workflow

### Step 1: Map the Class You're Modifying

Before adding code to a class, understand its full structure:

```bash
# List all methods (not just forward*)
grep -n "def " tensorrt_llm/_torch/modules/attention.py | head -50

# List all attributes set in __init__
grep -n "self\." tensorrt_llm/_torch/modules/attention.py | grep "__init__" -A 200 | head -80

# Find the class hierarchy
grep -n "class MLA\|class Attention\|class TrtllmAttention" tensorrt_llm/_torch/modules/attention.py
```

### Step 2: Trace Existing Forward Methods

Read EVERY forward method in the class. Understand what each one does, what inputs it expects, and what backends it uses.

```bash
# Find all forward methods
grep -n "def forward" tensorrt_llm/_torch/modules/attention.py

# For each one, read the full implementation (not just the signature)
```

**Ask yourself:**
- Does any existing forward method already compute what I need?
- Can I dispatch to an existing method by setting up the right state?
- What would I need to change (attributes, guards, assertions) to reuse it?

### Step 3: Search for Existing Backends and Utilities

| What you need | Search for | Common hits |
|--------------|-----------|-------------|
| Attention computation | `TrtllmAttention`, `create_attention`, `FlashInferAttention` | Handles packed seqs, variable lengths, KV cache natively |
| Compiled fusion | `maybe_compile`, `maybe_compiled_cat`, `maybe_compiled_copy_` | Already in `tensorrt_llm/_torch/utils.py` |
| RoPE application | `RotaryEmbedding`, `apply_rotary_pos_emb`, `rope_fusion` | Multiple implementations exist; check which one the current code path uses |
| KV cache management | `mla_rope_append_paged_kv`, `append_paged_kv`, `latent_cache` | Fused RoPE + cache operations in C++ kernels |
| Sparse attention | `DSATrtllmAttention`, `indexer`, `topk_indices` | DSA-specific backend with sparse routing |

```bash
# Generic search pattern
grep -rn "KEYWORD" tensorrt_llm/_torch/ --include="*.py" | head -20
```

### Step 4: Check What the Fused Kernels Handle

Many operations you might implement manually are already handled by fused C++ kernels:

```bash
# Find what the attention kernel handles internally
grep -rn "latent_cache\|rope.*fuse\|rope_fusion" tensorrt_llm/_torch/attention_backend/
```

**Common surprise**: When `rope_fusion=True` (`apply_rotary_emb=False`), the fused attention kernel handles RoPE internally via `latent_cache`. Writing custom RoPE code in Python is unnecessary and will double-apply RoPE.

### Step 5: Check Assertions and Invariants

Existing assertions may need updating when you add a new code path. Don't work around them — change them if your new path makes them invalid:

```bash
# Find assertions in the class
grep -n "assert " tensorrt_llm/_torch/modules/attention.py
```

**Example**: DSA models had `assert self.mha is None`. When adding short-seq MHA (which creates `self.mha` for DSA models), the assertion was changed to `assert self.mqa is not None` — the actual invariant being tested.

### Step 6: Understand Weight Layouts

Weight layouts often differ between HuggingFace checkpoints and TRT-LLM's loaded format:

```bash
# Find weight loading/transformation code
grep -rn "load_.*weight\|weight.*transform\|load_kv_b_proj" tensorrt_llm/_torch/models/

# Check how weights are laid out after loading
grep -n "def load_" tensorrt_llm/_torch/models/modeling_deepseekv3.py
```

**Critical for tests**: Always initialize test weights in the **loaded layout**, not the HF checkpoint layout.

### Step 7: Trace Method Limitations

After identifying a method to reuse, understand what it does **NOT** handle:

```bash
# Find all callers of the method to see its dispatch context
grep -rn "forward_context_default\|forward_context(" tensorrt_llm/_torch/modules/attention.py

# Look for the dispatcher that routes to this method
# Often named similarly but without a suffix (e.g., forward_context dispatches to forward_context_default)
```

**Ask yourself:**
- What scenarios does this method handle? (fresh prefill? cached KV? chunked context?)
- What scenarios does it NOT handle?
- Is there a higher-level dispatcher that routes to this method for the correct subset of cases?
- If I call this method directly, which scenarios will I silently mishandle?

**Example:** `forward_context_default()` handles fresh prefill but does NOT attend over cached KV tokens. `forward_context()` is the dispatcher that routes to `forward_context_default`, `forward_context_with_cached_kv`, or `forward_context_with_chunked_prefill` based on context state and SM version. Calling `forward_context_default` directly during chunked context silently drops cached tokens.

## Key Discovery Patterns

### Pattern: "Can I Reuse an Existing Forward Method?"

1. Read the target forward method (e.g., `forward_context_default`)
2. Compare it to what your new code path needs to do
3. If >70% overlap, dispatch to the existing method instead of writing a new one
4. Adjust attributes/state in `__init__` to make the dispatch work

### Pattern: "Is This Already Handled by a Fused Kernel?"

1. Check if the operation is in the attention backend's scope
2. Check the `apply_rotary_emb` / `rope_fusion` flag
3. Check `latent_cache` handling
4. If the fused kernel handles it, DON'T reimplement in Python

### Pattern: "Am I Calling the Right Abstraction Level?"

1. Identify the method you plan to call
2. Search for methods that CALL this method — there may be a dispatcher above it
3. Check if the dispatcher handles edge cases your direct call would miss
4. Prefer calling the dispatcher over the specific handler

```bash
# Find what calls forward_context_default to discover the dispatch chain
grep -n "forward_context_default" tensorrt_llm/_torch/modules/attention.py
```

### Pattern: "Does a Utility Already Exist?"

1. Search `tensorrt_llm/_torch/utils.py` for compiled helpers
2. Search `tensorrt_llm/_torch/modules/` for module-level utilities
3. Search test fixtures in `tests/unittest/_torch/` for test setup patterns

## Common Exploration Mistakes

| Mistake | Consequence | Prevention |
|---------|------------|------------|
| Reading only the method you're modifying | Miss that another method does what you need | Read ALL methods in the class |
| Searching only for the exact function name | Miss equivalent implementations | Search for the *concept* (e.g., "attention", "rope", "expand kv") |
| Assuming assertions are immutable | Work around them with hacks (separate attributes) | Question whether the assertion's intent still applies |
| Not reading the fused kernel's capabilities | Reimplement what it already does | Check what `latent_cache`, `rope_fusion` etc. control |
| Only reading Python code | Miss C++ implementations called via bindings | Check `tensorrt_llm/_torch/attention_backend/` for native kernels |
| Calling a method directly instead of through its dispatcher | Miss edge cases (cached KV, chunked prefill, SM-version gating) | Search for callers of the method to find the dispatch chain |
| Assuming hardware-uniform numerical behavior | Silent accuracy degradation on specific SM versions | Check for `get_sm_version()` guards near the call site; test on multiple hardware |

## File Reference for Exploration

| Area | Key files to read |
|------|-------------------|
| Attention modules | `tensorrt_llm/_torch/modules/attention.py` |
| Attention backends | `tensorrt_llm/_torch/attention_backend/` (trtllm_attention.py, sparse/) |
| Model definitions | `tensorrt_llm/_torch/models/modeling_*.py` |
| Utilities | `tensorrt_llm/_torch/utils.py` |
| RoPE | `tensorrt_llm/_torch/modules/rotary_embedding.py` |
| Test fixtures | `tests/unittest/_torch/attention/` |
| Weight loading | `tensorrt_llm/_torch/models/modeling_deepseekv3.py` (search `load_`) |
