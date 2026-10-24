# Contract evidence — what makes a sentence in an entry defensible

An entry's contract is its `OpWrapper` class: `__call__`, `reference`,
`is_valid` and `CELLS` are executable and the test keeps them true; the prose
around them — the class docstring, each cell's `why`, the `note` — is checked
by nothing but you. Whoever calls the op reads the class and nothing else, so
a sentence that is *almost* true is worse than a missing one — the caller acts
on it. Prefer moving a claim into a member (a cell, a guard, a reference) over
writing it down; what stays prose is what this page is about.

Three failure modes have each been found repeatedly in this catalog. All are about
evidence, not about prose quality.

**Write the path the cells certify, not the path the source describes.**
The one failure mode that survives review is a contract sentence that is
true of *some* execution path, just not the one the test drove. Four found
in this catalog so far: norm Preconditions describing the CUDA-JIT path while
the cells certified CuTe DSL; `thop_attention`'s fp8 `kv_scale_*` rule
transcribed from an upstream comment about XQA kernels, on an arch whose
fp8 decode runs trtllm-gen JIT; `fused_moe`'s "tactic choice does not
change results", true of the cold autotuner the test runs in and false of
the warm one serving runs in; and MLA's "mixed batches are rejected",
which is a check in the Python backend that this op bypasses. Before
writing a precondition into the `note`, name the path you measured — backend,
kernel family, cache state — and say so if it is not the only one.

**A claim that the op rejects something is the expensive kind to get
wrong**, because it promises a guard the caller then does not write. Every
"raises" / "is rejected" / "not survivable" sentence needs a probe that
actually saw the raise, in the certified configuration, with the error
text quoted — ideally as a test (`pytest.raises` with the message). Absent
that, write what you measured: accepted, and what it then does. An input the
op accepts and answers wrongly is not a sentence at all: it is `is_valid`.

**Describe what the engine does and name where to look — never transcribe
what its code says.** An entry records things learned about TensorRT-LLM,
and how you write them down decides whether they are still true next month.
The engine moves every day; an entry's prose does not. So anything you copy *out*
of engine source — a line number, a pasted excerpt, an argument list
transcribed by hand — is wrong shortly after you write it, silently, and the
next reader believes it because it reads like a citation.

This is measured, not a worry. Seven line-number citations were written into
this catalog's former contract files and **all seven pointed at the wrong line within days** — one
set off by about 35 lines, another by about 1100, landing on an unrelated
f-string. In the very same sentences, every file path and every symbol name
still resolved exactly. The durable pointer and the rotting one sat side by
side, and only one of them rotted.

So write two things and stop:

- **the behaviour**, in your own words, at the level the caller needs — what
  it does, when, and what it means for the call you are making;
- **the pointer**: a file path plus a *name* the reader can grep — a
  function, a class, a field, a config key, or the exact assert text.

| instead of | write |
|---|---|
| `` `_torch/pyexecutor/model_engine.py:1103,1278` `` | warmup calls `_run_autotuner_warmup` (`_torch/pyexecutor/model_engine.py`) |
| `` `llmapi/llm_args.py:4669` `` | `enable_autotuner`, a `TorchLlmArgs` field, defaults to `True` |
| `` `attentionOp.cpp:1045` `` | the `seqQOffset is nullptr.` guard in `common/attentionOp.cpp` |
| a pasted 20-line excerpt of the op's C++ | one sentence on what it does, and the file to read |

The one thing that keeps its line number is a **quoted diagnostic**:
`RuntimeError: ... (../tensorrt_llm/runtime/torchUtils.h:113)` is the message
the engine prints, so it is what someone greps for when they hit it. Quote it
whole and verbatim. That is a symptom, not a direction to navigate.

`tests/unittest/_torch/modeling_v2/test_modeling_v2_no_stale_claims.py` scans
every file of the package and fails on a pinned version or a measurement date.
Line-number citations and pasted excerpts are not machine-checked: they are on
you.

## Before writing any precondition, answer these questions

1. **Which execution path did the cells actually drive?** Backend, kernel
   family, autotuner cache state. If the sentence is true of another path, say
   which.
2. **Did I see it, or did I read it?** An upstream comment is a hypothesis
   about the code, not a measurement of this arch.
3. **If this says the op refuses something, where is the probe that saw the
   refusal?** In the certified configuration, with the error text quoted. No
   probe, no claim — write what you measured instead: accepted, and what it
   then does.
4. **Will this sentence still be true after someone refactors the engine?**
   If it depends on a line number, a pasted excerpt, or an argument order you
   copied, it will not. Keep the behaviour and the name; drop the rest.
