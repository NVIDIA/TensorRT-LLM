---
name: trtllm-modeling-v2-host-offload
description: Load when a weight tensor is too large to keep resident and must live in host memory, with the forward gathering the rows it needs each step. An output-identical feature, which is exactly what makes it dangerous — both standing gates pass whether or not it is doing anything, so it needs a residency witness.
---

# trtllm-modeling-v2-host-offload

`--features host_offload`. One or more weight tensors stay in **pinned host
memory** for the life of the engine; the forward gathers the rows a step
actually needs and copies those to the device.

The first instance is Qwen3.8-Flash-Next's PLE n-gram embedding table:
`ngram_vocab_size_base: 20000000` at `ple_embed_dim: 2560` in FP8, which is
**53.7 GB of a 124 GB checkpoint**. Whether it is resident decides how much of
a 284 GB device is left for KV. That is the test for whether this feature
applies: *a single tensor whose size is comparable to the rest of the model*.
It is not a general memory-pressure remedy. A hashed n-gram memory table
larger than the rest of its checkpoint is the same case again.

Read `trtllm-modeling-v2-core/references/adapter-rules.md` first if the
offload is expressed as an adapter rather than inside the forward.

## The property that shapes everything else

**Offloading is output-identical by construction.** The same rows are gathered
and the same arithmetic runs; only where the bytes waited changes. So:

- the **boot** gate passes whether or not the tensor was offloaded;
- the **accuracy** gate passes whether or not the tensor was offloaded;
- a target that quietly loaded the table onto the device anyway reports a
  green feature unit.

This is the same shape of blindness the acceptance gate exists for, and it
wants the same answer: a third measurement that can only come out one way if
the feature is real. See **Gates** below.

## What this feature changes

**The manifest.** The offloaded tensor is materialized on CPU with
`pin_memory=True` rather than on the device. It must still be *covered* by the
weight manifest — an offloaded tensor is loaded, not skipped, and putting it in
the known-not-loaded set would be a lie that the coverage check cannot catch.
Record the residency, not just the presence.

**The forward gains a gather.** Two designs, below: a **zero-copy gather**
that the GPU issues against host memory mapped into its address space
(recommended, and CUDA-graph capturable), and a **host `index_select`** that
brings the indices to the host and copies the rows back (the fallback). Both
read the same rows, so the choice changes cost and capturability, never
output.

**Three things decide whether it is usable at all, and none is about
correctness:**

| decision | why it matters |
|---|---|
| pinned vs pageable host memory | pageable H2D on some driver/platform combinations degrades pathologically — `tensorrt_llm/_torch/pinned_weight_staging.py` documents one such case on GB300, a 36-minute to 4-hour stall with no error. Pinned is not an optimization here |
| which stream the copy runs on | a copy on the compute stream serializes the gather against everything else in the step |
| whether the index computation is host-visible | see below — this is the CUDA-graph question |

**CUDA graph capture.** A gather whose indices are read back to the host inside
the captured region cannot be captured: the host read is a synchronization the
graph cannot express. There are two honest resolutions and they are not
equivalent:

1. make the gather **device-driven** — a kernel indexes the host table through
   a device pointer to it — which is capturable, and is the design to reach
   for on Grace-Blackwell; see the next section;
2. keep the offloaded layer **outside** the captured region with a host
   `index_select` — states the feature's cost plainly, and is the fallback when
   the device cannot read host memory well enough (see the last section).

Whichever is chosen goes in the run journal beside the identity record, because
a reader who assumes capture will misread every later perf number. **Do not
report the target as CUDA-graph enabled without saying which layers are in the
graph.**

## The capturable design: a zero-copy gather through UVA

**The in-tree precedent is `tensorrt_llm/_torch/modules/qwen4_exp/ple.py`**, the
Qwen4-Exp PLE table. Read it before writing the gather; each piece below is a
problem it already solved.

- **An environment switch, parsed strictly.** `TRTLLM_QWEN4_EXP_PLE_HOST_OFFLOAD`,
  read by `_uses_ple_host_offload`, which raises on anything that is not a
  boolean rather than reading a typo as "off". The residency witness's
  `contrast_env` is exactly this switch, so it has to mean one thing.
- **One stable pinned allocation.** `Qwen4ExpPinnedHostEmbedding` creates the
  table on `meta` and materializes it pinned exactly once
  (`materialize_pinned`); its `_apply` override keeps `.to(device)` and
  `.cuda()` from moving or replacing the table. A mapped pointer to memory
  that later moves is a pointer to freed memory.
- **The device pointer is resolved before capture.** `_mapped_device_ptr`
  calls `cudaHostGetDevicePointer` once per device, caches it, and refuses to
  do so while a stream is capturing. So warmup must touch the gather before
  the first capture — a graph cannot record the lookup, only its result.
- **A Triton zero-copy gather.** `_gather_pinned_embedding_kernel` loads the
  rows straight from pinned host memory through the mapped pointer — no
  staging copy, no host sync — writes zeros for rows another rank owns, and
  applies the FP8 scale in the same pass.
- **Row sharding plus a reduction.** Under tensor parallelism each rank holds
  `ceil(rows / tp)` rows and the partial lookups are summed with an
  `AllReduce` (`Qwen4ExpNGramEmbedding`, `_finish_embedding_lookup`). Under
  attention DP each rank serves its own tokens, so `_prepare_embedding_lookup`
  **allgathers the ids**, every rank gathers the rows it owns for all of them,
  and a **reduce-scatter** returns each rank its own tokens' sums.
- **A prefetch stream.** `Qwen4ExpPLE.start_prefetch` issues the gather on a
  side stream before execution reaches the layer, so the host read overlaps
  the layers before it.

### The shared single-copy layout (recommended on Grace-Blackwell)

Row sharding holds `1/tp` of the table per rank — every byte of it held once
on the node — but pays a collective per lookup. The alternative removes the
collective and still holds the table **once per node, not once per rank**:
one host copy that every rank on the node maps and registers.

- rank 0 creates the table in one anonymous shared region (`memfd_create`),
  backed by huge pages (`madvise(MADV_HUGEPAGE)`);
- the other ranks on the node map the same region; reaching it through
  `/proc/<pid>/fd/<n>` needs every rank in one PID namespace, so one
  container per node;
- every rank `cudaHostRegister`s the mapping and resolves its own device
  pointer, before capture as above;
- every GPU can then gather **any** row: no sharding, no zero-fill, no
  allreduce or reduce-scatter, and the lookup is bitwise the resident one.

For a table of `T` bytes on a node of `G` GPUs, per-rank copies cost `G x T`
of host memory and the shared layout costs `T`, once. Where TRT has no shared
layout yet, it is new target code, and the `ple.py` pieces above are the parts
it can follow.

What to check before choosing it, none of which is about correctness:

| check | why |
|---|---|
| huge pages | without them expect lookups an order of magnitude slower: TLB misses on a mapping of hundreds of GB |
| NUMA | a GB300 node puts four GPUs on two Grace sockets; one copy lives on one socket (or interleaved), and the far GPUs read it over the socket link |
| host RAM | the table once, plus pinned staging for weight loading, must fit beside everything else the node holds |
| lifetime | rank 0 owns the region; the others must map it before rank 0 can exit, and a restarted rank 0 invalidates every mapping |

The residency witness is unchanged by the layout — it compares per-rank
device memory with the offload on and off — but `expected_mib` is the OFF
leg's **per-rank** footprint: a table row-sharded at TP4 contributes a
quarter of its bytes, in MiB, per rank.

### The fallback: a host `index_select`

Per step: compute the row indices on device, bring them to host,
`index_select` on the host tensor, copy the result back. Two host-device round
trips per invocation, and a host read the graph cannot capture — so the layer
stays outside the captured region. It needs no mapped pointer and no
registration, which makes it the path for a platform where device reads of
host memory are slow, and the first thing to fall back to when the zero-copy
gather is suspected.

## Gates

boot → accuracy → **residency witness**, all under
`modeling_v2=require`.

The witness is the gate that can actually see this feature. Shape it so only
one outcome is possible if the feature is real:

- run the same workload twice in the same session, once with the feature and
  once with `contrast_env` turning it off, and read **device memory after the
  weights load and before the KV cache pool is allocated**. Only the target
  knows that moment, so the target prints it, once per rank:
  `__DEVICE_MEMORY_MIB__ <MiB>`, `torch.cuda.memory_allocated()` in MiB (from
  `post_load_weights`, say). Not after the engine reports ready, and not from
  `nvidia-smi`: the KV pool is sized from the memory left, so by then it has
  taken what the offload saved and the two legs read alike. A leg without the
  line fails the witness;
- assert the delta is **at least the offloaded tensor's own size**, computed
  from the checkpoint rather than from the run;
- assert the generated text is **token-identical** between the two, greedy. If
  it is not, the gather is wrong and the accuracy gate's tolerance was wide
  enough to hide it. Both legs print their token ids under
  `TRTLLM_WITNESS_FULL_OUTPUT=1` (`[i] Witness token ids: [...]` from the
  tree's `examples/llm-api/quickstart_advanced.py`); text that agrees without
  them is not a pass.

A witness that only checks "the run still works" certifies nothing. Write the
expected delta down from the checkpoint's own shapes *before* running it.

## What the gates still do not tell you

Whether the per-step gather is fast enough to be worth having. That is a perf
question and it belongs to tuning, not to onboarding. Onboarding's claim is
"the table is not resident and the output is unchanged" — say exactly that and
no more.

## Report additions

- the tensor(s) offloaded, each with its checkpoint shape, dtype and byte size
- measured device memory with and without, and the delta against the predicted
  one
- confirmation of token-identical output, and at what sampling settings
- which layers are inside the CUDA graph and which are not, and why
- which gather design — zero-copy (and which layout: sharded or shared) or
  host `index_select` — and which stream it runs on
