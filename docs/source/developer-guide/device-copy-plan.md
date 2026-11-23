(device-copy-plan)=

# Per-Step Device Copies: the Copy Plan

Between two decode CUDA graph replays, the next step's inputs and metadata reach their graph-stable device buffers as
a chain of small copies (position ids, KV lengths, page tables, the graph runner's static inputs, speculative RNG
seeds). Each one is a separate operation on the stream that the next graph waits for, so the GPU idles for the whole
chain. The copy plan in `tensorrt_llm/_torch/pyexecutor/device_copy.py` is the runtime's mechanism for these copies.
New runtime code that issues per-step copies uses it.

## The mechanism

Two questions decide what happens to a per-step copy, and two different owners answer them.

* **When the copy must land** is decided by the model, because it depends on who reads the destination. Every copy in
  shared runtime code names its call site with a `CopySite` and goes through `device_copy.copy` (or
  `device_copy.fill_tail`). A model declares a `CopyPlan` as `copy_plan` on the model:
  * `merge`: the sites whose copies are staged into the step's pack. The pack is issued as one upload and one scatter
    launch, at the latest when the graph replays.
  * `flush_after`: barrier sites, whose staged copy must land before anything that runs next.
  * `parallel_scatter`, `changed_span` and `final_flush=InGraph(...)`: how the pack is issued. `InGraph` makes the
    replayed graph's own prologue issue it.
* **Whether the copy is needed at all** is decided by the buffer owner, because it depends on the values and on who
  else writes the buffer. The owner passes `unchanged=True` when the destination already holds the values, or keeps a
  `device_copy.DeviceCopyBuffer` with the copy's site as its record of what the buffer holds.

`TLLM_DEVICE_COPY_CHECK=1` verifies at runtime what a plan and its owners assert:
* that nothing writes a merged destination before its flush;
* that every skipped (`unchanged`) copy really holds its values.

The unit tests run with this check on (`tests/unittest/conftest.py`).

## Adopting the copy plan in a model

1. Name the model's own copy sites in a `CopySiteEnum` of its own, next to the code that issues them. Shared sites are
   in `device_copy.CopySite`.
2. Build the plan from the readers of each destination:
   * merge what no device work touches before the graph replay;
   * anchor a `flush_after` barrier on each copy that an eager reader consumes;
   * set the plan as the model's `copy_plan`.
3. Let each buffer owner skip its unchanged uploads on its own, with a `DeviceCopyBuffer` that has the copy's site.
4. Turn off the model's `RuntimeHooks` step-input pack switches (`pack_step_inputs`, `skip_unchanged_h2d` and the
   `pack_*` options). The engine refuses a model that declares a plan and also turns these on (`check_one_copy_path`).
5. Run the model's tests with `TLLM_DEVICE_COPY_CHECK=1`, and compare its step time with the step-input pack it
   replaces.

## Coexistence with the step-input pack

The copy plan replaces an older mechanism:
* the step-input pack, `pyexecutor/step_inputs.py`;
* the skip cache, `pyexecutor/h2d_cache.py`;
* `tensorrt_llm/_device_copy.DeviceCopyBuffer`;
* the `RuntimeHooks` switches that turn them on.

Both mechanisms are supported while models move over:

* A model that declares no copy plan has no copier, and its copies run exactly as before. A shared call site that has
  a `CopySite` takes its original path whenever no copier is active (`device_copy.current()` is None).
* A model that declares a plan has a copier. The shared sites route through it, and the step-input pack stays off.
* A model uses one mechanism for all of its step's copies, never both.

The rules while both exist:

* New per-step copies in runtime code go through `device_copy` with a named site.
* Do not add new users of `step_inputs`, `h2d_cache`, `tensorrt_llm/_device_copy` or the `RuntimeHooks` step-input
  pack switches. Existing users keep working until their model moves to a copy plan.
* When a shared call site gains a `CopySite`, keep its original path for engines without a copier, so that models
  without a plan see no behavior or performance change.
* Once every model has moved, a follow-up change removes the step-input pack, the skip cache and their `RuntimeHooks`
  switches.

The same approach applies to any shared runtime mechanism that a new one replaces:
1. Add the new mechanism beside the old one, opt in and off by default.
2. Move the models in follow-up changes.
3. Remove the old mechanism last.
