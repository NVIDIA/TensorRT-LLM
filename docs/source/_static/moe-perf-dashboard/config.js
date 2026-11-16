// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Dashboard UI + display config. Plain script; edit freely and refresh.
// Measurements live in data.js — replace that one file to update the docs.
window.MOE_DASHBOARD_CONFIG = {
  title: "MoE Perf Dashboard",
  subtitle: "Best parallel / backend / comm per (hardware, model, precision, world size, workload)",

  // The user-selectable scenario axes.
  //   source "index" -> options come from index (hardware/model/precision), narrowing to one file.
  //   source "data"  -> options come from the selected file's rows (world_size/workload).
  //   required:false  -> defaults to "all" and renders one result card per value.
  dimensions: [
    // Docs ship a single data version in data.js, so there is no Version control.
    { key: "hardware",   label: "Hardware",         source: "index", required: true },
    { key: "model",      label: "Model",            source: "index", required: true },
    { key: "precision",  label: "Precision",        source: "index", required: true },
    // Optional + fanout: leaving this on "All" renders one card per world size,
    // ordered 4 -> 8 -> 16 -> 32, which is how the optimum's drift across scale is read.
    // Cards never merge sizes, so rankings are still computed within a single size.
    { key: "world_size", label: "World Size",       source: "data",  required: false, type: "number" },
    // Attention parallel strategy (first letter of parallel_mode: T=attn-tp, D=attn-dp).
    // fanout:false -> pure filter; "All" merges strategies and re-ranks instead of
    // splitting into one card per strategy.
    // stableOptions: always offer every value in the file, never narrowed by the other
    // dropdowns. The control stays put so it can be reached without first arranging the
    // rest of the form; the cost is that an unmeasured combination is now selectable and
    // reports itself as unmeasured.
    // facetText: card titles print the value alone. The labels already read
    // "Attention-TP"/"Attention-DP", so prefixing the dimension name would say
    // "Attention" twice in the same breath.
    { key: "attn",       label: "Attention Parallel", source: "data", required: false, fanout: false,
      stableOptions: true, facetText: (v) => v },
    // Tokens arriving on ONE rank, derived in index.html as num_tokens / world_size.
    // Only attention-DP splits the instance total across ranks; under attention-TP
    // every rank sees all of them, so the quantity does not exist there — hence
    // showWhen, which also clears the selection when the axis goes away rather than
    // leaving an invisible filter in force. The merged "All" view is excluded too:
    // one local-batch axis across both attention plans would mean two things at once.
    { key: "local_batch", label: "Local Batch", source: "data",
      required: false, type: "number",
      showWhen: (sel) => sel.attn === "D",
      facetText: (v) => `Local batch ${v}` },
    // Workload as the sweep declares it: the raw bench_moe `num_tokens`, i.e. the token
    // total across the whole instance. Deliberately NOT normalized to tokens/rank — the
    // total is the figure the workloads and the disagg configs are written in, so a card
    // groups the sweep points that were actually run together.
    // Single-valued, unlike the standalone dashboard, which let a reader hold a SET
    // of workloads for comparison — a scratchpad behaviour that does not belong on a
    // documentation page. "All" is still here and still fans out one card per
    // measured value; it is a scalar dimension now, not a set.
    // facetText: the unit belongs next to the number on a card title
    // ("Workload 256 tokens"), not parenthesised in a form label.
    // Options are narrowed by the other data dims, so the list follows CUDA Graph —
    // the prefill (graph-off) and decode (graph-on) sweeps measured different totals.
    { key: "num_tokens", label: "Workload (tokens)", source: "data",
      required: false, type: "number",
      facetText: (v) => `Workload ${v} tokens` },
    // Required: an "All" that merges dispatch patterns is not a shape anything ran.
    // stableOptions: today every measured row is dispatch=balanced, and without the
    // exemption the single-value rule below would delete this control entirely. Holding
    // its place keeps the axis visible for the imbalanced data that will land later.
    { key: "dispatch",   label: "Dispatch",          source: "data", required: true,
      stableOptions: true },
    // Stage-derived: prefill workloads run CUDA-graph off, decode workloads on.
    // Optional, so "All" stays available.
    { key: "cuda_graph", label: "CUDA Graph",        source: "data", required: false,
      stableOptions: true },
  ],

  // Columns of the data that describe the recommended configuration.
  // emphasis:true -> shown as headline chips; the rest go in the detail table.
  configFields: [
    { key: "parallel_mode", label: "Parallel", emphasis: true },
    { key: "backend",       label: "Backend",  emphasis: true },
    { key: "comm",          label: "Comm",     emphasis: true },
    { key: "moe_ep_size",   label: "MoE EP" },
    { key: "moe_tp_size",   label: "MoE TP" },
    // num_chunks stays in the data (reserved) but is not displayed for now.
  ],

  // Measured columns. goal min/max sets ranking direction; objective is what we rank on.
  metrics: [
    { key: "score_ms", label: "MoE Latency", unit: "ms", goal: "min" },
  ],
  objective: "score_ms",

  // Curation over the index: what to surface and default selections.
  display: {
    topN: 3,
    hardwareOrder: ["GB200", "GB300", "B200", "B300", "H100"],
    hiddenFiles: [],
    // Backstop, not a curation rule: every real scenario fans out well under this
    // (107 cards is the widest dataset today). It exists to catch a future dataset
    // that explodes, not to stop you from opening "All workloads".
    maxFanout: 300,
    // The required data dims need a landing value, or the page opens on
    // "Choose a scenario" with nothing rendered. 512 tokens is measured in every
    // chunk of this data version; if a future scenario drops it, renderFilters
    // falls back to that dimension's first offered value.
    defaults: {
      hardware: "GB200", model: "deepseek_v4_pro", precision: "NVFP4",
      num_tokens: 512, dispatch: "balanced",
    },
  },
};
