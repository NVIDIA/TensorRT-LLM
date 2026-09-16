// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/* MoE Perf Dashboard — in-page widget logic.
 *
 * Rendered by the `trtllm_moe_dashboard` directive (docs/source/_ext/), alongside
 * moe_dashboard.css. Loaded only on the page carrying that directive, after
 * moe-perf-dashboard/config.js and moe-perf-dashboard/data.js, whose globals
 * it consumes.
 *
 * Ported from the standalone MoE Perf Dashboard (moe-perf-analysis/web/index.html).
 * The logic is unchanged apart from four adaptations to living inside a page
 * rather than owning the document, each marked `PORTED:` below.
 */
// PORTED: the original ran from the end of <body>, so the DOM was always there.
// Sphinx injects this into <head>, so boot is deferred until the placeholder the
// directive emits actually exists — the same guard config_selector.js uses.
function moeDashboardMain() {
  "use strict";
  const CONFIG = window.MOE_DASHBOARD_CONFIG;
  const DATA = window.MOE_DATA;
  // PORTED: element ids are namespaced (moe-results, moe-filters, ...) so they
  // cannot collide with ids the documentation theme owns. Every call site
  // below passes the bare name and is unchanged.
  const $ = (id) => document.getElementById("moe-" + id);
  const el = (tag, cls, text) => {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text != null) n.textContent = text;
    return n;
  };
  const svg = (paths, extra) => {
    const s = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    s.setAttribute("viewBox", "0 0 16 16");
    s.setAttribute("fill", "none");
    s.setAttribute("stroke", "currentColor");
    s.setAttribute("stroke-width", extra?.width ?? "1.5");
    s.setAttribute("stroke-linecap", "round");
    s.setAttribute("stroke-linejoin", "round");
    s.setAttribute("aria-hidden", "true");
    for (const d of paths) {
      const p = document.createElementNS("http://www.w3.org/2000/svg", "path");
      p.setAttribute("d", d);
      s.append(p);
    }
    return s;
  };

  // PORTED: theme is still not scripted, but the switch moved. The stylesheet now
  // resolves every colour through the documentation theme's --pst-color-* variables,
  // so the widget follows the page's light/dark control instead of the OS preference.

  if (!CONFIG || !DATA) {
    $("results").append(placeholder(
      "Data not loaded",
      "config.js or data.js is missing. Replace data.js to refresh the measurements, then reload."));
    return;
  }

  // Rows live in data.js (`MOE_DATA.files`), keyed by the index `file` field.
  // Docs ship one version, so there is nothing to fetch on scenario change.
  const loadChunk = (file) => {
    const rows = DATA.files?.[file];
    if (rows) return Promise.resolve(rows);
    return Promise.reject(new Error(`data.js has no rows for ${file}`));
  };

  const labels = DATA.labels || {};
  const hidden = new Set(CONFIG.display?.hiddenFiles ?? []);
  const index = (DATA.index || []).filter((r) => !hidden.has(r.file));
  const label = (kind, v) => labels[kind]?.[v] ?? String(v);

  // The index dims as one sentence, for the status line under the cards.
  const scopeText = () => indexDims()
    .map((d) => label(d.key, selection[d.key]))
    .join(" · ");
  // How a dimension names itself on a card title. Default is "<Dimension> <value>";
  // a dim with facetText writes its own phrasing, because the form label and the
  // card sentence are not the same register — "Workload (tokens)" is a control,
  // "Workload 256 tokens" is a caption.
  const facetText = (dim, raw) => {
    const v = label(dim.key, raw);
    return dim.facetText ? dim.facetText(v) : `${dim.label} ${v}`;
  };
  const MAX_FANOUT = CONFIG.display?.maxFanout ?? 12;

  const indexDims = () => CONFIG.dimensions.filter((d) => d.source === "index");
  const dataDims = () => CONFIG.dimensions.filter((d) => d.source === "data");

  const selection = {};   // dimension key -> chosen value
  let dataRows = [];      // rows of the currently resolved file

  // Data dims that apply to the current selection. A dim with showWhen can be
  // meaningless under some selections (local batch has no meaning under
  // attention-TP), and a hidden dim MUST NOT keep filtering: an invisible
  // constraint shows the reader an empty result with no control to explain it.
  // Clearing here, at the single point every caller goes through, is what makes
  // "hidden" and "not applied" the same state.
  const visibleDims = () => dataDims().filter((d) => {
    if (!d.showWhen) return true;
    if (d.showWhen(selection)) return true;
    delete selection[d.key];
    return false;
  });

  // Tokens on ONE rank. Derived here rather than in bundle.py: the 14-column CSV
  // schema is kept in lockstep across aggregate/validate/bundle/config.js, and a
  // display-only axis does not earn a change to that contract.
  //
  // Only attention-DP splits the instance total across ranks. A split that is not
  // integral is left null rather than rounded — a fractional local batch is not a
  // shape anything ran, and null keeps those rows out of the options list and out
  // of any local-batch filter, while they stay visible when the axis is "All".
  const deriveLocalBatch = (rows) => {
    for (const r of rows) {
      if ("local_batch" in r) continue;   // rows are shared objects in the chunk cache
      const nt = r.num_tokens, ws = r.world_size;
      r.local_batch = (r.attn === "D" && ws > 0 && nt % ws === 0) ? nt / ws : null;
    }
    return rows;
  };

  // The scenario survives a reload. Without this the engineer who refreshes lands
  // back on "Choose a scenario" and re-picks four dropdowns to get where they were.
  const SEL_KEY = "moe-dashboard-selection";
  const saveSelection = () => {
    try { localStorage.setItem(SEL_KEY, JSON.stringify(selection)); } catch (e) { /* file:// */ }
  };
  const savedSelection = () => {
    try { return JSON.parse(localStorage.getItem(SEL_KEY) || "null"); } catch (e) { return null; }
  };

  function placeholder(head, body) {
    const box = el("div", "placeholder");
    box.append(el("h2", null, head), el("p", null, body));
    return box;
  }

  // PORTED: the original let a reader pin a card to hold it on the page while
  // the filters moved on. Dropped here — the documentation page is a lookup
  // surface, not a comparison scratchpad, and the control added a mode with no
  // way to explain itself in context.
  const setStatus = (text) => { $("status").textContent = text; };

  // --- merged view ---------------------------------------------------------
  // One sorted table instead of one card per scenario. Session-scoped and never
  // persisted: it is a way of reading the current selection, not part of it.
  //
  // Offered ONLY once a specific local batch is chosen, and that is a
  // precondition rather than a gate. A fixed local batch means every row did the
  // same amount of work per rank, which is what makes absolute latencies from
  // different world sizes and workloads comparable at all. Ranking a mixture of
  // token scales by raw milliseconds would just sort by size.
  let mergeView = false;
  let mergeBtn = null;
  const localBatchDim = () => CONFIG.dimensions.find((d) => d.key === "local_batch");
  const mergeAvailable = () => {
    const d = localBatchDim();
    return !!d && (!d.showWhen || d.showWhen(selection)) && hasSel(d);
  };
  // A control that has gone away must not keep changing the page — the same rule
  // visibleDims() applies to hidden dimensions. Otherwise clearing the local batch
  // leaves a merged table on screen with no toggle anywhere to undo it.
  const syncMerge = () => {
    const on = mergeAvailable();
    if (!on) mergeView = false;
    if (!mergeBtn) {
      mergeBtn = el("button", "merge-toggle", "Merge");
      mergeBtn.type = "button";
      mergeBtn.title = "Merge every scenario into one table ranked by absolute latency";
      mergeBtn.addEventListener("click", () => { mergeView = !mergeView; renderResults(); });
      $("status").parentNode.append(mergeBtn);
    }
    mergeBtn.hidden = !on;
    mergeBtn.setAttribute("aria-pressed", String(mergeView));
    mergeBtn.classList.toggle("on", mergeView);
    return mergeView;
  };

  // --- data helpers -------------------------------------------------------
  const distinct = (rows, key, numeric) => {
    const vals = [...new Set(rows.map((r) => r[key]))];
    return numeric ? vals.sort((a, b) => Number(a) - Number(b)) : vals.sort();
  };
  // A selection value is either one value or, for a `multi` dimension, a set of
  // them. Empty (""/null/[]) always means "All" — no narrowing on that key.
  const filterRows = (rows, sel) =>
    rows.filter((r) => Object.entries(sel).every(([k, v]) => {
      if (v === "" || v == null) return true;
      if (Array.isArray(v)) return !v.length || v.some((x) => String(r[k]) === String(x));
      return String(r[k]) === String(v);
    }));

  // Selection accessors. Every dimension is read as a LIST of chosen values, so
  // the single/multi difference lives here instead of at fifteen call sites.
  // Wrapping a scalar also means a selection persisted before this dimension went
  // multi still restores cleanly.
  const selValues = (dim) => {
    const v = selection[dim.key];
    if (v == null || v === "") return [];
    return (Array.isArray(v) ? v : [v]).filter((x) => x !== "" && x != null);
  };
  const hasSel = (dim) => selValues(dim).length > 0;
  const matchesSel = (dim, rowVal) => {
    const vals = selValues(dim);
    return !vals.length || vals.some((v) => String(rowVal) === String(v));
  };
  // Group rows by the given dims (one card per distinct combination), sorted by
  // each dim in turn (numeric where the dim is numeric).
  const groupRows = (rows, dims) => {
    const g = new Map();
    for (const r of rows) {
      const key = dims.map((d) => r[d.key]).join("␟");
      (g.get(key) ?? g.set(key, []).get(key)).push(r);
    }
    return [...g.values()].sort((A, B) => {
      for (const d of dims) {
        const a = A[0][d.key], b = B[0][d.key];
        const cmp = d.type === "number" ? Number(a) - Number(b)
                                         : String(a).localeCompare(String(b));
        if (cmp) return cmp;
      }
      return 0;
    });
  };
  const rankGroup = (rows, objective, goal) => {
    const valued = rows.filter((r) => r[objective] != null);
    // Always rank by the objective: the precomputed `rank` column is scoped per
    // attention strategy (T/D), so ranks collide across strategies and cannot
    // order a merged "All" view. `rank` is kept in the data only as a reference.
    const dir = goal === "max" ? -1 : 1;
    return [...valued].sort((a, b) => dir * (a[objective] - b[objective]));
  };

  // --- index-dim options (values valid given the other index dims) --------
  const matchingFile = () => {
    const m = index.filter((r) =>
      indexDims().every((d) => !selection[d.key] || r[d.key] === selection[d.key]));
    return m.length === 1 ? m[0].file : null;
  };
  const indexOptions = (dim) => {
    const others = indexDims().filter((d) => d.key !== dim.key);
    const rows = index.filter((r) =>
      others.every((d) => !selection[d.key] || r[d.key] === selection[d.key]));
    let vals = [...new Set(rows.map((r) => r[dim.key]))];
    const order = CONFIG.display?.[dim.key + "Order"];
    vals.sort(order
      ? (a, b) => (order.indexOf(a) - order.indexOf(b)) || a.localeCompare(b)
      : (a, b) => String(a).localeCompare(String(b)));
    if (dim.sortDesc) vals.reverse();
    return vals.map((v) => ({ value: v, text: label(dim.key, v) }));
  };

  // --- rendering ----------------------------------------------------------
  // PORTED: single-valued dimensions render as a segmented row of option buttons
  // rather than a dropdown, matching the recipe selector on the deployment-guide
  // page — every value is visible without opening anything, and the chosen one is
  // marked by fill rather than by having to read the closed control. A dimension
  // with more options than fit that treatment keeps the dropdown.
  const BUTTON_MAX = 8;

  const makeField = (dim) => {
    const wrap = el("div", "filter");
    wrap.setAttribute("role", "group");
    wrap.setAttribute("aria-label", dim.label);
    wrap.append(el("span", "filter-label", dim.label));
    return wrap;
  };

  const makeSelect = (dim, options, current, onChange) => {
    const wrap = makeField(dim);
    // "All" is an option like any other, first in the row.
    const all = dim.required ? [] : [{ value: "", text: dim.allLabel ?? "All" }];
    const entries = [...all, ...options];

    if (entries.length <= BUTTON_MAX) {
      const group = el("div", "filter-options");
      for (const o of entries) {
        const b = el("button", "opt", o.text);
        b.type = "button";
        const on = String(o.value) === String(current ?? "");
        b.dataset.status = on ? "active" : "available";
        b.setAttribute("aria-pressed", String(on));
        b.addEventListener("click", () => { if (!on) onChange(String(o.value)); });
        group.append(b);
      }
      wrap.append(group);
      return wrap;
    }

    const sel = el("select");
    if (dim.required && !current) {
      // A required dim has no "All" entry, so an empty selection would match no
      // option and render the control blank — unreadable, and it never fires a
      // change event. Name the ask instead.
      const ph = new Option("Select…", "");
      ph.disabled = true;
      sel.append(ph);
    }
    for (const o of entries) sel.append(new Option(o.text, o.value));
    if (current != null) sel.value = current;
    sel.addEventListener("change", () => onChange(sel.value));
    wrap.append(sel);
    return wrap;
  };

  // A checkbox panel rather than <select multiple>: the list runs to 70 workloads,
  // and a native multi-select both loses the whole set to one stray click and
  // refuses the padding/height every other control here uses.
  //
  // onChange(values, done) — `done` marks the interaction as finished (panel
  // closed), which is when it is safe to rebuild the filter rail. Toggling a box
  // reports done=false so the panel survives its own change.
  let closeOpenPanel = null;   // only one panel open at a time
  const makeMultiSelect = (dim, options, values, onChange) => {
    const wrap = makeField(dim);
    const chosen = new Set(values.map(String));

    const trigger = el("button", "multi-trigger");
    trigger.type = "button";
    trigger.setAttribute("aria-haspopup", "true");
    trigger.setAttribute("aria-expanded", "false");
    const caption = el("span", "multi-caption");
    const chevron = svg(["m4 6.5 4 4 4-4"]);
    chevron.setAttribute("class", "multi-chevron");   // svg() has no class hook
    trigger.append(caption, chevron);
    const paint = () => {
      const n = chosen.size;
      caption.textContent = n === 0 ? (dim.allLabel ?? "All")
        : n === 1 ? label(dim.key, [...chosen][0])
        : `${n} selected`;
      trigger.classList.toggle("has-sel", n > 0);
    };
    paint();

    const panel = el("div", "multi-panel");
    panel.hidden = true;
    const tools = el("div", "multi-tools");
    const toolBtn = (text, fn) => {
      const b = el("button", "multi-tool", text);
      b.type = "button";
      b.addEventListener("click", fn);
      return b;
    };
    const emit = (done) => { paint(); onChange([...chosen], done); };
    tools.append(
      toolBtn("All", () => {
        for (const o of options) chosen.add(String(o.value));
        for (const cb of panel.querySelectorAll("input")) cb.checked = true;
        emit(false);
      }),
      toolBtn("Clear", () => {
        chosen.clear();
        for (const cb of panel.querySelectorAll("input")) cb.checked = false;
        emit(false);
      }));
    panel.append(tools);

    const list = el("div", "multi-list");
    for (const o of options) {
      const row = el("label", "multi-item");
      const cb = el("input");
      cb.type = "checkbox";
      cb.value = String(o.value);
      cb.checked = chosen.has(String(o.value));
      cb.addEventListener("change", () => {
        if (cb.checked) chosen.add(cb.value); else chosen.delete(cb.value);
        emit(false);
      });
      row.append(cb, el("span", null, o.text));
      list.append(row);
    }
    panel.append(list);

    // Fixed, not absolute: the rail is itself fixed with overflow:auto, so an
    // in-flow popover would be clipped by its own scroll container.
    const place = () => {
      const r = trigger.getBoundingClientRect();
      const below = window.innerHeight - r.bottom - 12;
      const above = r.top - 12;
      // Drop down by default; flip up only when below is genuinely cramped and
      // above is roomier, so a control near the foot of the rail still opens a
      // usable list instead of one clipped by the viewport.
      const flip = below < 200 && above > below;
      const maxH = Math.min(300, Math.max(140, flip ? above : below));
      panel.style.left = r.left + "px";
      panel.style.width = r.width + "px";
      panel.style.maxHeight = maxH + "px";
      panel.style.top = flip ? Math.max(8, r.top - 4 - maxH) + "px" : (r.bottom + 4) + "px";
    };
    const close = (silent) => {
      if (panel.hidden) return;
      panel.hidden = true;
      trigger.setAttribute("aria-expanded", "false");
      document.removeEventListener("pointerdown", onDocDown, true);
      document.removeEventListener("keydown", onKey, true);
      window.removeEventListener("resize", onReflow, true);
      window.removeEventListener("scroll", onReflow, true);
      closeOpenPanel = null;
      if (!silent) onChange([...chosen], true);   // now safe to rebuild the rail
    };
    const onDocDown = (e) => {
      if (!panel.contains(e.target) && !trigger.contains(e.target)) close();
    };
    const onKey = (e) => { if (e.key === "Escape") { close(); trigger.focus(); } };
    // The panel is fixed, so anything that moves the trigger has to move the panel
    // with it. Scrolling the panel's OWN list must do neither — that is the reader
    // browsing the options, and closing on it makes a long list unusable. The
    // listener is capturing (scroll does not bubble), so it sees the list's scroll
    // too and has to filter it out by target.
    const onReflow = (e) => {
      // e.target is an element for an element's scroll, but the document or the
      // window itself for a page scroll — and contains() throws on a non-Node.
      const t = e && e.target;
      if (t && t.nodeType === 1 && panel.contains(t)) return;
      const r = trigger.getBoundingClientRect();
      // Only when the control itself has been scrolled out of sight does the panel
      // have nothing left to attach to.
      if (r.bottom < 0 || r.top > window.innerHeight) close();
      else place();
    };
    const open = () => {
      closeOpenPanel?.();
      panel.hidden = false;
      trigger.setAttribute("aria-expanded", "true");
      place();
      document.addEventListener("pointerdown", onDocDown, true);
      document.addEventListener("keydown", onKey, true);
      window.addEventListener("resize", onReflow, true);
      window.addEventListener("scroll", onReflow, true);
      closeOpenPanel = () => close(true);
    };
    trigger.addEventListener("click", () => (panel.hidden ? open() : close()));

    wrap.append(trigger);
    document.body.append(panel);   // fixed panels live at the top level
    return wrap;
  };

  function fmtMetric(v, metric) {
    const unit = metric.unit;
    if (v == null) return "—";
    if (!Number.isFinite(v)) return (v < 0 ? "-inf" : "inf") + " " + unit;
    const decimals = metric.decimals ?? 3;
    return v.toFixed(decimals) + " " + unit;
  }

  // Mobile turns each row into a labeled block, so every data cell carries the
  // column name it loses when the header row is dropped.
  const cell = (td, labelText) => { td.dataset.label = labelText; return td; };

  const isStacked = () => window.matchMedia("(max-width: 700px)").matches;

  // Pin one column template across every card, then shrink the card column to the
  // content it actually holds. Two cross-view reads depend on this: the vertical
  // scan down the latency column, and reading the margin figures that sit in a
  // scale but would otherwise start at different x per card.
  // One rail governs the page. Writing the measured width onto the results block
  // alone left the filter bar and the status row overhanging their own content.
  const RAIL_MAX = 1320, RAIL_MIN = 720;
  // The rail is derived from content, so it would otherwise move with the DATA
  // rather than the viewport: choosing a workload slid the control rail and the
  // status line sideways. On a tool whose whole interaction is "change a
  // select, re-read", the chrome holds still. Monotonic within the session.
  let railSeen = 0;
  // Takes a CONTENT width. box-sizing is border-box, so the rail's own horizontal
  // padding has to be added back or the tables overflow it by exactly that much.
  const setRail = (contentPx) => {
    // PORTED: --rail is set on the widget root, not on <html>, so it cannot
    // affect the documentation page. The CSS leaves it inert (the column
    // governs width) but the measurement is kept rather than ripped out.
    const root = $("dashboard");
    if (contentPx == null) { root.style.removeProperty("--rail"); railSeen = 0; return; }
    const cs = getComputedStyle($("stage"));
    const pad = (parseFloat(cs.paddingLeft) || 0) + (parseFloat(cs.paddingRight) || 0);
    // Only the tables are measured here. The control rail is out of flow and its
    // width is already reserved by body's padding, so it must NOT be added again.
    const rail = Math.min(RAIL_MAX,
      Math.max(RAIL_MIN, railSeen, Math.ceil(contentPx) + pad));
    railSeen = rail;
    // Drives every .wrap — title, filters, status and cards — off one value.
    root.style.setProperty("--rail", rail + "px");
  };

  function alignColumns(host) {
    // The merged table opts out: it carries extra provenance columns, and this
    // template is applied by column POSITION across every table it touches.
    const tables = [...host.querySelectorAll(".topn:not(.merged)")];
    host.style.maxWidth = "";
    // Hand every table back to automatic layout: no pinned column template, no
    // max-content width. This is the state in which the long configuration names
    // wrap and the table fits whatever width it is given.
    const release = () => {
      for (const t of tables) {
        t.style.tableLayout = "";
        t.style.width = "";
        for (const c of t.querySelectorAll("col")) c.style.width = "";
      }
    };
    if (!tables.length || isStacked()) {
      setRail(null);   // the stacked breakpoint owns its own width
      release();
      return;
    }
    // Measure at max-content, not at the stretched 100% width: under `width:100%`
    // the columns are already spread to fill the card, so the measured widths sum
    // to the full container and pinning them overflows by the trailing column.
    for (const t of tables) {
      t.style.tableLayout = "auto";
      t.style.width = "max-content";
      for (const c of t.querySelectorAll("col")) c.style.width = "";
    }
    const widths = [];
    for (const t of tables) {
      const cells = t.tHead.rows[0].cells;
      for (let i = 0; i < cells.length; i++)
        widths[i] = Math.max(widths[i] || 0, Math.ceil(cells[i].getBoundingClientRect().width));
    }
    // Every column but the trailing slack cell is pinned to its widest instance.
    let total = 0;
    for (let i = 0; i < widths.length - 1; i++) total += widths[i];

    // PORTED: pinning the template also imposes it as a MINIMUM width — the widths
    // are px and the layout becomes fixed, so the table can no longer be made to
    // fit anything narrower. The original could afford that because the viewport
    // grew to meet it; a documentation column cannot, and the card then spilled
    // past the page and had to be scrolled sideways to read the latency, which is
    // the column the page exists for. So align only when the aligned table would
    // actually fit. Otherwise stay on automatic layout, where the long names
    // (`AllGather + ReduceScatter`, `MegaMoE DeepGEMM`) wrap instead of forcing
    // the width. Cross-card column alignment is the thing given up, and it is
    // worth less than having the numbers on screen.
    const slack = widths[widths.length - 1] || 0;
    const avail = host.clientWidth || 0;
    if (avail && total + slack + 2 > avail) {
      setRail(null);
      release();
      return;
    }

    for (const t of tables) {
      const cols = t.querySelectorAll("col");
      for (let i = 0; i < cols.length - 1; i++) cols[i].style.width = widths[i] + "px";
      t.style.tableLayout = "fixed";
      t.style.width = "";
    }
    // The page stops where its content stops — filters, status and cards on
    // one rail. Left over, the slack column rendered as ~40% of unexplained empty card.
    setRail(total + slack + 2);
  }

  function renderCard(facets, ranked) {
    const card = el("article", "card");
    const metric = CONFIG.metrics.find((m) => m.key === CONFIG.objective);
    const emph = CONFIG.configFields.filter((f) => f.emphasis);
    const detail = CONFIG.configFields.filter((f) => !f.emphasis);
    // Descriptive workload columns, if a deployment declares any. They sit before
    // the configuration and describe the workload rather than the settings.
    const rowFields = CONFIG.rowFields ?? [];

    // Every varying facet is part of the card's name. Promoting the first and
    // muting the rest made two adjacent cards read as the same title.
    const head = el("div", "card-head");
    head.append(el("h3", "card-title", facets.join("  ·  ")));
    card.append(head);

    if (!ranked.length) {
      card.append(el("p", "card-empty", "No successful configuration was measured here."));
      return card;
    }

    const best0 = Number(ranked[0][CONFIG.objective]);
    const table = el("table", "topn");
    // One <col> per column so alignColumns() can pin a single template across every
    // card. Cards are read as a vertical progression; per-table column widths made
    // the latency column start at a different x in each one.
    const colgroup = el("colgroup");
    // rank + workload columns + config columns + metric + "vs fastest".
    const nCols = 1 + rowFields.length + emph.length + detail.length + 2;
    for (let i = 0; i < nCols; i++) colgroup.append(el("col"));
    table.append(colgroup);
    const thead = el("thead");
    const hr = el("tr");
    const th = (txt, cls) => { const n = el("th", cls, txt); n.scope = "col"; return n; };
    hr.append(th("#", "rank"));
    rowFields.forEach((f) => hr.append(th(f.label, "detail num")));
    emph.forEach((f) => hr.append(th(f.label)));
    detail.forEach((f) => hr.append(th(f.label, "detail")));
    hr.append(th(metric.label, "num"));
    hr.append(th("vs fastest", "num"));
    thead.append(hr);
    table.append(thead);

    const tbody = el("tbody");
    ranked.forEach((row, i) => {
      const tr = el("tr", i === 0 ? "best" : null);
      tr.append(cell(el("td", "rank", String(i + 1)), "Rank"));
      rowFields.forEach((f) => tr.append(
        cell(el("td", "detail num", label(f.key, row[f.key])), f.label)));

      emph.forEach((f) => tr.append(cell(el("td", null, label(f.key, row[f.key])), f.label)));
      detail.forEach((f) => tr.append(cell(el("td", "detail", label(f.key, row[f.key])), f.label)));

      tr.append(cell(el("td", "num metric", fmtMetric(row[CONFIG.objective], metric)),
                     metric.label));

      // vs fastest: signed percentage against rank 1. Negative = slower.
      const val = Number(row[CONFIG.objective]);
      let shown = "—";
      if (i === 0) {
        shown = "fastest";
      } else if (Number.isFinite(val) && Number.isFinite(best0) && best0 > 0) {
        const pct = -100 * (val - best0) / best0;
        // A tie at display precision is "same", not "-0.0%" — a signed zero reads
        // as a real difference the reader then hunts for.
        shown = Math.abs(pct) < 0.05
          ? "same"
          : (pct > 0 ? "+" : "−") + Math.abs(pct).toFixed(1) + "%";
      }
      tr.append(cell(el("td", "num delta-num", shown), "vs fastest"));

      tbody.append(tr);
    });
    table.append(tbody);

    const wrap = el("div", "table-wrap");
    wrap.append(table);
    card.append(wrap);
    return card;
  }

  // One card holding every filtered row, ranked by absolute latency. The facets
  // that name the individual cards become columns here, since a merged row has to
  // carry its own provenance.
  //
  // The table is classed `merged` so alignColumns() skips it: that routine pins a
  // single column template across every .topn by POSITION, and this table has extra
  // leading columns — sharing the template would misalign both it and the cards.
  function renderMerged(rows, facetDims, title) {
    const card = el("article", "card");
    const metric = CONFIG.metrics.find((m) => m.key === CONFIG.objective);
    const emph = CONFIG.configFields.filter((f) => f.emphasis);
    const detail = CONFIG.configFields.filter((f) => !f.emphasis);

    const head = el("div", "card-head");
    head.append(el("h3", "card-title", title));
    card.append(head);

    if (!rows.length) {
      card.append(el("p", "card-empty", "No successful configuration was measured here."));
      return card;
    }

    const best0 = Number(rows[0][CONFIG.objective]);
    const table = el("table", "topn merged");
    const thead = el("thead");
    const hr = el("tr");
    const th = (txt, cls) => { const n = el("th", cls, txt); n.scope = "col"; return n; };
    hr.append(th("#", "rank"));
    facetDims.forEach((d) => hr.append(th(d.label, "detail" + (d.type === "number" ? " num" : ""))));
    emph.forEach((f) => hr.append(th(f.label)));
    detail.forEach((f) => hr.append(th(f.label, "detail")));
    hr.append(th(metric.label, "num"));
    hr.append(th("vs fastest", "num"));
    thead.append(hr);
    table.append(thead);

    const tbody = el("tbody");
    rows.forEach((row, i) => {
      const tr = el("tr", i === 0 ? "best" : null);
      tr.append(cell(el("td", "rank", String(i + 1)), "Rank"));
      facetDims.forEach((d) => {
        // A row with no value for a facet (attention-TP has no local batch) says so
        // rather than printing "null".
        const v = row[d.key];
        tr.append(cell(el("td", "detail" + (d.type === "number" ? " num" : ""),
                          v == null ? "—" : label(d.key, v)), d.label));
      });
      emph.forEach((f) => tr.append(cell(el("td", null, label(f.key, row[f.key])), f.label)));
      detail.forEach((f) => tr.append(cell(el("td", "detail", label(f.key, row[f.key])), f.label)));
      tr.append(cell(el("td", "num metric", fmtMetric(row[CONFIG.objective], metric)),
                     metric.label));

      const val = Number(row[CONFIG.objective]);
      let shown = "—";
      if (i === 0) {
        shown = "fastest";
      } else if (Number.isFinite(val) && Number.isFinite(best0) && best0 > 0) {
        const pct = -100 * (val - best0) / best0;
        shown = Math.abs(pct) < 0.05
          ? "same"
          : (pct > 0 ? "+" : "−") + Math.abs(pct).toFixed(1) + "%";
      }
      tr.append(cell(el("td", "num delta-num", shown), "vs fastest"));
      tbody.append(tr);
    });
    table.append(tbody);

    const wrap = el("div", "table-wrap");
    wrap.append(table);
    card.append(wrap);
    return card;
  }

  // Make the index selection self-consistent before anything reads it.
  //
  // Pruning runs LEFT TO RIGHT, each dim checked against only the dims before
  // it, never against the ones after. That asymmetry is the whole point:
  // dimensions are declared in priority order and version leads, so checking
  // version against a stale hardware left over from the previous version
  // would find it unavailable and clear VERSION — the page would abandon the
  // newest data to keep a hardware the reader never asked for. Left to right,
  // the later dims yield to the earlier ones.
  //
  // Then every required dim the pruning emptied is refilled with its first
  // remaining option. Without that the page opens on "Choose a scenario"
  // whenever the newest sweep did not happen to cover the configured default
  // model or hardware — the reader asked for the latest data, and an empty
  // page with a form to fill in is not that. The first option is whatever the
  // configured ordering puts first (GB200 before GB300), so this lands on the
  // usual default when it was measured and on whatever exists when it wasn't.
  //
  // A select whose value matches no option would otherwise render blank and
  // never fire a change event — an unreadable control the reader cannot use
  // to escape. This runs before the file is resolved, not during rendering,
  // so the rows and the controls are built from the same selection.
  function reconcileIndexSelection() {
    const validated = [];
    for (const dim of indexDims()) {
      const cur = selection[dim.key];
      if (cur == null || cur === "") continue;
      const reachable = index.some((r) =>
        String(r[dim.key]) === String(cur) &&
        validated.every((d) => String(r[d.key]) === String(selection[d.key])));
      if (reachable) validated.push(dim);
      else delete selection[dim.key];
    }
    for (const dim of indexDims()) {
      if (!dim.required || selection[dim.key]) continue;
      const first = indexOptions(dim)[0];
      if (first) selection[dim.key] = first.value;
    }
  }

  function renderFilters() {
    const host = $("filters");
    // Multi-select panels are parented to <body> to escape the rail's clipping,
    // so replaceChildren() cannot collect them — they have to be swept by hand or
    // every rebuild leaks one.
    closeOpenPanel?.();
    for (const p of document.querySelectorAll(".multi-panel")) p.remove();
    host.replaceChildren();
    for (const dim of indexDims())
      host.append(makeSelect(dim, indexOptions(dim), selection[dim.key] ?? "", (v) => {
        selection[dim.key] = v;
        onIndexChange();
        saveSelection();
      }));
    if (dataRows.length) {
      for (const dim of visibleDims()) {
        // Options normally narrow by the other data dims already chosen (e.g. the
        // workloads valid for the selected world_size). A stableOptions dim opts
        // out: it always offers every value in the file, so the control never
        // reshuffles or vanishes underneath you.
        // null is "this row has no such value" (a local batch that does not divide
        // evenly), not a value to offer — it would render as a blank option that
        // matches nothing.
        const optionsFrom = (rows) => distinct(rows, dim.key, dim.type === "number")
          .filter((v) => v != null)
          .map((v) => ({ value: v, text: label(dim.key, v) }));
        const all = optionsFrom(dataRows);
        let opts = all;
        if (!dim.stableOptions) {
          const others = visibleDims().filter((d) => d.key !== dim.key);
          opts = optionsFrom(dataRows.filter((r) =>
            others.every((d) => matchesSel(d, r[d.key]))));
          // Narrowing must never destroy the control. An empty intersection would
          // otherwise render this select blank (selectedIndex -1, no change event) or
          // delete it outright by the single-option rule below — and the dropdown you
          // need in order to escape the dead end is exactly the one that disappears.
          // A multi dim survives differently: it keeps whichever of its chosen
          // values still exist rather than collapsing the whole control.
          const has = (v) => opts.some((o) => String(o.value) === String(v));
          if (dim.multi) {
            const kept = selValues(dim).filter(has);
            if (kept.length !== selValues(dim).length) selection[dim.key] = kept;
          } else {
            const keepsCurrent = !hasSel(dim) || selValues(dim).every(has);
            // PORTED: the original widened the list back to every value when the
            // current selection was narrowed away, so the reader kept it and could
            // see it was unmeasured here. That leaves a filter in force that matches
            // no row: an empty page with every control still looking valid. It bites
            // for real — CUDA Graph off (prefill) and on (decode) measured almost
            // disjoint workload totals, so switching stage stranded the workload.
            // Follow the narrowing instead and re-seat the value: an optional dim
            // falls back to "All", a required one to its first offered value (below).
            if (!keepsCurrent && !dim.required) selection[dim.key] = "";
          }
          if (!opts.length) opts = all;
        }
        // PORTED: a required single-valued data dim must always hold one of the
        // values on offer. It can lose its selection two ways — the resolved file
        // never measured it, or a sibling dim narrowed it away (picking CUDA Graph
        // "off" drops every decode-only workload). Either way the control would
        // render with nothing marked and the page would sit on "Choose a scenario".
        // Falling back to the first offered value keeps a scenario on screen.
        if (dim.required && !dim.multi && opts.length &&
            !opts.some((o) => String(o.value) === String(selection[dim.key]))) {
          selection[dim.key] = opts[0].value;
        }
        // A dimension with a single possible value is not a choice; showing it as
        // a dropdown spends attention and offers nothing. A stableOptions dim is
        // exempt — holding its place is the point.
        if (opts.length <= 1 && !dim.required && !dim.stableOptions) {
          if (opts.length === 1) selection[dim.key] = dim.multi ? [] : "";
          continue;
        }
        if (dim.multi) {
          host.append(makeMultiSelect(dim, opts, selValues(dim), (vals, done) => {
            selection[dim.key] = vals;
            renderResults();
            saveSelection();
            // Rebuilding the rail mid-interaction would tear the open panel out of
            // the DOM, so dependent dropdowns are re-narrowed only once it closes.
            if (done) renderFilters();
          }));
          continue;
        }
        host.append(makeSelect(dim, opts, selection[dim.key] ?? "", (v) => {
          selection[dim.key] = v;
          renderFilters();   // refresh dependent dropdowns
          renderResults();
          saveSelection();
        }));
      }
    }
  }

  function renderResults() {
    const host = $("results");
    host.replaceChildren();
    host.style.maxWidth = "";
    // Deliberately no setRail(null) here: releasing on the empty path let the
    // placeholder reset the rail and slide the whole page chrome on the next render.

    const file = matchingFile();
    // Reconciles the toggle with the current selection and reports whether the
    // merged view is actually in force — it turns itself off when its precondition
    // (a chosen local batch) disappears.
    const merge = syncMerge();
    const finish = (statusText) => {
      alignColumns(host);
      setStatus(statusText);
    };

    const missing = CONFIG.dimensions.filter((d) => d.required && !selection[d.key]);
    if (missing.length || !dataRows.length) {
      const need = (missing.length ? missing : indexDims()).map((d) => d.label).join(", ");
      host.append(placeholder("Choose a scenario", `Select ${need} to see ranked configurations.`));
      finish("");
      return;
    }

    const metric = CONFIG.metrics.find((m) => m.key === CONFIG.objective);
    const dataSel = {};
    // hasSel, not truthiness: an empty multi selection is [], which is truthy.
    for (const d of visibleDims()) if (hasSel(d)) dataSel[d.key] = selValues(d);
    const filtered = filterRows(dataRows, dataSel);

    // Optional scenario axes describe each card; the ones left unselected are what
    // we fan out into separate cards — except dims flagged fanout:false, which
    // only filter the merged view.
    const optional = visibleDims().filter((d) => !d.required);
    // A facet that holds one value across every matching row is not information.
    // Splitting on it yields nothing, and naming it in each card title is noise —
    // unless the reader chose that value, where it confirms what they asked for.
    const varies = (d) => new Set(filtered.map((r) => r[d.key])).size > 1;
    // Constant *in this view* and constant *in the whole file* are different facts.
    // Dispatch holds one value everywhere, so naming it is noise. A world size that
    // happens to be the only one carrying this workload is still the reader's answer
    // to "which world size am I looking at" — and its dropdown may have collapsed to
    // a single option, leaving the card title as the only place that says so.
    const variesInFile = (d) => new Set(dataRows.map((r) => r[d.key])).size > 1;

    // Merged view: no fan-out at all, so MAX_FANOUT is deliberately not consulted.
    // That cap exists to stop a wall of cards and its advice is "narrow an axis" —
    // collapsing to one ranked table is the other cure, and must not be blocked by
    // the very condition it fixes.
    if (merge) {
      const ranked = rankGroup(filtered, CONFIG.objective, metric.goal);
      // Only the axes that actually move here earn a column; a constant one would
      // repeat the same value down the page.
      const facetDims = optional.filter(varies);
      const lb = selValues(localBatchDim()).join(", ");
      host.append(renderMerged(ranked, facetDims,
        `All scenarios at local batch ${lb}`));
      alignColumns(host);
      const scope = scopeText();
      setStatus(
        `${ranked.length} configuration${ranked.length === 1 ? "" : "s"} · ${scope} · ` +
        `ranked by ${metric.label} (${metric.goal === "min" ? "lower is better" : "higher is better"}).`);
      return;
    }

    // A multi dim always fans out over whatever it still admits: picking three
    // workloads means three cards, picking one means one. varies() already says
    // whether more than one value survived, so it decides for both kinds.
    const groupDims = optional.filter(
      (d) => (d.multi || !hasSel(d)) && d.fanout !== false && varies(d));
    const groups = groupRows(filtered, groupDims);

    // A fan-out that produces dozens of cards is not a view, it is a data dump.
    // Name the axis to narrow rather than rendering it.
    if (groups.length > MAX_FANOUT) {
      const widest = groupDims
        .map((d) => ({ d, n: new Set(filtered.map((r) => r[d.key])).size }))
        .sort((a, b) => b.n - a.n)[0];
      host.append(placeholder(
        `${groups.length} scenarios match`,
        `That is too many to compare at once. Narrow ${widest.d.label} ` +
        `(${widest.n} values) to bring this under ${MAX_FANOUT} cards.`));
      finish("");
      return;
    }

    const topN = CONFIG.display?.topN ?? 3;
    const titleDims = optional.filter(
      (d) => hasSel(d) || (d.fanout !== false && variesInFile(d)));

    // Stable-option dims can be set to a value that was never measured alongside the
    // rest of the selection. That is the trade the fixed controls buy, so it has to
    // report itself rather than render a blank page.
    if (!groups.length) {
      const chosen = visibleDims()
        .filter(hasSel)
        .map((d) => selValues(d).map((v) => facetText(d, v)).join(" / "))
        .join(", ");
      host.append(placeholder(
        "Nothing measured for this combination",
        chosen
          ? `The sweep has no successful run for ${chosen}. Change one of them.`
          : "The sweep has no successful run here."));
      finish("");
      return;
    }

    const cards = groups.map((rows) => {
      return {
        // A dim the card's rows have no value for names nothing — a workload whose
        // tokens do not divide across ranks would otherwise be titled
        // "Local batch null".
        facets: titleDims.filter((d) => rows[0][d.key] != null)
          .map((d) => facetText(d, rows[0][d.key])),
        ranked: rankGroup(rows, CONFIG.objective, metric.goal).slice(0, topN),
      };
    });

    for (const c of cards) host.append(renderCard(c.facets, c.ranked));
    const scope = scopeText();
    finish(
      `${groups.length} scenario${groups.length === 1 ? "" : "s"}` +
      ` · ${scope} · ` +
      `ranked by ${metric.label} (${metric.goal === "min" ? "lower is better" : "higher is better"}).`);
  }

  // Bumped on every index change. A load that resolves after a newer one was
  // started belongs to a selection the reader has already left, and rendering
  // it would show data the form no longer describes — so it is dropped.
  let indexChangeToken = 0;

  async function onIndexChange() {
    reconcileIndexSelection();
    const file = matchingFile();
    const token = ++indexChangeToken;
    let rows = [];
    if (file) {
      // Announce loading only if it is actually slow. A local chunk usually
      // lands in tens of milliseconds, and a placeholder that flashes on every
      // dropdown change is more disruptive than the wait it describes.
      const slow = setTimeout(() => {
        if (token !== indexChangeToken) return;
        $("results").replaceChildren(
          placeholder("Loading…", "Fetching this scenario's measurements."));
      }, 150);
      try {
        rows = await loadChunk(file);
      } catch (e) {
        clearTimeout(slow);
        if (token !== indexChangeToken) return;
        // The index offered this scenario and the data is not there. Say so
        // and keep the controls, so the reader can move somewhere that works.
        $("results").replaceChildren(placeholder(
          "Data missing",
          `${e.message} — replace data.js with a complete payload.`));
        renderFilters();
        setStatus("");
        return;
      }
      clearTimeout(slow);
      if (token !== indexChangeToken) return;
    }
    dataRows = deriveLocalBatch(rows);
    // Keep every data-dim choice that still exists in the newly resolved file.
    // Clearing them unconditionally silently reset the workload you were studying
    // on each hardware switch, and made display.defaults unusable for data dims —
    // they were wiped between being seeded and being read.
    for (const d of dataDims()) {
      const vals = selValues(d);
      if (!vals.length) continue;
      // Element-wise for a multi dim: the workloads the new file also measured are
      // kept, the rest drop out. A scalar dim keeps or loses its single value.
      const kept = vals.filter((v) => dataRows.some((r) => String(r[d.key]) === String(v)));
      if (kept.length === vals.length) continue;
      if (d.multi) selection[d.key] = kept;
      else delete selection[d.key];
    }
    renderFilters();
    renderResults();
  }

  // A measured column template goes stale the moment the viewport changes, and the
  // stacked breakpoint has to release it entirely.
  let realignTimer = 0;
  window.addEventListener("resize", () => {
    clearTimeout(realignTimer);
    realignTimer = setTimeout(() => alignColumns($("results")), 120);
  });

  // --- boot ---------------------------------------------------------------
  // No masthead in the page; the title survives only as the browser tab name.
  document.title = CONFIG.title;
  if (DATA.mock) {
    const b = $("banner");
    b.hidden = false;
    b.append(
      svg(["M8 6v3.2", "M8 11.4h.01",
           "M7 2.6 1.7 12.2c-.4.7.1 1.6.9 1.6h10.8c.8 0 1.3-.9.9-1.6L9 2.6a1.1 1.1 0 0 0-2 0Z"]),
      el("span", null, "Mock data — placeholder values, not real measurements."));
  }
  // Defaults first, then whatever the reader last looked at. Index dims that no
  // longer resolve to a file, and data dims absent from it, are dropped by
  // onIndexChange rather than trusted.
  Object.assign(selection, CONFIG.display?.defaults ?? {});
  const restored = savedSelection();
  if (restored && typeof restored === "object") {
    const known = new Set(CONFIG.dimensions.map((d) => d.key));
    for (const [k, v] of Object.entries(restored)) if (known.has(k)) selection[k] = v;
  }
  if (!matchingFile()) {
    // A stale index selection resolves to nothing; fall back to the configured
    // defaults.
    for (const d of indexDims()) delete selection[d.key];
    Object.assign(selection, CONFIG.display?.defaults ?? {});
  }
  onIndexChange();
}

// Only pages carrying the directive have the placeholder; on any other page this
// is a no-op, and a missing data payload is reported in place rather than thrown.
function moeDashboardBoot() {
  if (!document.getElementById("moe-dashboard")) return;
  moeDashboardMain();
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", moeDashboardBoot);
} else {
  moeDashboardBoot();
}
