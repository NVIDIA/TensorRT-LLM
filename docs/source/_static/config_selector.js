(function () {
  "use strict";

  let dbPromise = null;
  let widgetId = 0;

  function $(root, sel) {
    return root.querySelector(sel);
  }

  function el(tag, attrs = {}, children = []) {
    const node = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (k === "class") node.className = String(v);
      else if (k === "text") node.textContent = String(v);
      else if (k.startsWith("data-")) node.setAttribute(k, String(v));
      else if (k === "for") node.htmlFor = String(v);
      else node.setAttribute(k, String(v));
    }
    for (const c of children) node.appendChild(c);
    return node;
  }

  function uniqBy(arr, keyFn) {
    const seen = new Set();
    const out = [];
    for (const x of arr) {
      const k = keyFn(x);
      if (!seen.has(k)) {
        seen.add(k);
        out.push(x);
      }
    }
    return out;
  }

  function sortStrings(a, b) {
    return String(a).localeCompare(String(b));
  }

  function sortNums(a, b) {
    return Number(a) - Number(b);
  }

  async function loadDb(dbUrl) {
    if (!dbPromise) {
      dbPromise = fetch(dbUrl, { credentials: "same-origin" }).then((r) => {
        if (!r.ok) {
          throw new Error(`Failed to load config DB (${r.status}): ${dbUrl}`);
        }
        return r.json();
      });
    }
    return dbPromise;
  }

  function defaultDbUrl() {
    const scriptEl = document.querySelector('script[src*="config_selector.js"]');
    if (scriptEl && scriptEl.src) {
      const u = new URL(scriptEl.src, document.baseURI);
      u.pathname = u.pathname.replace(/config_selector\.js$/, "config_db.json");
      u.search = "";
      u.hash = "";
      return u.toString();
    }
    return new URL("_static/config_db.json", document.baseURI).toString();
  }

  async function copyText(text) {
    if (navigator.clipboard && navigator.clipboard.writeText) {
      await navigator.clipboard.writeText(text);
      return;
    }
    const ta = el("textarea", { "aria-hidden": "true" });
    ta.value = text;
    ta.style.position = "fixed";
    ta.style.left = "-9999px";
    document.body.appendChild(ta);
    ta.select();
    document.execCommand("copy");
    document.body.removeChild(ta);
  }

  function escapeHtml(s) {
    return String(s)
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#039;");
  }

  function highlightYaml(yamlText) {
    const lines = String(yamlText).split("\n");
    const out = [];

    function highlightScalar(raw) {
      const m = String(raw).match(/^(\s*)(.*?)(\s*)$/);
      const lead = m ? m[1] : "";
      const core = m ? m[2] : String(raw);
      const trail = m ? m[3] : "";
      const t = core.trim();
      if (!t) return escapeHtml(raw);

      const boolNull = /^(true|false|null|~)$/;
      const num = /^-?\d+(\.\d+)?$/;
      const dq = t.length >= 2 && t.startsWith('"') && t.endsWith('"');
      const sq = t.length >= 2 && t.startsWith("'") && t.endsWith("'");

      if (boolNull.test(t)) {
        return `${escapeHtml(lead)}<span class="yaml-bool">${escapeHtml(core)}</span>${escapeHtml(trail)}`;
      }
      if (num.test(t)) {
        return `${escapeHtml(lead)}<span class="yaml-num">${escapeHtml(core)}</span>${escapeHtml(trail)}`;
      }
      if (dq || sq) {
        return `${escapeHtml(lead)}<span class="yaml-str">${escapeHtml(core)}</span>${escapeHtml(trail)}`;
      }
      return escapeHtml(raw);
    }

    for (const line of lines) {
      const hashIdx = line.indexOf("#");
      const hasComment = hashIdx >= 0;
      const codePart = hasComment ? line.slice(0, hashIdx) : line;
      const commentPart = hasComment ? line.slice(hashIdx) : "";

      const mList = codePart.match(/^(\s*)(-\s+)?(.*)$/);
      const indent = mList ? mList[1] : "";
      const dash = mList && mList[2] ? mList[2] : "";
      const rest = mList ? mList[3] : codePart;

      const idx = rest.indexOf(":");
      let html = "";
      if (idx >= 0) {
        const keyRaw = rest.slice(0, idx);
        const after = rest.slice(idx + 1);
        html += escapeHtml(indent);
        if (dash) html += `<span class="yaml-punct">-</span>${escapeHtml(dash.slice(1))}`;
        html += `<span class="yaml-key">${escapeHtml(keyRaw.trimEnd())}</span>`;
        html += `<span class="yaml-punct">:</span>`;
        html += highlightScalar(after);
      } else {
        html += escapeHtml(indent);
        if (dash) html += `<span class="yaml-punct">-</span>${escapeHtml(dash.slice(1))}`;
        html += highlightScalar(rest);
      }

      if (commentPart) {
        html += `<span class="yaml-comment">${escapeHtml(commentPart)}</span>`;
      }
      out.push(html);
    }
    return out.join("\n");
  }

  function formatCommand(entry) {
    const model = entry.model || "";
    const configPath = entry.config_path || "";
    if (!model || !configPath) return entry.command || "";
    return [
      `trtllm-serve ${model} \\`,
      `  --config \${TRTLLM_DIR}/${configPath}`,
    ].join("\n");
  }

  function parseCsvModels(s) {
    if (!s) return null;
    const parts = String(s)
      .split(",")
      .map((x) => x.trim())
      .filter(Boolean);
    return parts.length ? parts : null;
  }

  function initOne(container, payload) {
    const allowedModels = parseCsvModels(container.getAttribute("data-models"));

    const allEntries = Array.isArray(payload.entries) ? payload.entries : [];
    const entries = allowedModels
      ? allEntries.filter((e) => allowedModels.includes(e.model))
      : allEntries.slice();

    const modelsInfo = payload.models || {};

    const state = {
      model: "",
      topology: "",
      islOsl: "",
      profile: "",
      concurrency: "",
    };

    container.innerHTML = "";
    container.classList.add("trtllm-config-selector");

    const header = el("div", { class: "trtllm-config-selector__header" }, [
      el("div", {
        class: "trtllm-config-selector__subtitle",
        text: "Select a model + deployment shape to generate a trtllm-serve command.",
      }),
    ]);

    const form = el("div", { class: "trtllm-config-selector__form" });

    function mkSelect(labelText, id, stepNumber) {
      const label = el("label", {
        class: "trtllm-config-selector__label",
        for: id,
      });
      label.appendChild(
        el("span", {
          class: "trtllm-config-selector__step",
          "aria-hidden": "true",
          text: String(stepNumber),
        }),
      );
      label.appendChild(
        el("span", {
          class: "trtllm-config-selector__labelText",
          text: labelText,
        }),
      );
      const select = el("select", { class: "trtllm-config-selector__select", id });
      const wrap = el("div", { class: "trtllm-config-selector__field" }, [label, select]);
      return { wrap, select };
    }

    const id = ++widgetId;
    const selModel = mkSelect("Model", `trtllm-model-${id}`, 1);
    const selTopo = mkSelect("GPU(s)", `trtllm-topo-${id}`, 2);
    const selSeq = mkSelect("ISL / OSL", `trtllm-seq-${id}`, 3);
    const selProf = mkSelect("Performance profile", `trtllm-prof-${id}`, 4);
    const selConc = mkSelect("Concurrency", `trtllm-conc-${id}`, 5);

    form.appendChild(selModel.wrap);
    form.appendChild(selTopo.wrap);
    form.appendChild(selSeq.wrap);
    form.appendChild(selProf.wrap);
    form.appendChild(selConc.wrap);

    const output = el("div", { class: "trtllm-config-selector__output" });
    const cmdPre = el("pre", { class: "trtllm-config-selector__cmd" }, [
      el("code", { class: "trtllm-config-selector__cmdcode", text: "" }),
    ]);
    const cmdCopyBtn = el("button", {
      class: "trtllm-config-selector__copyInline",
      type: "button",
      title: "Copy command",
      "aria-label": "Copy command",
      text: "Copy",
    });
    const meta = el("div", { class: "trtllm-config-selector__meta", text: "" });

    output.appendChild(cmdPre);
    output.appendChild(meta);
    cmdPre.appendChild(cmdCopyBtn);

    const yamlDetails = el("details", { class: "trtllm-config-selector__yamlDetails" }, [
      el("summary", { class: "trtllm-config-selector__yamlSummary", text: "Show config YAML" }),
    ]);
    const yamlBox = el("div", { class: "trtllm-config-selector__yamlBox" });
    const yamlPre = el("pre", { class: "trtllm-config-selector__yamlPre" }, [
      el("code", { class: "trtllm-config-selector__yamlCode", text: "" }),
    ]);
    const yamlCopyBtn = el("button", {
      class: "trtllm-config-selector__copyInline",
      type: "button",
      title: "Copy YAML",
      "aria-label": "Copy YAML",
      text: "Copy",
    });
    yamlBox.appendChild(yamlPre);
    yamlDetails.appendChild(yamlBox);
    output.appendChild(yamlDetails);
    yamlPre.appendChild(yamlCopyBtn);

    const errorBox = el("div", { class: "trtllm-config-selector__error", text: "" });

    container.appendChild(header);
    container.appendChild(form);
    container.appendChild(output);
    container.appendChild(errorBox);

    const yamlCache = new Map();
    let currentEntry = null;
    let currentYamlText = "";
    const yamlCodeEl = $(yamlPre, "code");

    async function fetchYamlFor(entry) {
      const url = entry.config_raw_url || "";
      if (!url) return null;
      if (yamlCache.has(url)) return yamlCache.get(url) || "";
      const r = await fetch(url, { credentials: "omit" });
      if (!r.ok) throw new Error(`Failed to fetch YAML (${r.status}): ${url}`);
      const txt = await r.text();
      yamlCache.set(url, txt);
      return txt;
    }

    function resetYamlPanel() {
      yamlDetails.open = false;
      yamlDetails.dataset.state = "idle";
      yamlCodeEl.textContent = "";
      yamlCopyBtn.disabled = true;
      currentYamlText = "";
    }

    resetYamlPanel();

    yamlDetails.addEventListener("toggle", async () => {
      if (!yamlDetails.open) return;
      if (!currentEntry) {
        yamlDetails.dataset.state = "idle";
        yamlCodeEl.textContent = "Select a configuration above to view its YAML.";
        return;
      }
      if (yamlDetails.dataset.state === "loaded") return;
      if (yamlDetails.dataset.state === "loading") return;

      const e = currentEntry;
      if (!e.config_raw_url) {
        yamlDetails.dataset.state = "error";
        yamlCodeEl.textContent = "No raw URL available for this config.";
        return;
      }

      yamlDetails.dataset.state = "loading";
      yamlCodeEl.textContent = `Loading YAML from ${e.config_raw_url} …`;
      try {
        const txt = await fetchYamlFor(e);
        currentYamlText = txt || "";
        yamlDetails.dataset.state = "loaded";
        yamlCodeEl.innerHTML = highlightYaml(currentYamlText);
        yamlCopyBtn.disabled = !currentYamlText;
      } catch (err) {
        yamlDetails.dataset.state = "error";
        yamlCopyBtn.disabled = true;
        yamlCodeEl.textContent = `Failed to load YAML.\n\n${String(err)}`;
      }
    });

    yamlCopyBtn.addEventListener("click", async () => {
      const txt = currentYamlText || yamlCodeEl.textContent || "";
      if (!txt) return;
      try {
        await copyText(txt);
        yamlCopyBtn.textContent = "Copied";
        setTimeout(() => (yamlCopyBtn.textContent = "Copy"), 1200);
      } catch (_) {
        yamlCopyBtn.textContent = "Copy failed";
        setTimeout(() => (yamlCopyBtn.textContent = "Copy"), 1500);
      }
    });

    function setSelectOptions(select, options, value, placeholder) {
      select.innerHTML = "";
      select.appendChild(el("option", { value: "", text: placeholder || "Select…" }));
      for (const opt of options) {
        select.appendChild(el("option", { value: opt.value, text: opt.label }));
      }
      select.value = value || "";
      select.disabled = options.length === 0;
    }

    function filteredByState(prefixOnly = false) {
      return entries.filter((e) => {
        if (state.model && e.model !== state.model) return false;
        if (state.topology) {
          const [ng, gpu] = state.topology.split("|");
          if (String(e.num_gpus) !== ng || e.gpu !== gpu) return false;
        }
        if (state.islOsl) {
          const [isl, osl] = state.islOsl.split("|");
          if (String(e.isl) !== isl || String(e.osl) !== osl) return false;
        }
        if (!prefixOnly && state.profile && e.performance_profile !== state.profile) return false;
        if (!prefixOnly && state.concurrency && String(e.concurrency) !== state.concurrency) return false;
        return true;
      });
    }

    function render() {
      errorBox.textContent = "";

      // Model options
      const modelOpts = uniqBy(
        entries.map((e) => e.model),
        (m) => m
      )
        .sort(sortStrings)
        .map((m) => {
          const info = modelsInfo[m];
          const label = info && info.display_name ? `${info.display_name} (${m})` : m;
          return { value: m, label };
        });
      if (state.model && !modelOpts.some((o) => o.value === state.model)) state.model = "";
      if (!state.model && modelOpts.length === 1) state.model = modelOpts[0].value;
      setSelectOptions(selModel.select, modelOpts, state.model, "Select a model…");

      // GPU(s) options
      const topoEntries = entries.filter((e) => !state.model || e.model === state.model);
      const topoOpts = uniqBy(
        topoEntries.map((e) => ({
          value: `${e.num_gpus}|${e.gpu}`,
          label: e.gpu_display || `${e.num_gpus}x${e.gpu}`,
          num_gpus: e.num_gpus,
          gpu: e.gpu,
        })),
        (o) => o.value
      )
        .sort((a, b) => sortNums(a.num_gpus, b.num_gpus) || sortStrings(a.gpu, b.gpu));
      if (state.topology && !topoOpts.some((o) => o.value === state.topology)) state.topology = "";
      if (!state.topology && topoOpts.length === 1) state.topology = topoOpts[0].value;
      setSelectOptions(selTopo.select, topoOpts, state.topology, "Select GPU(s)…");

      // ISL/OSL options
      const seqEntries = entries.filter((e) => {
        if (state.model && e.model !== state.model) return false;
        if (state.topology) {
          const [ng, gpu] = state.topology.split("|");
          if (String(e.num_gpus) !== ng || e.gpu !== gpu) return false;
        }
        return true;
      });
      const seqOpts = uniqBy(
        seqEntries.map((e) => ({
          value: `${e.isl}|${e.osl}`,
          label: `${e.isl} / ${e.osl}`,
          isl: e.isl,
          osl: e.osl,
        })),
        (o) => o.value
      ).sort((a, b) => sortNums(a.isl, b.isl) || sortNums(a.osl, b.osl));
      if (state.islOsl && !seqOpts.some((o) => o.value === state.islOsl)) state.islOsl = "";
      if (!state.islOsl && seqOpts.length === 1) state.islOsl = seqOpts[0].value;
      setSelectOptions(selSeq.select, seqOpts, state.islOsl, "Select ISL/OSL…");

      // Profile options
      const prefEntries = filteredByState(true);
      const profOpts = uniqBy(
        prefEntries.map((e) => e.performance_profile),
        (p) => p
      )
        .sort(sortStrings)
        .map((p) => ({ value: p, label: p }));
      if (state.profile && !profOpts.some((o) => o.value === state.profile)) state.profile = "";
      if (!state.profile && profOpts.length === 1) state.profile = profOpts[0].value;
      // Prefer Balanced if present (nicer default).
      if (!state.profile && profOpts.some((o) => o.value === "Balanced")) state.profile = "Balanced";
      setSelectOptions(selProf.select, profOpts, state.profile, "Select a profile…");

      // Concurrency options (filtered by profile if chosen)
      const profEntries2 = filteredByState(true).filter((e) => !state.profile || e.performance_profile === state.profile);
      const concOpts = uniqBy(
        profEntries2.map((e) => ({ value: String(e.concurrency), label: String(e.concurrency), conc: e.concurrency })),
        (o) => o.value
      ).sort((a, b) => sortNums(a.conc, b.conc));
      if (state.concurrency && !concOpts.some((o) => o.value === state.concurrency)) state.concurrency = "";
      if (!state.concurrency && concOpts.length === 1) state.concurrency = concOpts[0].value;
      setSelectOptions(selConc.select, concOpts, state.concurrency, "Select concurrency…");

      // Resolve final selection
      const finalEntries = filteredByState(false).filter((e) => {
        if (state.profile && e.performance_profile !== state.profile) return false;
        if (state.concurrency && String(e.concurrency) !== state.concurrency) return false;
        return true;
      });

      const code = cmdPre.querySelector("code");
      if (finalEntries.length === 1) {
        const e = finalEntries[0];
        code.textContent = formatCommand(e);
        cmdCopyBtn.disabled = !code.textContent;
        meta.textContent = "";
        meta.appendChild(el("span", { text: "Config: " }));
        const cfgHref = e.config_github_url || e.config_raw_url || "";
        if (cfgHref) {
          meta.appendChild(
            el("a", {
              class: "trtllm-config-selector__configLink",
              href: cfgHref,
              target: "_blank",
              rel: "noopener",
              text: e.config_path || cfgHref,
            })
          );
        } else {
          meta.appendChild(el("span", { text: e.config_path || "" }));
        }

        currentEntry = e;
        resetYamlPanel();
      } else {
        code.textContent = "";
        cmdCopyBtn.disabled = true;
        meta.textContent = "";
        currentEntry = null;
        resetYamlPanel();
        if (entries.length === 0) {
          errorBox.textContent = "No configuration entries available for this page.";
        } else if (state.model && topoOpts.length === 0) {
          errorBox.textContent = "No matching topologies for this model.";
        } else if (state.topology && seqOpts.length === 0) {
          errorBox.textContent = "No matching ISL/OSL options for this selection.";
        } else if (state.islOsl && profOpts.length === 0) {
          errorBox.textContent = "No matching performance profiles for this selection.";
        } else if (state.profile && concOpts.length === 0) {
          errorBox.textContent = "No matching concurrencies for this profile.";
        } else if (state.model && state.topology && state.islOsl && state.profile && state.concurrency) {
          errorBox.textContent = "Selection did not resolve to a single configuration.";
        } else {
          errorBox.textContent = "Select options above to generate a command.";
        }
      }
    }

    selModel.select.addEventListener("change", () => {
      state.model = selModel.select.value;
      state.topology = "";
      state.islOsl = "";
      state.profile = "";
      state.concurrency = "";
      render();
    });
    selTopo.select.addEventListener("change", () => {
      state.topology = selTopo.select.value;
      state.islOsl = "";
      state.profile = "";
      state.concurrency = "";
      render();
    });
    selSeq.select.addEventListener("change", () => {
      state.islOsl = selSeq.select.value;
      state.profile = "";
      state.concurrency = "";
      render();
    });
    selProf.select.addEventListener("change", () => {
      state.profile = selProf.select.value;
      state.concurrency = "";
      render();
    });
    selConc.select.addEventListener("change", () => {
      state.concurrency = selConc.select.value;
      render();
    });

    cmdCopyBtn.addEventListener("click", async () => {
      const code = $(cmdPre, "code");
      const txt = (code && code.textContent) || "";
      if (!txt) return;
      try {
        await copyText(txt);
        cmdCopyBtn.textContent = "Copied";
        setTimeout(() => (cmdCopyBtn.textContent = "Copy"), 1200);
      } catch (e) {
        cmdCopyBtn.textContent = "Copy failed";
        setTimeout(() => (cmdCopyBtn.textContent = "Copy"), 1500);
      }
    });

    render();
  }

  async function main() {
    const containers = Array.from(document.querySelectorAll("[data-trtllm-config-selector]"));
    if (!containers.length) return;

    const first = containers[0];
    const dbPath = first.getAttribute("data-config-db");
    const dbUrl = dbPath
      ? new URL(dbPath, document.baseURI).toString()
      : defaultDbUrl();

    try {
      const payload = await loadDb(dbUrl);
      for (const c of containers) initOne(c, payload);
    } catch (err) {
      for (const c of containers) {
        c.textContent = `Failed to load configuration database: ${String(err)}`;
      }
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", main);
  } else {
    main();
  }
})();
