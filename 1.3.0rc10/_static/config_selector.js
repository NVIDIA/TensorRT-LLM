(function () {
  "use strict";

  let dbPromise = null;
  let widgetId = 0;
  const GROUP_ORDER = ["model", "topology", "islOsl", "concurrency"];
  const GROUP_LABELS = {
    model: "Model",
    topology: "GPU(s)",
    islOsl: "ISL / OSL",
    concurrency: "Concurrency",
  };

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

  function normalizeState(state = {}) {
    return {
      model: state.model || "",
      topology: state.topology || "",
      islOsl: state.islOsl || "",
      concurrency:
        state.concurrency != null && state.concurrency !== ""
          ? String(state.concurrency)
          : "",
    };
  }

  function withStateValue(state, key, value) {
    return normalizeState({ ...normalizeState(state), [key]: value });
  }

  function clearStateFromKey(state, key) {
    const normalizedState = normalizeState(state);
    const nextState = { ...normalizedState };
    const startIndex = GROUP_ORDER.indexOf(key);
    if (startIndex < 0) return nextState;
    for (let i = startIndex; i < GROUP_ORDER.length; i += 1) {
      nextState[GROUP_ORDER[i]] = "";
    }
    return normalizeState(nextState);
  }

  function nextStateAfterSelection(state, key, option) {
    const normalizedState = normalizeState(state);
    if (!option || option.status === "incompatible") {
      return normalizedState;
    }
    if (option.selected) {
      return clearStateFromKey(normalizedState, key);
    }

    const nextState = { ...normalizedState, [key]: option.value };
    const keyIndex = GROUP_ORDER.indexOf(key);
    if (keyIndex >= 0 && normalizedState[key] && normalizedState[key] !== option.value) {
      for (let i = keyIndex + 1; i < GROUP_ORDER.length; i += 1) {
        nextState[GROUP_ORDER[i]] = "";
      }
    }
    return normalizeState(nextState);
  }

  function limitList(items, maxItems = 4) {
    const uniq = uniqBy(
      items.filter(Boolean).map((item) => String(item)),
      (item) => item
    );
    if (uniq.length <= maxItems) return uniq;
    return uniq.slice(0, maxItems).concat(`and ${uniq.length - maxItems} more`);
  }

  function joinLabels(items) {
    return limitList(items).join(", ");
  }

  function modelOption(model, modelsInfo) {
    const info = modelsInfo[model];
    const displayName = info && info.display_name ? info.display_name : model;
    return {
      value: model,
      label: displayName,
      hint: displayName !== model ? model : "",
    };
  }

  function topologyOption(entry) {
    return {
      value: `${entry.num_gpus}|${entry.gpu}`,
      label: entry.gpu_display || `${entry.num_gpus}x${entry.gpu}`,
      num_gpus: entry.num_gpus,
      gpu: entry.gpu,
    };
  }

  function sequenceOption(entry) {
    return {
      value: `${entry.isl}|${entry.osl}`,
      label: `${entry.isl} / ${entry.osl}`,
      isl: entry.isl,
      osl: entry.osl,
    };
  }

  function profileLabel(profile) {
    const text = String(profile || "").trim();
    return text || "Unknown Profile";
  }

  function formatProfileSummary(profiles) {
    const labels = uniqBy(profiles.map((profile) => profileLabel(profile)), (label) => label);
    if (!labels.length) {
      return "Unknown Profile";
    }
    if (labels.length <= 2) {
      return labels.join(" / ");
    }
    return `${labels[0]} +${labels.length - 1} more`;
  }

  function concurrencyOption(entriesForConcurrency) {
    const first = entriesForConcurrency[0];
    const profiles = uniqBy(
      entriesForConcurrency.map((entry) => entry.performance_profile).filter(Boolean),
      (profile) => profile
    );
    const profileSummary = formatProfileSummary(profiles);
    return {
      value: String(first.concurrency),
      label: `${first.concurrency} · ${profileSummary}`,
      concurrency: first.concurrency,
      profiles,
    };
  }

  function buildDomains(entries, modelsInfo) {
    const model = uniqBy(entries.map((entry) => entry.model), (value) => value)
      .sort(sortStrings)
      .map((value) => modelOption(value, modelsInfo));

    const topology = uniqBy(
      entries.map((entry) => topologyOption(entry)),
      (option) => option.value
    ).sort(
      (a, b) => sortNums(a.num_gpus, b.num_gpus) || sortStrings(a.gpu, b.gpu)
    );

    const islOsl = uniqBy(
      entries.map((entry) => sequenceOption(entry)),
      (option) => option.value
    ).sort((a, b) => sortNums(a.isl, b.isl) || sortNums(a.osl, b.osl));

    const concurrencyGroups = new Map();
    for (const entry of entries) {
      const key = String(entry.concurrency);
      if (!concurrencyGroups.has(key)) concurrencyGroups.set(key, []);
      concurrencyGroups.get(key).push(entry);
    }
    const concurrency = Array.from(concurrencyGroups.values())
      .map((group) => concurrencyOption(group))
      .sort((a, b) => sortNums(a.concurrency, b.concurrency));

    return { model, topology, islOsl, concurrency };
  }

  function filterEntriesByState(entries, state) {
    const normalizedState = normalizeState(state);
    return entries.filter((entry) => {
      if (normalizedState.model && entry.model !== normalizedState.model) {
        return false;
      }
      if (normalizedState.topology) {
        const [numGpus, gpu] = normalizedState.topology.split("|");
        if (String(entry.num_gpus) !== numGpus || entry.gpu !== gpu) return false;
      }
      if (normalizedState.islOsl) {
        const [isl, osl] = normalizedState.islOsl.split("|");
        if (String(entry.isl) !== isl || String(entry.osl) !== osl) return false;
      }
      if (normalizedState.concurrency) {
        if (String(entry.concurrency) !== normalizedState.concurrency) return false;
      }
      return true;
    });
  }

  function prefixStateForKey(state, key) {
    const normalizedState = normalizeState(state);
    const prefixState = {};
    for (const groupKey of GROUP_ORDER) {
      if (groupKey === key) break;
      prefixState[groupKey] = normalizedState[groupKey];
    }
    return normalizeState(prefixState);
  }

  function optionStateFor(entries, state, key, value) {
    const normalizedState = normalizeState(state);
    const prefixState = prefixStateForKey(normalizedState, key);
    const matches = filterEntriesByState(entries, withStateValue(prefixState, key, value));
    const isSelected = normalizedState[key] === value;
    if (isSelected && matches.length === 0) {
      return "active-incompatible";
    }
    if (isSelected) return "active";
    if (matches.length > 0) return "available";
    return "incompatible";
  }

  function buildCompatibilityGroups(entries, modelsInfo, state) {
    const normalizedState = normalizeState(state);
    const domains = buildDomains(entries, modelsInfo);
    const groups = {};
    for (const key of GROUP_ORDER) {
      const prefixEntries = filterEntriesByState(entries, prefixStateForKey(normalizedState, key));
      const prefixDomains = buildDomains(prefixEntries, modelsInfo);
      groups[key] = {
        key,
        label: GROUP_LABELS[key],
        options: domains[key].map((option) => {
          const optionState = optionStateFor(entries, normalizedState, key, option.value);
          const contextualOption =
            key === "concurrency"
              ? prefixDomains.concurrency.find((candidate) => candidate.value === option.value) ||
                option
              : option;
          return {
            ...contextualOption,
            status: optionState,
            selected: normalizedState[key] === option.value,
          };
        }),
      };
    }
    return groups;
  }

  function availableOptions(group) {
    return group.options.filter((option) => option.status === "available");
  }

  function statesEqual(left, right) {
    const leftState = normalizeState(left);
    const rightState = normalizeState(right);
    return GROUP_ORDER.every((key) => leftState[key] === rightState[key]);
  }

  function autoSelectSingletonState(state, groups) {
    const nextState = normalizeState(state);

    if (!nextState.model) {
      const modelOptions = availableOptions(groups.model);
      if (modelOptions.length === 1) {
        nextState.model = modelOptions[0].value;
      }
    }

    if (
      nextState.model &&
      nextState.topology &&
      nextState.islOsl &&
      !nextState.concurrency
    ) {
      const concurrencyOptions = availableOptions(groups.concurrency);
      if (concurrencyOptions.length === 1) {
        nextState.concurrency = concurrencyOptions[0].value;
      }
    }

    return normalizeState(nextState);
  }

  function buildSelectorMessage(entries, groups, state, finalEntries) {
    if (!entries.length) return "No configuration entries available for this page.";

    const normalizedState = normalizeState(state);
    if (finalEntries.length === 1) return "";

    const invalidSelectionKey = [...GROUP_ORDER]
      .reverse()
      .find((key) =>
        groups[key].options.some(
          (option) => option.selected && option.status === "active-incompatible"
        )
      );
    if (invalidSelectionKey) {
      const invalidOption = groups[invalidSelectionKey].options.find(
        (option) => option.selected && option.status === "active-incompatible"
      );
      const availableOptions = groups[invalidSelectionKey].options
        .filter((option) => option.status === "available")
        .map((option) => option.label);
      if (availableOptions.length) {
        return `${GROUP_LABELS[invalidSelectionKey]} ${invalidOption.label} is not available for the current selection. ` +
          `Available ${GROUP_LABELS[invalidSelectionKey]} options: ${joinLabels(
            availableOptions
          )}.`;
      }
      return `${GROUP_LABELS[invalidSelectionKey]} ${invalidOption.label} is not available for the current selection.`;
    }

    const nextGroupKey = GROUP_ORDER.find((key) => !normalizedState[key]);
    if (nextGroupKey) {
      const availableOptions = groups[nextGroupKey].options
        .filter((option) => option.status === "available")
        .map((option) => option.label);
      const incompatibleCount = groups[nextGroupKey].options.filter(
        (option) => option.status === "incompatible"
      ).length;
      if (
        availableOptions.length &&
        (normalizedState.model ||
          normalizedState.topology ||
          normalizedState.islOsl)
      ) {
        let message = `Available ${GROUP_LABELS[nextGroupKey]} options for the current selection: ${joinLabels(
          availableOptions
        )}.`;
        if (incompatibleCount > 0) {
          message += " Greyed-out options are unavailable for the current selection.";
        }
        return message;
      }
      return "Select options above to generate a command.";
    }

    if (
      normalizedState.model &&
      normalizedState.topology &&
      normalizedState.islOsl &&
      normalizedState.concurrency
    ) {
      return "Selection did not resolve to a single configuration.";
    }
    return "Select options above to generate a command.";
  }

  function createSelectorViewModel(entries, modelsInfo, state) {
    const normalizedState = normalizeState(state);
    let groups = buildCompatibilityGroups(entries, modelsInfo, normalizedState);
    const effectiveState = autoSelectSingletonState(normalizedState, groups);
    if (!statesEqual(normalizedState, effectiveState)) {
      groups = buildCompatibilityGroups(entries, modelsInfo, effectiveState);
    }
    const finalEntries = filterEntriesByState(entries, effectiveState);
    const resolvedEntry = finalEntries.length === 1 ? finalEntries[0] : null;
    return {
      state: effectiveState,
      groups,
      finalEntries,
      resolvedEntry,
      commandText: resolvedEntry ? formatCommand(resolvedEntry) : "",
      message: buildSelectorMessage(entries, groups, effectiveState, finalEntries),
      selectedConcurrencyBadge: "",
    };
  }

  function isFileProtocol() {
    return window.location.protocol === "file:";
  }

  function xhrLoadJson(url) {
    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open("GET", url, true);
      xhr.onload = function () {
        if (xhr.status === 0 || (xhr.status >= 200 && xhr.status < 300)) {
          try {
            resolve(JSON.parse(xhr.responseText));
          } catch (e) {
            reject(new Error("Failed to parse config DB JSON: " + String(e)));
          }
        } else {
          reject(new Error("Failed to load config DB (" + xhr.status + "): " + url));
        }
      };
      xhr.onerror = function () {
        reject(
          new Error(
            "Failed to load configuration database. If viewing locally, " +
              "serve the docs with an HTTP server:\n" +
              "  python -m http.server -d docs/build/html\n" +
              "Then open http://localhost:8000/deployment-guide/"
          )
        );
      };
      xhr.send();
    });
  }

  async function loadDb(dbUrl) {
    if (!dbPromise) {
      if (isFileProtocol()) {
        dbPromise = xhrLoadJson(dbUrl);
      } else {
        dbPromise = fetch(dbUrl, { credentials: "same-origin" }).then((r) => {
          if (!r.ok) {
            throw new Error("Failed to load config DB (" + r.status + "): " + dbUrl);
          }
          return r.json();
        });
      }
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

  function highlightBash(text) {
    return String(text).split("\n").map(function (line) {
      var html = escapeHtml(line);
      // export keyword
      html = html.replace(/^(export)\b/, '<span class="bash-kw">$1</span>');
      // variable expansion ${...}
      html = html.replace(/\$\{([^}]+)\}/g, '<span class="bash-var">${$1}</span>');
      // flags --word
      html = html.replace(/(--\w[\w-]*)/g, '<span class="bash-flag">$1</span>');
      // command name (trtllm-serve at start of line or after newline)
      html = html.replace(/^(trtllm-serve)/, '<span class="bash-cmd">$1</span>');
      // line continuation backslash
      html = html.replace(/\\$/, '<span class="bash-cont">\\</span>');
      // comment
      html = html.replace(/(#.*)$/, '<span class="bash-comment">$1</span>');
      return html;
    }).join("\n");
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
    const model = entry._hfModel || entry.model || "";
    const configPath = entry.config_path || "";
    if (!model || !configPath) return entry.command || "";
    return [
      `export TRTLLM_DIR=/app/tensorrt_llm`,
      `trtllm-serve ${model} \\`,
      `  --config \${TRTLLM_DIR}/${configPath}`,
    ].join("\n");
  }

  function curatedEntriesForModel(curatedEntries, model) {
    if (!model || !curatedEntries || !curatedEntries.length) return [];
    return curatedEntries.filter((entry) => entry.model === model);
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
    const allCurated = Array.isArray(payload.curated_entries) ? payload.curated_entries : [];
    const modelsInfo = payload.models || {};

    // Filter by allowed models (using original HF IDs), then normalize
    // model field to display_name so the view-model groups by friendly name.
    // _hfModel preserves the original HF ID for command generation.
    function normalizeEntry(e) {
      const info = modelsInfo[e.model];
      const displayName = (info && info.display_name) || e.model;
      return { ...e, _hfModel: e.model, model: displayName };
    }

    const entries = (allowedModels
      ? allEntries.filter((e) => allowedModels.includes(e.model))
      : allEntries
    ).map(normalizeEntry);

    const curatedEntries = (allowedModels
      ? allCurated.filter((e) => allowedModels.includes(e.model))
      : allCurated
    ).map(normalizeEntry);

    // curatedIndex lives outside normalizeState's scope — it is preserved
    // across Object.assign(state, view.state) because normalizeState only
    // touches the four filter keys.
    const state = {
      model: "",
      topology: "",
      islOsl: "",
      concurrency: "",
      curatedIndex: null,
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

    function mkOptionGroup(labelText, stepNumber) {
      const label = el("div", {
        class: "trtllm-config-selector__label",
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
      const options = el("div", {
        class: "trtllm-config-selector__options",
        role: "group",
        "aria-label": labelText,
      });
      const wrap = el("div", { class: "trtllm-config-selector__field" }, [label, options]);
      return { wrap, options };
    }

    function mkSelectField(labelText, id, stepNumber) {
      const label = el("label", {
        class: "trtllm-config-selector__label",
        for: id,
      });
      label.appendChild(
        el("span", {
          class: "trtllm-config-selector__step",
          "aria-hidden": "true",
          text: String(stepNumber),
        })
      );
      label.appendChild(
        el("span", {
          class: "trtllm-config-selector__labelText",
          text: labelText,
        })
      );
      const select = el("select", { class: "trtllm-config-selector__select", id });
      const wrap = el("div", { class: "trtllm-config-selector__field" }, [label, select]);
      return { wrap, select };
    }

    const id = ++widgetId;
    const selModel = mkSelectField("Model", `trtllm-model-${id}`, 1);
    const selTopo = mkOptionGroup("GPU(s)", 2);
    const selSeq = mkOptionGroup("ISL / OSL", 3);
    const selConc = mkSelectField("Concurrency", `trtllm-conc-${id}`, 4);

    form.appendChild(selModel.wrap);

    const curatedPanel = el("div", { class: "trtllm-config-selector__curated" });
    curatedPanel.hidden = true;
    const curatedHeading = el("div", { class: "trtllm-config-selector__curatedHeading" }, [
      el("span", { text: "Recommended Configs" }),
      el("span", { class: "trtllm-config-selector__curatedBadge", text: "Curated" }),
    ]);
    const curatedGrid = el("div", { class: "trtllm-config-selector__curatedGrid" });
    const curatedHint = el("div", {
      class: "trtllm-config-selector__curatedHint",
      text: "or choose specific GPU & workload below",
    });
    curatedPanel.appendChild(curatedHeading);
    curatedPanel.appendChild(curatedGrid);
    curatedPanel.appendChild(curatedHint);
    form.appendChild(curatedPanel);

    form.appendChild(selTopo.wrap);
    form.appendChild(selSeq.wrap);
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

    const errorBox = el("div", {
      class: "trtllm-config-selector__error",
      text: "",
      role: "status",
      "aria-live": "polite",
    });
    errorBox.hidden = true;

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

    function captureFocusDescriptor() {
      const activeEl = document.activeElement;
      if (!activeEl || !container.contains(activeEl)) return null;
      if (activeEl === selModel.select) {
        return { type: "model-select" };
      }
      if (activeEl === selConc.select) {
        return { type: "select" };
      }
      if (
        activeEl.classList &&
        activeEl.classList.contains("trtllm-config-selector__option")
      ) {
        return {
          type: "button",
          key: activeEl.getAttribute("data-selector-key") || "",
          value: activeEl.getAttribute("data-selector-value") || "",
        };
      }
      return null;
    }

    function restoreFocusDescriptor(descriptor) {
      if (!descriptor) return;
      if (descriptor.type === "model-select") {
        selModel.select.focus();
        return;
      }
      if (descriptor.type === "select") {
        selConc.select.focus();
        return;
      }
      if (descriptor.type !== "button") return;

      const buttons = Array.from(
        container.querySelectorAll(".trtllm-config-selector__option[data-selector-key]")
      );
      const nextButton = buttons.find(
        (button) =>
          button.getAttribute("data-selector-key") === descriptor.key &&
          button.getAttribute("data-selector-value") === descriptor.value
      );
      if (nextButton && !nextButton.disabled) {
        nextButton.focus();
      }
    }

    function applySelection(key, option) {
      state.curatedIndex = null;
      Object.assign(state, nextStateAfterSelection(state, key, option));
      render();
    }

    function applyCuratedSelection(index) {
      if (state.curatedIndex === index) {
        state.curatedIndex = null;
      } else {
        state.curatedIndex = index;
      }
      render();
    }

    function renderCuratedCards(cards, selectedIndex) {
      curatedGrid.innerHTML = "";
      for (let i = 0; i < cards.length; i++) {
        const card = cards[i];
        const isActive = selectedIndex === i;
        const btn = el("button", {
          class: "trtllm-config-selector__curatedCard",
          type: "button",
          "data-status": isActive ? "active" : "available",
          "aria-pressed": isActive ? "true" : "false",
          title: `${card.scenario} — ${card.gpu_compatibility}`,
        });
        btn.appendChild(el("span", {
          class: "trtllm-config-selector__curatedScenario",
          text: card.scenario,
        }));
        const gpuText = (card.gpu_compatibility && card.gpu_compatibility !== "Any")
          ? card.gpu_compatibility : "";
        if (gpuText) {
          btn.appendChild(el("span", {
            class: "trtllm-config-selector__curatedGpu",
            text: gpuText,
          }));
        }
        btn.appendChild(el("span", {
          class: "trtllm-config-selector__curatedFile",
          text: card.config_filename || "",
        }));
        const idx = i;
        btn.addEventListener("click", () => applyCuratedSelection(idx));
        curatedGrid.appendChild(btn);
      }
    }

    function setOptionButtons(groupEl, key, group) {
      groupEl.innerHTML = "";
      for (const option of group.options) {
        const isIncompatible = option.status === "incompatible";
        const button = el("button", {
          class: "trtllm-config-selector__option",
          type: "button",
          "data-status": option.status,
          "data-selected": option.selected ? "true" : "false",
          "data-selector-key": key,
          "data-selector-value": option.value,
          "aria-pressed": option.selected ? "true" : "false",
          "aria-disabled": isIncompatible ? "true" : "false",
          title: isIncompatible
            ? `${option.label} is unavailable for the current selection.`
            : option.hint
              ? `${option.label} (${option.hint})`
              : option.label,
        });
        if (isIncompatible) button.disabled = true;
        button.appendChild(
          el("span", {
            class: "trtllm-config-selector__optionLabel",
            text: option.label,
          })
        );
        if (option.hint) {
          button.appendChild(
            el("span", {
              class: "trtllm-config-selector__optionHint",
              text: option.hint,
            })
          );
        }
        if (option.status === "active-incompatible") {
          button.appendChild(
            el("span", {
              class: "trtllm-config-selector__optionHint",
              text: "Unavailable",
            })
          );
        }
        button.addEventListener("click", () => applySelection(key, option));
        groupEl.appendChild(button);
      }
    }

    function setModelSelect(selectEl, group) {
      const previousValue = state.model || "";
      selectEl.innerHTML = "";
      selectEl.appendChild(
        el("option", {
          value: "",
          text: group.options.length ? "Select model" : "No models available",
        })
      );
      for (const option of group.options) {
        const isUnavailable = option.status === "incompatible";
        const optNode = el("option", {
          value: option.value,
          text: option.label,
        });
        if (isUnavailable) optNode.disabled = true;
        optNode.dataset.status = option.status;
        selectEl.appendChild(optNode);
      }

      const canPreserveValue = group.options.some((option) => option.value === previousValue);
      selectEl.value = canPreserveValue ? previousValue : "";
      const selectedOption = group.options.find((option) => option.value === selectEl.value);
      selectEl.dataset.status = (selectedOption && selectedOption.status) || "idle";
    }

    function setSelectOptions(selectEl, group) {
      const previousValue = state.concurrency || "";
      const visibleOptions = group.options.filter(
        (option) => option.status !== "incompatible"
      );
      selectEl.innerHTML = "";
      selectEl.appendChild(
        el("option", {
          value: "",
          text: visibleOptions.length ? "Select concurrency" : "No concurrency available",
        })
      );
      for (const option of visibleOptions) {
        const optNode = el("option", {
          value: option.value,
          text: option.label,
        });
        optNode.dataset.status = option.status;
        selectEl.appendChild(optNode);
      }

      const canPreserveValue = group.options.some((option) => option.value === previousValue);
      selectEl.value = canPreserveValue ? previousValue : "";
      const selectedOption = group.options.find((option) => option.value === selectEl.value);
      selectEl.dataset.status = (selectedOption && selectedOption.status) || "idle";
      selectEl.setAttribute(
        "aria-invalid",
        selectedOption && selectedOption.status === "active-incompatible" ? "true" : "false"
      );
    }

    function render(depth) {
      if (depth === undefined) depth = 0;
      if (depth > 5) return;
      const focusDescriptor = captureFocusDescriptor();
      const view = createSelectorViewModel(entries, modelsInfo, state);
      if (!statesEqual(state, view.state)) {
        Object.assign(state, view.state);
        return render(depth + 1);
      }

      // Augment model group with curated-only models not in the database
      const dbModelValues = new Set(view.groups.model.options.map((o) => o.value));
      const curatedOnlyModels = uniqBy(
        curatedEntries.filter((e) => !dbModelValues.has(e.model)),
        (e) => e.model
      );
      const modelGroup = { ...view.groups.model };
      if (curatedOnlyModels.length) {
        modelGroup.options = modelGroup.options.slice();
        for (const ce of curatedOnlyModels) {
          const opt = modelOption(ce.model, modelsInfo);
          const isSelected = view.state.model === ce.model;
          modelGroup.options.push({
            ...opt,
            status: isSelected ? "active" : "available",
            selected: isSelected,
          });
        }
        modelGroup.options.sort((a, b) => sortStrings(a.label, b.label));
      }
      setModelSelect(selModel.select, modelGroup);

      const modelCurated = curatedEntriesForModel(curatedEntries, view.state.model);
      const hasCurated = modelCurated.length > 0;
      const curatedSelected = state.curatedIndex != null && state.curatedIndex < modelCurated.length;

      curatedPanel.hidden = !hasCurated;
      if (hasCurated) {
        renderCuratedCards(modelCurated, curatedSelected ? state.curatedIndex : null);
      }

      if (curatedSelected) {
        form.classList.add("trtllm-config-selector__form--dimmed");
      } else {
        form.classList.remove("trtllm-config-selector__form--dimmed");
      }

      setOptionButtons(selTopo.options, "topology", view.groups.topology);
      setOptionButtons(selSeq.options, "islOsl", view.groups.islOsl);
      setSelectOptions(selConc.select, view.groups.concurrency);

      const code = cmdPre.querySelector("code");
      if (curatedSelected) {
        const ce = modelCurated[state.curatedIndex];
        const cmdText = formatCommand(ce);
        code.innerHTML = highlightBash(cmdText);
        code.dataset.raw = cmdText;
        cmdCopyBtn.disabled = !cmdText;
        meta.textContent = "";
        const ceGpu = (ce.gpu_compatibility && ce.gpu_compatibility !== "Any") ? ce.gpu_compatibility : "";
        const metaPrefix = ceGpu
          ? `Scenario: ${ce.scenario || "N/A"} \u00b7 GPU: ${ceGpu} \u00b7 Config: `
          : `Scenario: ${ce.scenario || "N/A"} \u00b7 Config: `;
        meta.appendChild(el("span", { text: metaPrefix }));
        const cfgHref = ce.config_github_url || ce.config_raw_url || "";
        if (cfgHref) {
          meta.appendChild(
            el("a", {
              class: "trtllm-config-selector__configLink",
              href: cfgHref,
              target: "_blank",
              rel: "noopener",
              text: ce.config_path || cfgHref,
            })
          );
        } else {
          meta.appendChild(el("span", { text: ce.config_path || "" }));
        }

        currentEntry = ce;
        resetYamlPanel();
        errorBox.textContent = "";
        errorBox.hidden = true;
      } else if (view.resolvedEntry) {
        const e = view.resolvedEntry;
        code.innerHTML = highlightBash(view.commandText);
        code.dataset.raw = view.commandText;
        cmdCopyBtn.disabled = !view.commandText;
        meta.textContent = "";
        meta.appendChild(el("span", { text: `Profile: ${e.performance_profile || "N/A"} \u00b7 Config: ` }));
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
        errorBox.textContent = view.message;
        errorBox.hidden = !view.message;
      } else {
        code.textContent = "";
        cmdCopyBtn.disabled = true;
        meta.textContent = "";
        currentEntry = null;
        resetYamlPanel();
        errorBox.textContent = view.message;
        errorBox.hidden = !view.message;
      }

      restoreFocusDescriptor(focusDescriptor);
    }

    selModel.select.addEventListener("change", () => {
      state.curatedIndex = null;
      state.model = selModel.select.value;
      state.topology = "";
      state.islOsl = "";
      state.concurrency = "";
      render();
    });

    selConc.select.addEventListener("change", () => {
      state.concurrency = selConc.select.value;
      render();
    });

    cmdCopyBtn.addEventListener("click", async () => {
      const code = $(cmdPre, "code");
      const txt = (code && (code.dataset.raw || code.textContent)) || "";
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
      const hint = isFileProtocol()
        ? "\n\nThis page requires an HTTP server. " +
          "Run: python -m http.server -d docs/build/html " +
          "and open http://localhost:8000/deployment-guide/"
        : "";
      for (const c of containers) {
        c.classList.add("trtllm-config-selector");
        c.textContent = "Failed to load configuration database: " + String(err) + hint;
      }
    }
  }

  if (typeof module !== "undefined" && module.exports) {
    module.exports = {
      buildCompatibilityGroups,
      buildDomains,
      createSelectorViewModel,
      curatedEntriesForModel,
      filterEntriesByState,
      formatCommand,
      formatCuratedCommand: formatCommand,
      nextStateAfterSelection,
      normalizeState,
    };
  }

  if (typeof document === "undefined" || typeof window === "undefined") {
    return;
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", main);
  } else {
    main();
  }
})();
