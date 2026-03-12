const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

const REPO_ROOT = path.resolve(__dirname, "../../..");
const SELECTOR_JS = path.join(REPO_ROOT, "docs/source/_static/config_selector.js");
const CONFIG_DB_JSON = path.join(REPO_ROOT, "docs/source/_static/config_db.json");
const DEEPSEEK_MODEL = "deepseek-ai/DeepSeek-R1-0528";
const DEEPSEEK_NVFP4_MODEL = "nvidia/DeepSeek-R1-0528-FP4-v2";

function loadSelectorExports() {
  const source = fs.readFileSync(SELECTOR_JS, "utf8");
  const sandbox = {
    module: { exports: {} },
    exports: {},
    console,
    URL,
    Map,
    Set,
    Promise,
    setTimeout,
    clearTimeout,
    fetch() {
      throw new Error("fetch should not be used in selector logic tests");
    },
    XMLHttpRequest: function XMLHttpRequest() {},
    navigator: { clipboard: null },
    window: { location: { protocol: "https:" } },
    document: {
      readyState: "loading",
      baseURI: "https://example.com/docs/",
      addEventListener() {},
      querySelector() {
        return null;
      },
      querySelectorAll() {
        return [];
      },
      createElement() {
        throw new Error("DOM rendering should not run in selector logic tests");
      },
      body: {
        appendChild() {},
        removeChild() {},
      },
    },
  };
  vm.createContext(sandbox);
  vm.runInContext(source, sandbox, { filename: SELECTOR_JS });
  return sandbox.module.exports;
}

function loadConfigDb() {
  return JSON.parse(fs.readFileSync(CONFIG_DB_JSON, "utf8"));
}

function findOption(group, value) {
  return group.options.find((option) => option.value === value);
}

function setupModel(model) {
  const selector = loadSelectorExports();
  const payload = loadConfigDb();
  const entries = payload.entries.filter((entry) => entry.model === model);
  return { selector, payload, entries };
}

function setupDeepSeekNvfp4() {
  return setupModel(DEEPSEEK_NVFP4_MODEL);
}

test("selector exports a pure view-model helper for compatibility logic", () => {
  const selector = loadSelectorExports();
  assert.equal(typeof selector.createSelectorViewModel, "function");
});

test("selector keeps both 4x and 8x B200 topologies visible for DeepSeek NVFP4", () => {
  const { selector, payload, entries } = setupDeepSeekNvfp4();

  const view = selector.createSelectorViewModel(entries, payload.models, {
    model: DEEPSEEK_NVFP4_MODEL,
    topology: "",
    islOsl: "",
    concurrency: "",
  });

  assert.ok(findOption(view.groups.topology, "4|B200_NVL"));
  assert.ok(findOption(view.groups.topology, "8|B200_NVL"));
});

test("selector marks 1024/1024 unavailable for 4x B200 on DeepSeek NVFP4", () => {
  const { selector, payload, entries } = setupDeepSeekNvfp4();

  const view = selector.createSelectorViewModel(entries, payload.models, {
    model: DEEPSEEK_NVFP4_MODEL,
    topology: "4|B200_NVL",
    islOsl: "",
    concurrency: "",
  });

  assert.equal(findOption(view.groups.islOsl, "1024|1024").status, "incompatible");
  assert.equal(findOption(view.groups.islOsl, "1024|8192").status, "available");
  assert.equal(findOption(view.groups.islOsl, "8192|1024").status, "available");
});

test("selector preserves invalid active selections and explains the clash", () => {
  const { selector, payload, entries } = setupDeepSeekNvfp4();

  const view = selector.createSelectorViewModel(entries, payload.models, {
    model: DEEPSEEK_NVFP4_MODEL,
    topology: "4|B200_NVL",
    islOsl: "1024|1024",
    concurrency: "",
  });

  assert.equal(findOption(view.groups.islOsl, "1024|1024").status, "active-incompatible");
  assert.match(view.message, /ISL \/ OSL 1024 \/ 1024 is not available/);
  assert.match(view.message, /1024 \/ 8192/);
  assert.match(view.message, /8192 \/ 1024/);
});

test("selector resolves a single command for 8x B200 1024/1024 DeepSeek NVFP4", () => {
  const { selector, payload, entries } = setupDeepSeekNvfp4();

  const view = selector.createSelectorViewModel(entries, payload.models, {
    model: DEEPSEEK_NVFP4_MODEL,
    topology: "8|B200_NVL",
    islOsl: "1024|1024",
    concurrency: "1",
  });

  assert.equal(view.finalEntries.length, 1);
  assert.match(view.commandText, /1k1k_tp8_conc1\.yaml/);
  assert.match(view.commandText, /^trtllm-serve nvidia\/DeepSeek-R1-0528-FP4-v2 \\/m);
});

test("selector keeps alternative topologies available despite downstream clashes", () => {
  const { selector, payload, entries } = setupModel(DEEPSEEK_MODEL);

  const view = selector.createSelectorViewModel(entries, payload.models, {
    model: DEEPSEEK_MODEL,
    topology: "8|B200_NVL",
    islOsl: "1024|8192",
    concurrency: "1024",
  });

  assert.equal(findOption(view.groups.topology, "8|H200_SXM").status, "available");
});

test("changing ISL / OSL clears stale downstream concurrency selections", () => {
  const { selector } = setupModel(DEEPSEEK_MODEL);

  const nextState = selector.nextStateAfterSelection(
    {
      model: DEEPSEEK_MODEL,
      topology: "4|B200_NVL",
      islOsl: "8192|1024",
      concurrency: "1024",
    },
    "islOsl",
    {
      value: "1024|8192",
      selected: false,
      status: "available",
    }
  );

  assert.equal(nextState.model, DEEPSEEK_MODEL);
  assert.equal(nextState.topology, "4|B200_NVL");
  assert.equal(nextState.islOsl, "1024|8192");
  assert.equal(nextState.concurrency, "");
});

test("selector derives concurrency labels from the compatible prefix subset", () => {
  const { selector, payload, entries } = setupModel(DEEPSEEK_MODEL);

  const view = selector.createSelectorViewModel(entries, payload.models, {
    model: DEEPSEEK_MODEL,
    topology: "8|B200_NVL",
    islOsl: "1024|1024",
    concurrency: "",
  });

  assert.equal(findOption(view.groups.concurrency, "2").label, "2 · Low Latency");
  assert.equal(findOption(view.groups.concurrency, "2048").label, "2048 · Max Throughput");
});

test("selector auto-selects singleton concurrency without a separate profile badge", () => {
  const { selector, payload, entries } = setupDeepSeekNvfp4();

  const view = selector.createSelectorViewModel(entries, payload.models, {
    model: DEEPSEEK_NVFP4_MODEL,
    topology: "4|B200_NVL",
    islOsl: "1024|8192",
    concurrency: "",
  });

  assert.equal(view.state.concurrency, "2048");
  assert.equal(view.selectedConcurrencyBadge, "");
  assert.equal(view.finalEntries.length, 1);
});

test("model buttons show friendly display name, with HF ID as hint", () => {
  const { selector, payload, entries } = setupDeepSeekNvfp4();

  const view = selector.createSelectorViewModel(entries, payload.models, {
    model: "",
    topology: "",
    islOsl: "",
    concurrency: "",
  });

  const option = findOption(view.groups.model, DEEPSEEK_NVFP4_MODEL);
  assert.ok(option, "model option should exist");
  assert.ok(
    !option.label.includes(DEEPSEEK_NVFP4_MODEL),
    "label should not contain raw HF ID"
  );
  assert.equal(option.hint, DEEPSEEK_NVFP4_MODEL);
});
