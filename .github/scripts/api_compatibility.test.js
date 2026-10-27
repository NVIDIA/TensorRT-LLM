// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');

// Execute the actual github-script body; changes to workflow policy are tested
// without duplicating its path or label logic in a second implementation.
const workflow = fs.readFileSync(
  path.join(__dirname, '../workflows/llm-api-compatibility.yml'), 'utf8');
const body = workflow.split('          script: |\n')[1];
assert.ok(body, 'Workflow github-script body must be present');
const script = body.split('\n').map((line) => line.replace(/^            /, '')).join('\n');
const AsyncFunction = Object.getPrototypeOf(async function () {}).constructor;
const run = new AsyncFunction('github', 'context', 'core', script);

async function check(pages, labels = [], title = '[None][test] update') {
  const failures = [];
  const listFiles = () => { throw new Error('Use pagination'); };
  const github = {
    rest: { pulls: { listFiles } },
    paginate: async (method, options) => {
      assert.equal(method, listFiles);
      assert.equal(options.per_page, 100);
      assert.equal(options.pull_number, 42);
      return pages.flat();
    },
  };
  const context = {
    repo: { owner: 'NVIDIA', repo: 'TensorRT-LLM' },
    payload: { pull_request: { number: 42, title, labels } },
  };
  await run(github, context, {
    info: () => {},
    setFailed: (message) => failures.push(message),
  });
  return failures;
}

for (const filename of [
  'tensorrt_llm/visual_gen/__init__.py',
  'tensorrt_llm/visual_gen/args.py',
  'tensorrt_llm/visual_gen/params.py',
  'tests/unittest/api_stability/references/visual_gen_api.yaml',
  'tests/unittest/api_stability/references_committed/llm.yaml',
  'tensorrt_llm/llmapi/llm_args.py',
  'tensorrt_llm/usage/llm_args_golden_manifest.json',
]) {
  test(`requires classification: ${filename}`, async () => {
    assert.equal((await check([[{ filename }]])).length, 1);
    assert.deepEqual(await check([[{ filename }]], [{ name: 'api-compatible' }]), []);
  });
}

test('unrelated runtime changes require no API label', async () => {
  assert.deepEqual(await check([[{ filename: 'tensorrt_llm/_torch/visual_gen/pipeline.py' }]]), []);
});

test('renames and deletions use previous_filename, including later pages', async () => {
  assert.equal((await check([
    [{ filename: 'README.md' }],
    [{ filename: 'other/file.py', previous_filename: 'tensorrt_llm/visual_gen/args.py' }],
  ])).length, 1);
});

for (const labels of [undefined, null, [], [{ name: 'api-compatible' }, { name: 'api-breaking' }]]) {
  test(`rejects absent or conflicting classification: ${JSON.stringify(labels)}`, async () => {
    assert.equal((await check([[{ filename: 'tensorrt_llm/visual_gen/args.py' }]], labels)).length, 1);
  });
}

test('breaking classification requires BREAKING in title', async () => {
  const files = [[{ filename: 'tensorrt_llm/visual_gen/args.py' }]];
  const labels = [{ name: 'api-breaking' }];
  assert.match((await check(files, labels))[0], /BREAKING/);
  assert.deepEqual(await check(files, labels, '[None][test] BREAKING update'), []);
});
