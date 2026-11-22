// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
const assert = require('node:assert/strict');
const path = require('node:path');
process.env.GITHUB_REPOSITORY = 'NVIDIA/TensorRT-LLM';
const removal = require(process.env.SPEC_REMOVAL_SCRIPT || path.join(__dirname, 'remove_ready_specs.js'));
const A = 'a'.repeat(40), B = 'b'.repeat(40), C = 'c'.repeat(40), D = 'd'.repeat(40);
const T = '1'.repeat(40), U = '2'.repeat(40), V = '3'.repeat(40);
const HASH = 'f'.repeat(64);

function scenario({failUpload = false, stale = false, wrongArchive = false} = {}) {
  const order = [];
  let writes = 0;
  const github = {
    rest: {
      git: {
        getTree: async ({tree_sha}) => {
          if (tree_sha === B) return {data: {tree: []}};
          if (tree_sha === A || tree_sha === C) return {data: {tree: [{path: 'specs', type: 'tree', sha: T}]}};
          if (tree_sha === T) return {data: {tree: [{path: '001/spec.md', type: 'blob', mode: '100644', sha: D}]}};
          throw new Error(`Unexpected tree ${tree_sha}`);
        },
        getRef: async () => ({data: {object: {sha: C}}}),
      },
      repos: {compareCommitsWithBasehead: async () => ({data: {merge_base_commit: {sha: B}, files: [{filename: 'specs/001/spec.md'}]}})},
      issues: {listComments: () => {}, createComment: async () => ({data: {html_url: 'https://github.com/comment'}})},
    },
    paginate: async () => [],
  };
  const pusher = {graphql: async (query, args) => {
    if (query.startsWith('query')) return {viewer: {login: 'archive-bot', name: 'Archive Bot', databaseId: 123}};
    writes++;
    order.push(`write:${args.input.expectedHeadOid}`);
    if (stale && writes === 1) {
      throw {errors: [{type: 'STALE_DATA', message: 'expected branch to point to a different head'}]};
    }
    return {createCommitOnBranch: {commit: {oid: D, author: {name: 'Archive Bot', email: '123+archive-bot@users.noreply.github.com'}}}};
  }};
  const repo = {full_name: 'NVIDIA/TensorRT-LLM', owner: {login: 'NVIDIA', type: 'Organization'}, name: 'TensorRT-LLM'};
  const context = {repo: {owner: 'NVIDIA', repo: 'TensorRT-LLM'}, payload: {
    action: 'ready_for_review', repository: repo, pull_request: {
      number: 7, title: '[None][chore] example', state: 'open', draft: false, changed_files: 1,
      head: {sha: A, ref: 'feature', repo}, base: {sha: B, ref: 'main', repo},
    },
  }};
  const archive = async ({headSha}) => {
    order.push(`archive:${headSha}`);
    if (failUpload) throw new Error('upload failed');
    return {head_sha: wrongArchive ? C : headSha, sha256: HASH};
  };
  return {order, writes: () => writes, run: () => removal.run({github, pusher, context, core: {info() {}, warning() {}}, archive})};
}

(async () => {
  assert(removal.isSpecPath('specs/001/spec.md'));
  assert(removal.isSpecPath('openspec/changes/fast/proposal.md'));
  for (const name of ['openspec/specs/durable.md', 'specs-old/a.md', 'openspec/changes-old/a.md', 'specs/../code', 'specs/.github/evil.yml']) {
    assert.equal(removal.isSpecPath(name), false, name);
  }
  const failed = scenario({failUpload: true});
  await assert.rejects(failed.run, /upload failed/);
  assert.equal(failed.writes(), 0, 'upload failure must prevent every removal write');
  const wrong = scenario({wrongArchive: true});
  await assert.rejects(wrong.run, /exact head/);
  assert.equal(wrong.writes(), 0, 'wrong-head archive must prevent every removal write');
  const retry = scenario({stale: true});
  await retry.run();
  assert.deepEqual(retry.order, [`archive:${A}`, `write:${A}`, `archive:${C}`, `write:${C}`]);
  assert.throws(() => removal.specChangeSet({
    head: new Map([['specs/001/spec.md', {sha: D, mode: '100755'}]]),
    base: new Map([['specs/001/spec.md', {sha: D, mode: '100644'}]]),
  }), /mode faithfully/, 'mode-only deltas cannot be silently discarded');

  const seen = [];
  const github = {rest: {git: {getTree: async ({tree_sha}) => {
    seen.push(tree_sha);
    if (tree_sha === A) return {data: {tree: [{path: 'openspec', type: 'tree', sha: T}]}};
    if (tree_sha === T) return {data: {tree: [{path: 'changes', type: 'tree', sha: U}, {path: 'specs', type: 'tree', sha: V}]}};
    if (tree_sha === U) return {data: {tree: [{path: 'fast/proposal.md', type: 'blob', mode: '100644', sha: D}]}};
    throw new Error('Durable OpenSpec tree must not be read');
  }}}};
  const result = await removal.enumerateSpecBlobs(github, {owner: 'NVIDIA', repo: 'TensorRT-LLM', treeish: A});
  assert.deepEqual([...result.keys()], ['openspec/changes/fast/proposal.md']);
  assert(!seen.includes(V));
  const linkTree = {rest: {git: {getTree: async ({tree_sha}) => ({data: {tree: tree_sha === A
    ? [{path: 'specs', type: 'tree', sha: T}]
    : [{path: '001/link', type: 'blob', mode: '120000', sha: D}]}})}}};
  await assert.rejects(() => removal.enumerateSpecBlobs(linkTree,
    {owner: 'NVIDIA', repo: 'TensorRT-LLM', treeish: A}), /symlink/);
  console.log('GitHub migration: archive failure, head mismatch, stale retry and OpenSpec boundary passed');
})().catch(error => { console.error(error); process.exitCode = 1; });
