// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

const assert = require('node:assert/strict');
const fs = require('node:fs');

const workflow = fs.readFileSync(
  new URL('../workflows/cleanup-stale-prs.yml', `file://${__dirname}/`),
  'utf8'
);
const marker = '          script: |\n';
const start = workflow.indexOf(marker);
assert.notEqual(start, -1, 'workflow must contain an inline github-script');
const lines = workflow.slice(start + marker.length).split('\n');
const end = lines.findIndex(
  (line) => line.trim() !== '' && !line.startsWith(' '.repeat(12))
);
const script = (end === -1 ? lines : lines.slice(0, end))
  .map((line) => line.replace(/^ {12}/, ''))
  .join('\n');
const AsyncFunction = Object.getPrototypeOf(async function () {}).constructor;
const execute = new AsyncFunction(
  'github',
  'context',
  'core',
  'console',
  'process',
  'setTimeout',
  script
);

const oldDate = (days) =>
  new Date(Date.now() - days * 24 * 60 * 60 * 1000).toISOString();

function summaryCounts(summaries) {
  const [headers, values] = summaries.at(-1);
  return Object.fromEntries(
    headers.map((header, index) => [header.data, values[index]])
  );
}

function pullRequest(number, overrides = {}) {
  return {
    author: { login: `author-${number}` },
    isDraft: false,
    labels: {
      nodes: [],
      pageInfo: { endCursor: null, hasNextPage: false },
    },
    mergeable: 'MERGEABLE',
    number,
    reviewDecision: null,
    state: 'OPEN',
    updatedAt: oldDate(121),
    url: `https://example.test/pull/${number}`,
    ...overrides,
  };
}

async function run({
  pullRequests,
  listingPages,
  labelPages = {},
  dryRun = false,
  closeFailures = 0,
  requireCompleteListingBeforeWrites = false,
  eventLog = [],
}) {
  const events = eventLog;
  const warnings = [];
  const summaries = [];
  const states = new Map(
    Object.entries(pullRequests).map(([number, value]) => [
      Number(number),
      Array.isArray(value) ? value : [value],
    ])
  );
  const reads = new Map();
  const pages =
    listingPages || [Object.values(pullRequests).map((value) => (Array.isArray(value) ? value[0] : value))];
  let listedPages = 0;
  let remainingCloseFailures = closeFailures;

  const github = {
    graphql: async (query, variables) => {
      if (query.includes('pullRequests(')) {
        const pageIndex = variables.cursor === null ? 0 : Number(variables.cursor);
        listedPages += 1;
        events.push(`list:${pageIndex}`);
        return {
          repository: {
            pullRequests: {
              nodes: pages[pageIndex].map(({ number, updatedAt }) => ({
                number,
                updatedAt,
              })),
              pageInfo: {
                endCursor: pageIndex + 1 < pages.length ? String(pageIndex + 1) : null,
                hasNextPage: pageIndex + 1 < pages.length,
              },
            },
          },
        };
      }

      if (query.includes('labels(first: 100, after: $cursor)')) {
        events.push(`labels:${variables.number}:${variables.cursor}`);
        return {
          repository: {
            pullRequest: {
              labels: labelPages[variables.number][Number(variables.cursor)],
            },
          },
        };
      }

      const values = states.get(variables.number);
      const read = reads.get(variables.number) || 0;
      reads.set(variables.number, read + 1);
      events.push(`state:${variables.number}:${read}`);
      return {
        repository: {
          pullRequest: values[Math.min(read, values.length - 1)],
        },
      };
    },
    rest: {
      issues: {
        createComment: async ({ issue_number, body }) => {
          if (requireCompleteListingBeforeWrites) {
            assert.equal(listedPages, pages.length);
          }
          events.push(`comment:${issue_number}:${body}`);
        },
      },
      pulls: {
        update: async ({ pull_number }) => {
          if (requireCompleteListingBeforeWrites) {
            assert.equal(listedPages, pages.length);
          }
          events.push(`close:${pull_number}`);
          if (remainingCloseFailures > 0) {
            remainingCloseFailures -= 1;
            throw new Error('simulated close failure');
          }
        },
      },
    },
  };
  const summary = {
    addHeading: () => summary,
    addTable: (table) => {
      summaries.push(table);
      return summary;
    },
    write: async () => {},
  };
  const core = {
    summary,
    warning: (message) => warnings.push(message),
  };
  const quietConsole = { log: () => {} };
  const immediateTimeout = (callback) => {
    callback();
    return 0;
  };

  await execute(
    github,
    { eventName: dryRun ? 'workflow_dispatch' : 'schedule', repo: { owner: 'NVIDIA', repo: 'TensorRT-LLM' } },
    core,
    quietConsole,
    { env: { DRY_RUN: String(dryRun) } },
    immediateTimeout
  );
  return { events, reads, summaries, warnings };
}

async function main() {
  {
    const pr = pullRequest(1, {
      mergeable: 'CONFLICTING',
      reviewDecision: 'APPROVED',
      updatedAt: oldDate(181),
    });
    const { events, summaries } = await run({ pullRequests: { 1: pr } });
    assert.deepEqual(
      events.filter((event) => event.startsWith('close:') || event.startsWith('comment:')).map((event) => event.split(':')[0]),
      ['close', 'comment'],
      'an old conflicting PR must close before it is commented on'
    );
    assert.deepEqual(summaryCounts(summaries), {
      Mode: 'Run',
      Closed: '1',
      Exempt: '0',
      Pinged: '0',
      Scanned: '1',
      Skipped: '0',
    });
  }

  {
    const cases = {
      2: pullRequest(2),
      3: pullRequest(3, { mergeable: 'CONFLICTING', reviewDecision: 'APPROVED' }),
      4: pullRequest(4, { isDraft: true }),
      5: pullRequest(5, { updatedAt: oldDate(100) }),
    };
    const { events } = await run({ pullRequests: cases });
    const comments = events.filter((event) => event.startsWith('comment:'));
    assert(comments.some((event) => event.startsWith('comment:2:')));
    assert(!comments.some((event) => event.startsWith('comment:3:')));
    assert(comments.some((event) => event.startsWith('comment:4:')));
    assert(!events.some((event) => event.startsWith('state:5:')));
  }

  {
    const unknown = pullRequest(6, { mergeable: 'UNKNOWN' });
    const resolved = pullRequest(6, {
      labels: {
        nodes: Array.from({ length: 100 }, (_, index) => ({
          name: `label-${index}`,
        })),
        pageInfo: { endCursor: '1', hasNextPage: true },
      },
    });
    const unresolved = pullRequest(7, { mergeable: 'UNKNOWN' });
    const { events, reads, summaries, warnings } = await run({
      pullRequests: { 6: [unknown, unknown, resolved], 7: unresolved },
      labelPages: {
        6: [null, { nodes: [], pageInfo: { endCursor: null, hasNextPage: false } }],
      },
    });
    assert.equal(reads.get(6), 3);
    assert.equal(reads.get(7), 3);
    assert.equal(events.filter((event) => event === 'labels:6:1').length, 1);
    assert(events.some((event) => event.startsWith('comment:6:')));
    assert(!events.some((event) => event.startsWith('comment:7:')));
    assert(warnings.some((warning) => warning.includes('did not compute mergeability')));
    assert.deepEqual(
      {
        Pinged: summaryCounts(summaries).Pinged,
        Skipped: summaryCounts(summaries).Skipped,
      },
      { Pinged: '1', Skipped: '1' }
    );
  }

  {
    const firstLabels = {
      nodes: Array.from({ length: 100 }, (_, index) => ({ name: `label-${index}` })),
      pageInfo: { endCursor: '1', hasNextPage: true },
    };
    const pr = pullRequest(8, { labels: firstLabels });
    const { events, summaries } = await run({
      pullRequests: { 8: pr },
      labelPages: {
        8: [null, { nodes: [{ name: 'no-stale' }], pageInfo: { endCursor: null, hasNextPage: false } }],
      },
    });
    assert(events.some((event) => event === 'labels:8:1'));
    assert(!events.some((event) => event.startsWith('comment:8:')));
    assert.equal(summaryCounts(summaries).Exempt, '1');
  }

  {
    const first = pullRequest(9);
    const second = pullRequest(10);
    const { events } = await run({
      pullRequests: { 9: first, 10: second },
      listingPages: [[first], [second]],
      requireCompleteListingBeforeWrites: true,
    });
    assert(events.indexOf('list:1') < events.findIndex((event) => event.startsWith('comment:')));
  }

  {
    const prs = Object.fromEntries(
      Array.from({ length: 51 }, (_, index) => {
        const number = index + 20;
        return [number, pullRequest(number)];
      })
    );
    const { events, summaries, warnings } = await run({ pullRequests: prs });
    assert.equal(events.filter((event) => event.startsWith('comment:')).length, 50);
    assert.deepEqual(
      {
        Pinged: summaryCounts(summaries).Pinged,
        Scanned: summaryCounts(summaries).Scanned,
      },
      { Pinged: '50', Scanned: '50' }
    );
    assert(warnings.some((warning) => warning.includes('50-action safety limit')));
  }

  {
    const pr = pullRequest(80, { mergeable: 'CONFLICTING', updatedAt: oldDate(181) });
    const { events } = await run({ pullRequests: { 80: pr }, dryRun: true });
    assert(!events.some((event) => event.startsWith('close:') || event.startsWith('comment:')));
  }

  {
    const pr = pullRequest(81, { mergeable: 'CONFLICTING', updatedAt: oldDate(181) });
    const { events } = await run({ pullRequests: { 81: pr }, closeFailures: 2 });
    assert.equal(events.filter((event) => event === 'close:81').length, 3);
    assert(events.findIndex((event) => event.startsWith('comment:81:')) > events.lastIndexOf('close:81'));
    const failedEvents = [];
    await assert.rejects(
      run({ pullRequests: { 81: pr }, closeFailures: 3, eventLog: failedEvents }),
      /simulated close failure/
    );
    assert(!failedEvents.some((event) => event.startsWith('comment:81:')));
  }

  console.log('cleanup_stale_prs tests passed');
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
