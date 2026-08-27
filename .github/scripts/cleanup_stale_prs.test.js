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
  'Date',
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
    state: 'OPEN',
    updatedAt: oldDate(121),
    url: `https://example.test/pull/${number}`,
    ...overrides,
  };
}

function conflictWarning(number, createdAt) {
  return {
    issue_number: number,
    body:
      `Conflict warning\n\n<!-- stale-pr-cleanup:111:${number}:` +
      'conflict-warning -->',
    created_at: createdAt,
    user: { login: 'github-actions[bot]' },
  };
}

async function run({
  pullRequests,
  listingPages,
  labelPages = {},
  dryRun = false,
  closeFailures = 0,
  commentFailures = {},
  ambiguousCommentFailures = {},
  initialComments = [],
  stateFailures = {},
  requireCompleteListingBeforeWrites = false,
  eventLog = [],
  nowValues = [],
  eventName = dryRun ? 'workflow_dispatch' : 'schedule',
  // Value of the STALE_PR_CLEANUP_LIVE repository variable. GitHub renders an
  // unset variable as an empty string.
  live = 'true',
}) {
  const events = eventLog;
  const warnings = [];
  const summaries = [];
  const persistedComments = [...initialComments];
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
  const remainingCommentFailures = new Map(
    Object.entries(commentFailures).map(([number, count]) => [
      Number(number),
      count,
    ])
  );
  const remainingAmbiguousCommentFailures = new Map(
    Object.entries(ambiguousCommentFailures).map(([number, count]) => [
      Number(number),
      count,
    ])
  );
  const remainingStateFailures = new Map(
    Object.entries(stateFailures).map(([number, count]) => [
      Number(number),
      count,
    ])
  );

  const github = {
    paginate: async (operation, { issue_number }) => {
      assert.equal(operation, github.rest.issues.listComments);
      events.push(`list-comments:${issue_number}`);
      return persistedComments
        .filter((comment) => comment.issue_number === issue_number)
        .map(({ body, created_at, user }) => ({ body, created_at, user }));
    },
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
      const failures = remainingStateFailures.get(variables.number) || 0;
      if (failures > 0) {
        remainingStateFailures.set(variables.number, failures - 1);
        throw new Error('simulated state read failure');
      }
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
          const ambiguousFailures =
            remainingAmbiguousCommentFailures.get(issue_number) || 0;
          if (ambiguousFailures > 0) {
            remainingAmbiguousCommentFailures.set(
              issue_number,
              ambiguousFailures - 1
            );
            persistedComments.push({
              issue_number,
              body,
              created_at: new Date(defaultNow).toISOString(),
              user: { login: 'github-actions[bot]' },
            });
            throw new Error('simulated lost comment response');
          }
          const failures = remainingCommentFailures.get(issue_number) || 0;
          if (failures > 0) {
            remainingCommentFailures.set(issue_number, failures - 1);
            throw new Error('simulated comment failure');
          }
          persistedComments.push({
            issue_number,
            body,
            created_at: new Date(defaultNow).toISOString(),
            user: { login: 'github-actions[bot]' },
          });
        },
        listComments: async () => {},
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
  const defaultNow = Date.now();
  let nowIndex = 0;
  const fakeDate = {
    now: () => {
      if (nowValues.length === 0) {
        return defaultNow;
      }
      const value = nowValues[Math.min(nowIndex, nowValues.length - 1)];
      nowIndex += 1;
      return value;
    },
    parse: Date.parse,
  };

  await execute(
    github,
    {
      eventName,
      repo: { owner: 'NVIDIA', repo: 'TensorRT-LLM' },
      runId: 123456,
    },
    core,
    quietConsole,
    { env: { DRY_RUN: String(dryRun), LIVE: String(live) } },
    immediateTimeout,
    fakeDate
  );
  return { events, persistedComments, reads, summaries, warnings };
}

async function main() {
  {
    const pr = pullRequest(1, {
      mergeable: 'CONFLICTING',
      updatedAt: oldDate(181),
    });
    const { events, summaries } = await run({ pullRequests: { 1: pr } });
    assert(!events.some((event) => event.startsWith('close:')));
    assert(
      events.some(
        (event) =>
          event.startsWith('comment:1:') &&
          event.includes('closed after another 60 days of inactivity') &&
          event.includes(':conflict-warning -->')
      )
    );
    assert.deepEqual(summaryCounts(summaries), {
      Mode: 'Run',
      Closed: '0',
      Exempt: '0',
      Pinged: '1',
      Scanned: '1',
      Skipped: '0',
    });
  }

  {
    const cases = {
      2: pullRequest(2),
      3: pullRequest(3, { mergeable: 'CONFLICTING' }),
      4: pullRequest(4, { isDraft: true }),
      5: pullRequest(5, { updatedAt: oldDate(100) }),
    };
    const { events } = await run({ pullRequests: cases });
    const comments = events.filter((event) => event.startsWith('comment:'));
    assert(comments.some((event) => event.startsWith('comment:2:')));
    assert(comments.some((event) => event.startsWith('comment:3:')));
    assert(comments.some((event) => event.startsWith('comment:4:')));
    assert(events.some((event) => event.startsWith('state:5:')));
    assert(!comments.some((event) => event.startsWith('comment:5:')));
  }

  {
    // GitHub orders this connection loosely, so a recently updated pull request
    // can be returned ahead of long-idle ones, both within a page and across
    // pages. Every page must still be scanned and filtered locally.
    const fresh = pullRequest(10, { updatedAt: oldDate(2) });
    const idle = pullRequest(11);
    const oldest = pullRequest(12, { updatedAt: oldDate(200) });
    const { events, summaries } = await run({
      pullRequests: { 10: fresh, 11: idle, 12: oldest },
      listingPages: [[fresh, idle], [oldest]],
    });
    assert(!events.some((event) => event.startsWith('state:10:')));
    assert(events.some((event) => event.startsWith('comment:11:')));
    assert(events.some((event) => event.startsWith('comment:12:')));
    assert.deepEqual(summaryCounts(summaries), {
      Mode: 'Run',
      Closed: '0',
      Exempt: '0',
      Pinged: '2',
      Scanned: '2',
      Skipped: '0',
    });
  }

  {
    // Scheduled runs stay in dry-run mode until STALE_PR_CLEANUP_LIVE is set.
    const { events, summaries } = await run({
      pullRequests: { 13: pullRequest(13) },
      eventName: 'schedule',
      live: '',
    });
    assert(!events.some((event) => event.startsWith('comment:')));
    assert(!events.some((event) => event.startsWith('close:')));
    assert.equal(summaryCounts(summaries).Mode, 'Dry run');
    assert.equal(summaryCounts(summaries).Pinged, '1');
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
    assert(events.some((event) => event.startsWith('comment:7:')));
    assert.deepEqual(
      {
        Pinged: summaryCounts(summaries).Pinged,
        Skipped: summaryCounts(summaries).Skipped,
      },
      { Pinged: '2', Skipped: '0' }
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
    const { events, summaries } = await run({ pullRequests: { 80: pr }, dryRun: true });
    assert(!events.some((event) => event.startsWith('close:') || event.startsWith('comment:')));
    assert.equal(summaryCounts(summaries).Pinged, '1');
  }

  {
    const warningTime = oldDate(61);
    const pr = pullRequest(81, { mergeable: 'CONFLICTING', updatedAt: warningTime });
    const warning = conflictWarning(81, warningTime);
    const { events } = await run({
      pullRequests: { 81: pr },
      closeFailures: 2,
      initialComments: [warning],
    });
    assert.equal(events.filter((event) => event === 'close:81').length, 3);
    assert(events.findIndex((event) => event.startsWith('comment:81:')) > events.lastIndexOf('close:81'));
    const failedEvents = [];
    const next = pullRequest(82);
    const { summaries, warnings } = await run({
      pullRequests: { 81: pr, 82: next },
      closeFailures: 3,
      eventLog: failedEvents,
      initialComments: [warning],
    });
    assert(!failedEvents.some((event) => event.startsWith('comment:81:')));
    assert(failedEvents.some((event) => event.startsWith('comment:82:')));
    assert.equal(summaryCounts(summaries).Skipped, '1');
    assert(warnings.some((warning) => warning.includes('Continuing with the next')));
  }

  {
    const warningTime = oldDate(59);
    const pr = pullRequest(85, {
      mergeable: 'CONFLICTING',
      updatedAt: warningTime,
    });
    const { events, summaries } = await run({
      pullRequests: { 85: pr },
      initialComments: [conflictWarning(85, warningTime)],
    });
    assert(!events.some((event) => event.startsWith('state:85:')));
    assert.equal(summaryCounts(summaries).Scanned, '0');
  }

  {
    const warningTime = oldDate(181);
    const pr = pullRequest(86, {
      mergeable: 'CONFLICTING',
      updatedAt: oldDate(121),
    });
    const { events, summaries } = await run({
      pullRequests: { 86: pr },
      initialComments: [conflictWarning(86, warningTime)],
    });
    assert(!events.some((event) => event.startsWith('close:86')));
    assert(
      events.some(
        (event) =>
          event.startsWith('comment:86:') &&
          event.includes(':conflict-warning -->')
      )
    );
    assert.equal(summaryCounts(summaries).Pinged, '1');
  }

  {
    const warningTime = oldDate(61);
    const pr = pullRequest(87, {
      mergeable: 'MERGEABLE',
      updatedAt: warningTime,
    });
    const { events, summaries } = await run({
      pullRequests: { 87: pr },
      initialComments: [conflictWarning(87, warningTime)],
    });
    assert(!events.some((event) => event.startsWith('close:87')));
    assert(!events.some((event) => event.startsWith('comment:87:')));
    assert.equal(summaryCounts(summaries).Skipped, '1');
  }

  {
    const warningTime = oldDate(61);
    const closing = pullRequest(83, {
      mergeable: 'CONFLICTING',
      updatedAt: warningTime,
    });
    const next = pullRequest(84);
    const { events, summaries, warnings } = await run({
      pullRequests: { 83: closing, 84: next },
      commentFailures: { 83: 3 },
      initialComments: [conflictWarning(83, warningTime)],
    });
    assert.equal(events.filter((event) => event === 'close:83').length, 1);
    assert.equal(
      events.filter((event) => event.startsWith('comment:83:')).length,
      3
    );
    assert(events.some((event) => event.startsWith('comment:84:')));
    assert.equal(summaryCounts(summaries).Closed, '1');
    assert.equal(summaryCounts(summaries).Pinged, '1');
    assert(warnings.some((warning) => warning.includes('simulated comment failure')));
  }

  {
    const first = pullRequest(90);
    const second = pullRequest(91);
    const baseNow = Date.now();
    const { events, summaries, warnings } = await run({
      pullRequests: { 90: first, 91: second },
      nowValues: [baseNow, baseNow, baseNow, baseNow + 12 * 60 * 1000],
    });
    assert(events.some((event) => event.startsWith('state:90:')));
    assert(!events.some((event) => event.startsWith('state:91:')));
    assert.equal(summaryCounts(summaries).Scanned, '1');
    assert(warnings.some((warning) => warning.includes('12-minute processing limit')));
  }

  {
    const first = pullRequest(92);
    const second = pullRequest(93);
    const baseNow = Date.now();
    const { events, summaries, warnings } = await run({
      pullRequests: { 92: first, 93: second },
      listingPages: [[first], [second]],
      nowValues: [baseNow, baseNow, baseNow + 12 * 60 * 1000],
    });
    assert(events.includes('list:0'));
    assert(!events.includes('list:1'));
    assert(!events.some((event) => event.startsWith('state:')));
    assert.equal(summaryCounts(summaries).Scanned, '0');
    assert.equal(
      warnings.filter((warning) => warning.includes('12-minute processing limit')).length,
      1
    );
  }

  {
    const first = pullRequest(94);
    const second = pullRequest(95);
    const { events, summaries, warnings } = await run({
      pullRequests: { 94: first, 95: second },
      stateFailures: { 94: 1 },
    });
    assert(events.some((event) => event.startsWith('comment:95:')));
    assert.equal(summaryCounts(summaries).Pinged, '1');
    assert.equal(summaryCounts(summaries).Skipped, '1');
    assert(warnings.some((warning) => warning.includes('simulated state read failure')));
  }

  {
    const pr = pullRequest(96);
    const { events, persistedComments, summaries } = await run({
      pullRequests: { 96: pr },
      ambiguousCommentFailures: { 96: 1 },
    });
    assert.equal(
      events.filter((event) => event.startsWith('comment:96:')).length,
      1
    );
    assert(events.includes('list-comments:96'));
    assert.equal(persistedComments.length, 1);
    assert(
      persistedComments[0].body.includes(
        '<!-- stale-pr-cleanup:123456:96:ping -->'
      )
    );
    assert.equal(summaryCounts(summaries).Pinged, '1');
  }

  console.log('cleanup_stale_prs tests passed');
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
