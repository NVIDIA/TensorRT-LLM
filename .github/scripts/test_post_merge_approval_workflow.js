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

const fs = require("fs");
const path = require("path");

const AsyncFunction = Object.getPrototypeOf(async function () {}).constructor;
const WORKFLOW = path.join(
  __dirname,
  "..",
  "workflows",
  "post-merge-approval.yml",
);

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

function extractGithubScripts(source) {
  const lines = source.split(/\r?\n/);
  const scripts = [];

  for (let index = 0; index < lines.length; index += 1) {
    const marker = lines[index].match(/^(\s*)script:\s*\|\s*$/);
    if (!marker) continue;

    const markerIndent = marker[1].length;
    const contentIndent = markerIndent + 2;
    const body = [];
    for (index += 1; index < lines.length; index += 1) {
      const line = lines[index];
      if (line.trim() === "") {
        body.push("");
      } else if (line.match(/^ */)[0].length <= markerIndent) {
        index -= 1;
        break;
      } else {
        body.push(line.slice(contentIndent));
      }
    }
    scripts.push(body.join("\n"));
  }
  return scripts;
}

async function main() {
  const workflowSource = fs.readFileSync(WORKFLOW, "utf8");
  for (const expected of [
    "pull_request_target:",
    "types: [labeled]",
    "contents: read",
    "pull-requests: write",
    "github.repository == 'NVIDIA/TensorRT-LLM'",
    "github.event.label.name == 'ci: post-merge approved'",
    "concurrency:",
    "cancel-in-progress: false",
    "!cancelled()",
    "listEventsForTimeline",
    "VALIDATED_LABEL_EVENT_ID",
    "secrets.TRTLLM_AGENT_SHARED_TOKEN",
    "secrets.GITHUB_TOKEN",
  ]) {
    assert(workflowSource.includes(expected), "missing workflow contract: " + expected);
  }
  for (const forbidden of [
    "actions/checkout",
    "issues: write",
    "pull-requests: read",
    "synchronize",
  ]) {
    assert(!workflowSource.includes(forbidden), "unsafe workflow contract: " + forbidden);
  }
  const allowedSecrets = new Set([
    "TRTLLM_AGENT_SHARED_TOKEN",
    "GITHUB_TOKEN",
  ]);
  for (const match of workflowSource.matchAll(/\bsecrets\.([A-Z0-9_]+)/g)) {
    assert(
      allowedSecrets.has(match[1]),
      "unexpected workflow secret reference",
    );
  }

  const scripts = extractGithubScripts(workflowSource);
  assert(scripts.length === 2, "expected two github-script blocks");
  const validate = new AsyncFunction("github", "context", "core", scripts[0]);
  const cleanup = new AsyncFunction("github", "context", "core", scripts[1]);

  async function runValidation(
    outcome,
    actor = "candidate",
    sender = true,
    senderLogin = actor,
    events = null,
    timelineError = null,
  ) {
    let request;
    let warning = "";
    const outputs = {};
    const payload = { pull_request: { number: 123 } };
    if (sender) payload.sender = { login: senderLogin };
    const timelineEvents = events ?? [
      {
        id: 101,
        event: "labeled",
        label: { name: "ci: post-merge approved" },
        actor: null,
      },
    ];
    const result = await validate(
      {
        paginate: async () => {
          if (timelineError) throw timelineError;
          return timelineEvents;
        },
        rest: { issues: { listEventsForTimeline: () => {} } },
        request: async (route, args) => {
          request = { route, args };
          if (outcome instanceof Error) throw outcome;
          return { data: { state: outcome } };
        },
      },
      { actor, payload, repo: { owner: "NVIDIA", repo: "TensorRT-LLM" } },
      {
        warning: (message) => {
          warning = message;
        },
        setOutput: (name, value) => {
          outputs[name] = String(value);
        },
      },
    );
    return { result, request, warning, outputs };
  }

  const active = await runValidation(
    "active",
    "context-actor",
    true,
    "approved-user",
  );
  assert(active.result === "true", "active member must be authorized");
  assert(
    active.request.route ===
      "GET /orgs/{org}/teams/{team_slug}/memberships/{username}",
    "unexpected membership route",
  );
  assert(active.request.args.org === "NVIDIA", "unexpected organization");
  assert(
    active.request.args.team_slug === "trt-llm-ci-approvers",
    "unexpected Team slug",
  );
  assert(
    active.request.args.username === "approved-user",
    "label event sender must be checked",
  );
  assert(
    active.outputs.validated_actor === "approved-user",
    "validated actor output must identify the label sender",
  );
  assert(
    (await runValidation("active", "fallback-actor", false)).request.args
      .username === "fallback-actor",
    "context actor must be used when the event sender is absent",
  );
  const latestEvent = await runValidation(
    "active",
    "context-actor",
    true,
    "payload-actor",
    [
      {
        id: 100,
        event: "labeled",
        label: { name: "ci: post-merge approved" },
        actor: { login: "older-actor" },
      },
      {
        id: 101,
        event: "labeled",
        label: { name: "ci: post-merge approved" },
        actor: { login: "latest-approver" },
      },
    ],
  );
  assert(
    latestEvent.request.args.username === "latest-approver",
    "the latest approval-label event actor must be validated",
  );
  assert(
    latestEvent.outputs.label_event_id === "101",
    "the validated label event must be published for cleanup re-checking",
  );
  const missingTimeline = await runValidation(
    "active",
    "context-actor",
    true,
    "approved-user",
    [],
  );
  assert(
    missingTimeline.result === "false",
    "missing timeline event must fail closed",
  );
  assert(
    missingTimeline.request === undefined,
    "membership must not be trusted without a bound label event",
  );
  const timelineReadError = new Error("timeline unavailable");
  const unreadableTimeline = await runValidation(
    "active",
    "context-actor",
    true,
    "approved-user",
    null,
    timelineReadError,
  );
  assert(
    unreadableTimeline.result === "false",
    "timeline API failure must fail closed",
  );
  assert(
    (await runValidation("pending")).result === "false",
    "pending member must be rejected",
  );

  const notFound = new Error("not found");
  notFound.status = 404;
  assert(
    (await runValidation(notFound, "non-member")).result === "false",
    "non-member must be rejected",
  );

  const apiError = new Error("API unavailable");
  apiError.status = 500;
  const apiFailure = await runValidation(apiError);
  assert(apiFailure.result === "false", "API failure must fail closed");
  assert(apiFailure.warning.includes("API unavailable"), "failure must be logged");

  async function runCleanup({
    removeError = null,
    labels = [{ name: "ci: post-merge approved" }],
    events = [
      {
        id: 101,
        event: "labeled",
        label: { name: "ci: post-merge approved" },
        actor: { login: "label-actor" },
      },
    ],
    validatedActor = "label-actor",
    validatedEventId = "101",
    timelineError = null,
  } = {}) {
    const calls = { remove: [], comment: [], warning: [] };
    const previousActor = process.env.VALIDATED_ACTOR;
    const previousEventId = process.env.VALIDATED_LABEL_EVENT_ID;
    process.env.VALIDATED_ACTOR = validatedActor;
    process.env.VALIDATED_LABEL_EVENT_ID = validatedEventId;
    const github = {
      paginate: async () => {
        if (timelineError) throw timelineError;
        return events;
      },
      rest: {
        pulls: {
          get: async () => ({ data: { labels } }),
        },
        issues: {
          listEventsForTimeline: () => {},
          removeLabel: async (args) => {
            calls.remove.push(args);
            if (removeError) throw removeError;
          },
          createComment: async (args) => {
            calls.comment.push(args);
          },
        },
      },
    };
    try {
      await cleanup(
        github,
        {
          actor: "fallback",
          repo: { owner: "NVIDIA", repo: "TensorRT-LLM" },
          payload: {
            action: "labeled",
            sender: { login: "label-actor" },
            pull_request: { number: 123 },
          },
        },
        {
          warning: (message) => calls.warning.push(message),
        },
      );
    } finally {
      if (previousActor === undefined) delete process.env.VALIDATED_ACTOR;
      else process.env.VALIDATED_ACTOR = previousActor;
      if (previousEventId === undefined) {
        delete process.env.VALIDATED_LABEL_EVENT_ID;
      } else {
        process.env.VALIDATED_LABEL_EVENT_ID = previousEventId;
      }
    }
    return calls;
  }

  const unauthorized = await runCleanup();
  assert(unauthorized.remove.length === 1, "label must be removed");
  assert(unauthorized.comment.length === 1, "denial must be explained");
  assert(
    unauthorized.comment[0].body.includes("label-actor"),
    "comment must identify the label actor",
  );

  const concurrentRemoval = new Error("not found");
  concurrentRemoval.status = 404;
  const concurrent = await runCleanup({ removeError: concurrentRemoval });
  assert(
    concurrent.comment.length === 1,
    "concurrent removal must not suppress the denial comment",
  );

  const staleRun = await runCleanup({
    events: [
      {
        id: 101,
        event: "labeled",
        label: { name: "ci: post-merge approved" },
        actor: { login: "label-actor" },
      },
      {
        id: 102,
        event: "labeled",
        label: { name: "ci: post-merge approved" },
        actor: { login: "new-approver" },
      },
    ],
  });
  assert(staleRun.remove.length === 0, "a stale run must not remove a newer label");
  assert(staleRun.comment.length === 0, "a stale run must not post a denial comment");

  const alreadyAbsent = await runCleanup({ labels: [] });
  assert(alreadyAbsent.remove.length === 0, "an absent label needs no cleanup");

  const timelineFailure = new Error("timeline unavailable");
  const failClosed = await runCleanup({ timelineError: timelineFailure });
  assert(failClosed.remove.length === 1, "timeline failure must fail closed");
  assert(
    failClosed.warning[0].includes("timeline unavailable"),
    "timeline failure must be logged",
  );

  const unboundValidation = await runCleanup({ validatedEventId: "" });
  assert(
    unboundValidation.remove.length === 1,
    "cleanup without a bound event must fail closed",
  );

  const missingLatestActor = await runCleanup({
    events: [
      {
        id: 101,
        event: "labeled",
        label: { name: "ci: post-merge approved" },
        actor: null,
      },
    ],
  });
  assert(
    missingLatestActor.remove.length === 1,
    "cleanup without a latest actor must fail closed",
  );

  const sameActorNewEvent = await runCleanup({
    events: [
      {
        id: 102,
        event: "labeled",
        label: { name: "ci: post-merge approved" },
        actor: { login: "label-actor" },
      },
    ],
  });
  assert(
    sameActorNewEvent.remove.length === 0,
    "a newer event from the same actor must not be removed by a stale run",
  );

  console.log(
    "Post-merge approval workflow passed wiring, permissions, active, pending, " +
      "non-member, API-failure, unauthorized-cleanup, commit-persistence, and " +
      "stale-run cleanup tests.",
  );
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
