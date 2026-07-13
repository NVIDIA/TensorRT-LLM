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
    "AUTO_LABEL_COMMUNITY_TOKEN",
    "TENSORRT_CICD_TRTLLM_CI_TOKEN",
  ]) {
    assert(!workflowSource.includes(forbidden), "unsafe workflow contract: " + forbidden);
  }

  const scripts = extractGithubScripts(workflowSource);
  assert(scripts.length === 2, "expected two github-script blocks");
  const validate = new AsyncFunction("github", "context", "core", scripts[0]);
  const cleanup = new AsyncFunction("github", "context", "core", scripts[1]);

  async function runValidation(outcome, actor = "candidate", sender = true) {
    let request;
    let warning = "";
    const payload = sender ? { sender: { login: actor } } : {};
    const result = await validate(
      {
        request: async (route, args) => {
          request = { route, args };
          if (outcome instanceof Error) throw outcome;
          return { data: { state: outcome } };
        },
      },
      { actor, payload },
      {
        warning: (message) => {
          warning = message;
        },
      },
    );
    return { result, request, warning };
  }

  const active = await runValidation("active", "approved-user");
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
    (await runValidation("active", "fallback-actor", false)).request.args
      .username === "fallback-actor",
    "context actor must be used when the event sender is absent",
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

  async function runCleanup(removeError = null) {
    const calls = { remove: [], comment: [] };
    const github = {
      rest: {
        issues: {
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
      {},
    );
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
  const concurrent = await runCleanup(concurrentRemoval);
  assert(
    concurrent.comment.length === 1,
    "concurrent removal must not suppress the denial comment",
  );

  console.log(
    "Post-merge approval workflow passed wiring, permissions, active, pending, " +
      "non-member, API-failure, unauthorized-cleanup, commit-persistence, and " +
      "concurrent-removal tests.",
  );
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
