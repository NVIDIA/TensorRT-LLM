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

const { mkdtempSync, rmSync, writeFileSync } = require('node:fs');
const { execFileSync } = require('node:child_process');
const { tmpdir } = require('node:os');
const { join } = require('node:path');
const { withReserve } = require('./semantic_review_request');

const ARTIFACT = 'semantic-review-cursor';

async function findCursorArtifact({ github, context }) {
  if (context.eventName !== 'schedule') return;
  return withReserve(github, async () => {
    const artifacts = await github.paginate(github.rest.actions.listArtifactsForRepo,
      { ...context.repo, name: ARTIFACT, per_page: 100 });
    artifacts.sort((a, b) => Date.parse(b.created_at) - Date.parse(a.created_at) || b.id - a.id);
    const repository = `${context.repo.owner}/${context.repo.repo}`.toLowerCase();
    for (const artifact of artifacts) {
      if (artifact.name !== ARTIFACT) continue;
      const { data: run } = await github.rest.actions.getWorkflowRun({
        ...context.repo, run_id: artifact.workflow_run.id,
      });
      if (run.event !== 'schedule' ||
          run.path.split('@')[0] !== '.github/workflows/semantic-review.yml' ||
          run.repository.full_name.toLowerCase() !== repository ||
          run.head_repository.full_name.toLowerCase() !== repository) continue;
      if (artifact.expired) throw new Error('The semantic review cursor artifact has expired.');
      return artifact;
    }
  });
}

function parseCursor(text) {
  const cursor = JSON.parse(text)?.last_pr;
  if (!Number.isSafeInteger(cursor) || cursor <= 0) {
    throw new Error('The semantic review cursor must contain a positive PR number.');
  }
  return cursor;
}

async function restoreCursor({ github, context }) {
  const artifact = await findCursorArtifact({ github, context });
  if (!artifact) return;
  const { data } = await withReserve(github, () => github.rest.actions.downloadArtifact({
    ...context.repo, artifact_id: artifact.id, archive_format: 'zip',
  }));
  const directory = mkdtempSync(join(tmpdir(), 'semantic-review-cursor-'));
  try {
    const archive = join(directory, 'cursor.zip');
    writeFileSync(archive, Buffer.from(data));
    return parseCursor(execFileSync('unzip', ['-p', archive, 'cursor.json'],
      { encoding: 'utf8', maxBuffer: 4096 }));
  } finally {
    rmSync(directory, { recursive: true, force: true });
  }
}

module.exports = { findCursorArtifact, parseCursor, restoreCursor };
