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
// This bridge is vendored on the trusted default branch beside git-spec and
// ci_archive.py. Arguments are passed directly, without invoking a shell.
const {execFileSync} = require('node:child_process');
const path = require('node:path');

async function archiveSpecs({number, headSha}) {
  if (!Number.isInteger(number) || number < 1 || !/^[0-9a-f]{40}(?:[0-9a-f]{24})?$/.test(headSha)) {
    throw new Error('Invalid archive identity');
  }
  const descriptor = JSON.parse(execFileSync('python3', [
    path.join(__dirname, 'ci_archive.py'), 'archive', '--provider', 'github',
    '--pr', String(number), '--head', headSha,
  ], {
    encoding: 'utf8', maxBuffer: 1024 * 1024,
    env: {...process.env, SPEC_ARCHIVE_CORE: path.join(__dirname, 'git-spec')},
    stdio: ['ignore', 'pipe', 'inherit'],
  }));
  if (descriptor.head_sha !== headSha || !/^[0-9a-f]{64}$/.test(descriptor.sha256 ?? '')) {
    throw new Error('Archive upload did not return a finalized descriptor for the exact removal head');
  }
  return descriptor;
}

module.exports = {archiveSpecs};
