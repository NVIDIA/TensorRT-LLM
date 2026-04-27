# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""YAML test-db loading and block/stage matching for CBTS.

- `Stage` carries stage metadata (yaml_stem, cpu_arch, mako) as provided by Groovy.
- `Block` is a condition-block within a test-db YAML.
- `YAMLIndex` loads all test-db YAMLs and provides a test_id -> blocks lookup.
- `block_matches_stage` implements the trt-test-db matching semantics:
  ranges / wildcards / terms. Generic over field names — no hardcoded keys.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from fnmatch import fnmatch
from pathlib import Path
from typing import Iterable, Optional

import yaml


@dataclass
class Stage:
    name: str
    yaml_stem: str
    cpu_arch: str
    split_id: int
    total_splits: int
    mako: dict[str, str] = field(default_factory=dict)


@dataclass
class Block:
    yaml_stem: str
    block_index: int
    condition: dict
    tests: list[str]


# Strip trailing ` SKIP ...` / ` TIMEOUT ...` annotations from a test id.
# YAML entries can carry `TIMEOUT (n)`, waives.txt entries can carry both;
# the lookup must hit either form so we normalize on both sides.
_TEST_ID_SUFFIX_RE = re.compile(r"\s+(SKIP|TIMEOUT)\b.*$")

# Strip leading `full:<gpu>/` platform prefix used in waives.txt.
_TEST_ID_PREFIX_RE = re.compile(r"^full:[^/]+/")


def normalize_test_id(test_id: str) -> str:
    """Canonical form for cross-referencing test-db YAML and waives.txt.

    Strips trailing `SKIP`/`TIMEOUT` annotations, trailing `# comment`, and
    leading `full:<gpu>/` prefix. `YAMLIndex` indexes both the raw and the
    normalized form; `rules.waives_rule` looks up by this normalization.
    """
    s = test_id.strip()
    s = _TEST_ID_SUFFIX_RE.sub("", s).strip()
    if "#" in s:
        s = s.split("#", 1)[0].strip()
    return _TEST_ID_PREFIX_RE.sub("", s)


class YAMLIndex:
    """Index of all blocks across test-db YAMLs, with reverse lookup by test id."""

    def __init__(self) -> None:
        self.blocks: list[Block] = []
        self._test_to_blocks: dict[str, list[Block]] = {}

    @classmethod
    def load(cls, test_db_dir: Path) -> "YAMLIndex":
        idx = cls()
        for yml_path in sorted(test_db_dir.glob("l0_*.yml")):
            idx._load_one(yml_path)
        return idx

    def _load_one(self, yml_path: Path) -> None:
        yaml_stem = yml_path.stem
        data = yaml.safe_load(yml_path.read_text()) or {}
        blocks_data = data.get(yaml_stem, []) or []
        for i, block_data in enumerate(blocks_data):
            if not isinstance(block_data, dict):
                continue
            tests = block_data.get("tests") or []
            block = Block(
                yaml_stem=yaml_stem,
                block_index=i,
                condition=block_data.get("condition") or {},
                tests=list(tests),
            )
            self.blocks.append(block)
            # Index each test under both its raw YAML string (which may carry
            # ` -m "gpu2"`, ` TIMEOUT (90)`, etc.) and its normalized form, so
            # waives.txt lookups — which strip SKIP/TIMEOUT — still resolve.
            for test in tests:
                self._test_to_blocks.setdefault(test, []).append(block)
                normalized = normalize_test_id(test)
                if normalized and normalized != test:
                    self._test_to_blocks.setdefault(normalized, []).append(block)

    def blocks_containing_test(self, test_id: str) -> list[Block]:
        return list(self._test_to_blocks.get(test_id, []))

    def all_test_ids(self) -> Iterable[str]:
        return self._test_to_blocks.keys()


def _range_in(val_str: str | None, gte, lte) -> bool:
    if val_str is None:
        return False
    try:
        val = int(val_str)
    except (ValueError, TypeError):
        return False
    if gte is not None and val < gte:
        return False
    if lte is not None and val > lte:
        return False
    return True


# ---------------------------------------------------------------------------
# Stage config parsing from jenkins/L0_Test.groovy
# ---------------------------------------------------------------------------

# Matches a single entry like:
#   "A10-PyTorch-1": ["a10", "l0_a10", 1, 2],
#   "DGX_H100-4_GPUs-CPP-1": ["dgx-h100-x4", "l0_dgx_h100", 1, 1, 4],
# Same shape as scripts/test_to_stage_mapping.py::_STAGE_RE, extended to
# capture split_id / total_splits / gpu_count.
_STAGE_ENTRY_RE = re.compile(
    r'"(?P<stage>[^"]+)"\s*:\s*\['
    r'\s*"(?P<platform>[^"]+)"\s*,'
    r'\s*"(?P<yml>[^"]+)"'
    r"(?:\s*,\s*(?P<split_id>\d+))?"
    r"(?:\s*,\s*(?P<total_splits>\d+))?"
    r"(?:\s*,\s*(?P<gpu_count>\d+))?"
    r"\s*\]"
)

# Detects assignments opening a map literal, e.g. `x86TestConfigs = [`.
# Used to track which cpu_arch bucket the stage entries below belong to.
_MAP_OPEN_RE = re.compile(r"\b(?P<var>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*\[\s*$")


def _classify_map_var(var_name: str) -> Optional[str]:
    """Map a Groovy variable name to cpu_arch bucket, or None if unknown."""
    v = var_name.lower()
    if "sbsa" in v or "aarch64" in v:
        return "sbsa"
    if "x86" in v:
        return "x86"
    return None


# Backend name -> mako value. Same patterns as getMakoArgsFromStageName in
# jenkins/L0_Test.groovy (line ~2079). IMPORTANT: keep this list in sync.
_BACKEND_PATTERNS = [
    ("-PyTorch-", "pytorch"),
    ("-TensorRT-", "tensorrt"),
    ("-CPP-", "cpp"),
    ("-Triton-", "triton"),
    ("-FMHA-", "fmha"),
    ("-AutoDeploy-", "autodeploy"),
    ("-Verl-", "verl"),
]

# Regex mirror of parseTaskConfigFromStageName (jenkins/L0_Test.groovy:2066):
#   ([^-]+)(?:-(\d+)_GPUs)?
_GPU_NAME_RE = re.compile(r"^([^-]+)")
_GPU_COUNT_RE = re.compile(r"-(\d+)_GPUs")


def derive_mako_from_stage(stage_name: str) -> dict[str, str]:
    """Derive a stage's mako dict purely from its name.

    KEEP IN SYNC with the Groovy source of truth:
    - jenkins/L0_Test.groovy::getMakoArgsFromStageName (~line 2079)
    - jenkins/L0_Test.groovy::parseTaskConfigFromStageName (~line 2066)

    Fields produced (all values are strings, matching the Groovy behavior):
      stage, backend (optional), auto_trigger, orchestrator, gpu, system_gpu_count

    Runtime sysinfo keys like `linux_distribution_name` are NOT included here;
    they're only available once a stage executes. `block_matches_stage`
    treats missing keys as "assume match" so we over-include safely.
    """
    mako: dict[str, str] = {}
    mako["stage"] = "post_merge" if "Post-Merge" in stage_name else "pre_merge"

    for pat, val in _BACKEND_PATTERNS:
        if pat in stage_name:
            mako["backend"] = val
            break
    # else: no 'backend' key set -> matches any block backend term

    if "-DeepSeek-" in stage_name:
        mako["auto_trigger"] = "deepseek"
    elif "-GptOss-" in stage_name:
        mako["auto_trigger"] = "gpt_oss"
    else:
        mako["auto_trigger"] = "others"

    mako["orchestrator"] = "ray" if "-Ray-" in stage_name else "mpi"

    gpu_match = _GPU_NAME_RE.match(stage_name)
    if gpu_match:
        # Lowercased so YAML wildcards like `*a10*` / `*h100*` match.
        mako["gpu"] = gpu_match.group(1).lower()
    count_match = _GPU_COUNT_RE.search(stage_name)
    mako["system_gpu_count"] = count_match.group(1) if count_match else "1"

    return mako


def parse_stages_from_groovy(
    groovy_path: Path, include_post_merge: bool = False
) -> dict[str, Stage]:
    """Parse `jenkins/L0_Test.groovy` and return {stage_name -> Stage}.

    - cpu_arch is determined by which map literal (x86TestConfigs,
      SBSATestConfigs, ...) the entry lives in; falls back to stage-name
      heuristic if the map var is unfamiliar.
    - mako is derived via `derive_mako_from_stage` (pure stage-name logic).
    - Post-Merge stages are excluded by default (CBTS runs in pre-merge CI).
    """
    stages: dict[str, Stage] = {}
    current_arch: Optional[str] = None

    for line in groovy_path.read_text().splitlines():
        open_match = _MAP_OPEN_RE.search(line.rstrip())
        if open_match:
            # Reset on every map opening — including unfamiliar ones — so a
            # later map between classified sections can't inherit a stale
            # arch. Unknown maps fall back to the stage-name heuristic below.
            current_arch = _classify_map_var(open_match.group("var"))

        m = _STAGE_ENTRY_RE.search(line)
        if not m:
            continue
        stage_name = m.group("stage")
        if not include_post_merge and "Post-Merge" in stage_name:
            continue

        # Fallback heuristic if we haven't seen an obvious map-var yet.
        arch = current_arch
        if arch is None:
            arch = "sbsa" if ("GH200" in stage_name or "SBSA" in stage_name) else "x86"

        stages[stage_name] = Stage(
            name=stage_name,
            yaml_stem=m.group("yml"),
            cpu_arch=arch,
            split_id=int(m.group("split_id") or 1),
            total_splits=int(m.group("total_splits") or 1),
            mako=derive_mako_from_stage(stage_name),
        )
    return stages


# ---------------------------------------------------------------------------
# Block <-> Stage condition matching
# ---------------------------------------------------------------------------


def block_matches_stage(block: Block, stage: Stage) -> bool:
    """Return True iff the stage's mako satisfies the block's condition.

    Matching semantics (mirroring trt-test-db) — generic over field names:
    - terms[K]: stage.mako[K] must equal block.terms[K].
    - ranges[K]: int(stage.mako[K]) must lie in [gte, lte] (either bound optional).
    - wildcards[K]: stage.mako[K] must fnmatch at least one of the patterns.

    When a key referenced by the block is NOT present in stage.mako, we treat
    it as "unknown -> assume match". The mako we receive is derived from the
    stage name (stage/backend/auto_trigger/orchestrator/gpu/system_gpu_count)
    and doesn't include runtime sysinfo fields like `linux_distribution_name`
    that trt-test-db adds at execution time. Over-including in this case is
    safe (might launch one extra stage); under-including would mean silently
    skipping a test that should have run.
    """
    cond = block.condition
    if not isinstance(cond, dict):
        return False
    mako = stage.mako or {}

    terms = cond.get("terms") or {}
    for k, v in terms.items():
        if k not in mako:
            continue  # unknown key -> assume match
        if str(mako.get(k)) != str(v):
            return False

    ranges = cond.get("ranges") or {}
    for k, r in ranges.items():
        if not isinstance(r, dict):
            return False
        if k not in mako:
            continue  # unknown key -> assume match
        if not _range_in(mako.get(k), r.get("gte"), r.get("lte")):
            return False

    wildcards = cond.get("wildcards") or {}
    for k, patterns in wildcards.items():
        if k not in mako:
            continue  # unknown key -> assume match
        val = str(mako.get(k)).lower()
        if isinstance(patterns, str):
            patterns = [patterns]
        # Case-insensitive match — trt-test-db accepts uppercase mako values
        # (e.g. gpu="A10") against lowercase wildcards (e.g. "*a10*").
        if not any(fnmatch(val, str(p).lower()) for p in patterns):
            return False

    return True
