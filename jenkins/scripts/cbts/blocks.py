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


# Detect a pytest-style option flag (e.g. ` -k`, ` -m`) inside a YAML target
# spec. Used to peel `-k "..."` / `-m "..."` off so the bare path/node-id can
# be indexed as an additional lookup key — that lets a fine-grained waive id
# match a coarse YAML entry like `dir/x.py -k "deepseek"`.
_PYTEST_OPTION_RE = re.compile(r"\s+-[a-zA-Z]\b")


def _strip_pytest_options(s: str) -> str:
    """Return `s` with any pytest option flag (and everything after it) removed.

    `dir/x.py -k "deepseek"`        -> `dir/x.py`
    `x.py::test_y -m "gpu1"`        -> `x.py::test_y`
    `x.py`                          -> `x.py`  (unchanged)
    """
    m = _PYTEST_OPTION_RE.search(s)
    return s[: m.start()].rstrip() if m else s


def _iter_parent_ids(test_id: str):
    """Yield ancestor forms of a pytest target id, most-specific to least.

    `dir/x.py::TestC::test_m[a-b]` yields:
        `dir/x.py::TestC::test_m`     (strip `[params]`)
        `dir/x.py::TestC`             (strip `::test_m`)
        `dir/x.py`                    (strip `::TestC`)
        `dir`                         (strip `/x.py`)

    Lets the lookup match coarser YAML entries (file, class, directory) when
    the waive id is finer-grained.
    """
    s = test_id
    if "[" in s:
        s = s.rsplit("[", 1)[0]
        if s:
            yield s
    while "::" in s:
        s = s.rsplit("::", 1)[0]
        if s:
            yield s
    while "/" in s:
        s = s.rsplit("/", 1)[0]
        if s:
            yield s


# pytest's `-k` keyword expressions can be a single word like `"deepseek"` or
# a boolean expression like `"a and not b"`. We tokenize on identifier chars
# and strip the boolean operators, then check substring against the waive id.
# Over-includes for complex expressions; never under-includes.
_K_VALUE_RE = re.compile(r'\s+-k\s+"([^"]*)"')
_M_FLAG_RE = re.compile(r"\s+-m\b")
_K_RESERVED_WORDS = {"and", "or", "not"}


def _extract_k_keyword(entry: str) -> Optional[str]:
    """Return the raw `-k "<value>"` keyword text, or None."""
    m = _K_VALUE_RE.search(entry)
    return m.group(1) if m else None


def _entry_applies_to_waive(entry: str, waive_id: str) -> bool:
    """Best-effort check: would this entry's pytest options run the waived test.

    Skips ancestor-level entries whose `-k` filter excludes the waive (e.g.
    `file.py -k "weights"` shouldn't match a `test_biases[a]` waive).

    `-k "<expr>"`: extract identifier tokens, drop boolean operators, accept
                   if any remaining token appears in waive id.
    `-m "<mark>"`: always accept (markers are runtime metadata, can't verify
                   from the test id string alone).
    No options:    always accept.
    """
    kw = _extract_k_keyword(entry)
    if kw is None:
        return True  # no -k → unconstrained
    tokens = re.findall(r"[A-Za-z_]\w*", kw)
    real = [t for t in tokens if t.lower() not in _K_RESERVED_WORDS]
    if not real:
        return True  # all-reserved expression → can't decide; over-include
    return any(t in waive_id for t in real)


def _strip_params(s: str) -> str:
    """Strip `[params]` suffix if present.

    `file.py::test_x[a]`  → `file.py::test_x`
    `file.py::test_x`     → `file.py::test_x`
    """
    return s.rsplit("[", 1)[0] if "[" in s else s


def _entry_target(entry: str) -> str:
    """Canonical "target" key for indexing/lookup.

    Strips SKIP/TIMEOUT/full:gpu, pytest options (-k/-m), and `[params]` suffix.

    `file.py::TestC::test_m[a-b] TIMEOUT (90)` → `file.py::TestC::test_m`
    `file.py -k "kw"`                          → `file.py`
    """
    return _strip_params(_strip_pytest_options(normalize_test_id(entry)))


def _target_in_filter_subtree(target: str, filter_prefix: str) -> bool:
    """True iff `target` is in `filter_prefix`'s subtree.

    `target` matches when it is `filter_prefix` itself or a descendant of it
    (params / method / file / dir component below) in the pytest tree.
    """
    if target == filter_prefix:
        return True
    return (
        target.startswith(filter_prefix + "[")
        or target.startswith(filter_prefix + "::")
        or target.startswith(filter_prefix + "/")
    )


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
            # Index each test under up to four keys so a waive id can match
            # YAML entries written at any granularity:
            #   1. raw YAML string     (with TIMEOUT/markers as written)
            #   2. normalized form     (SKIP/TIMEOUT/full:gpu prefix stripped)
            #   3. target-w/-options   (pytest options `-k "..."` stripped)
            #   4. canonical target    (also `[params]` stripped — the level
            #                           the lookup walks against)
            # Waive-side lookup walks the pytest node-id parent chain. A
            # coarse YAML entry (file / dir / `dir/x.py -k "kw"`) and a fine
            # waive (`dir/x.py::TestC::test_m[params]`) meet in the middle
            # via key #4.
            for test in tests:
                seen: set[str] = set()
                norm = normalize_test_id(test)
                for key in (
                    test,
                    norm,
                    _strip_pytest_options(norm),
                    _strip_params(_strip_pytest_options(norm)),
                ):
                    if key and key not in seen:
                        seen.add(key)
                        self._test_to_blocks.setdefault(key, []).append(block)

    def find_match_for_waive(self, waive_id: str) -> Optional[tuple[str, list[Block]]]:
        """Walk up the pytest parent chain; first level with a match wins.

        Returns (level, blocks) for the first level whose YAML index has at
        least one matching entry, or None when even the root level misses.

        Detail:
            (level, blocks): `level` is the YAML target string where the match
                            was found; `blocks` are the affected blocks at
                            that level.
            None:            no level matched all the way to the root —
                            caller should treat as fallback (CBTS exits,
                            baseline runs).

        An entry "matches" at a level when its target (after stripping pytest
        options) equals the level AND its `-k` filter (if any) actually applies
        to the waive id.
        """
        # Step 1+2: normalize the waive id, strip [params] to get the start.
        #   `file.py::TestC::test_m[a-b]` → `file.py::TestC::test_m`
        #   `file.py::TestC::test_m`      → unchanged (already at function level)
        #   `file.py::TestC`              → unchanged (class)
        #   `file.py`                     → unchanged (file)
        target = _strip_pytest_options(normalize_test_id(waive_id))
        if "[" in target:
            target = target.rsplit("[", 1)[0]

        # Step 3: walk up until something matches.
        for level in [target, *_iter_parent_ids(target)]:
            candidates = self._test_to_blocks.get(level, [])
            matched: list[Block] = []
            seen_keys: set[tuple[str, int]] = set()
            for block in candidates:
                key = (block.yaml_stem, block.block_index)
                if key in seen_keys:
                    continue
                # Verify at least one entry in this block has canonical
                # target == level AND its -k constraint (if any) applies to
                # the waive. Canonical = strip pytest options and [params].
                for raw in block.tests:
                    if _entry_target(raw) == level and _entry_applies_to_waive(raw, waive_id):
                        matched.append(block)
                        seen_keys.add(key)
                        break
            if matched:
                return level, matched
        return None

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
#   "DGX_B200-PyTorch-1": ["auto:dgx-b200-flex", "l0_b200", 1, 3, 1, 1, true],
# Same shape as scripts/test_to_stage_mapping.py::_STAGE_RE, extended to
# capture split_id / total_splits / gpu_count and tolerate any number of
# trailing positional args (e.g. SLURM configs append a boolean flag).
_STAGE_ENTRY_RE = re.compile(
    r'"(?P<stage>[^"]+)"\s*:\s*\['
    r'\s*"(?P<platform>[^"]+)"\s*,'
    r'\s*"(?P<yml>[^"]+)"'
    r"(?:\s*,\s*(?P<split_id>\d+))?"
    r"(?:\s*,\s*(?P<total_splits>\d+))?"
    r"(?:\s*,\s*(?P<gpu_count>\d+))?"
    r"(?:\s*,[^\]]*)?"  # tolerate trailing positional args before `]`
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


# ---------------------------------------------------------------------------
# CBTS Layer 3 split-count heuristic: per-stage narrowed test count
# ---------------------------------------------------------------------------


def compute_stage_test_counts(
    yaml_index: "YAMLIndex",
    stages: dict[str, "Stage"],
    affected_stages: set[str],
    block_filters: dict[tuple[str, int], dict[str, set[str]]],
) -> dict[str, int]:
    """Sum the kept-test count per affected stage across matching blocks.

    Used by Groovy launchTestJobs to decide whether to collapse the stage's
    pytest-split splits to 1 (when narrowed_count < 20). The keep filter
    here mirrors the one in `write_filtered_test_db` exactly so the count
    matches what trt-test-db will eventually render.
    """
    block_by_key: dict[tuple[str, int], Block] = {
        (b.yaml_stem, b.block_index): b for b in yaml_index.blocks
    }
    counts: dict[str, int] = {}
    for stage_name in affected_stages:
        stage = stages.get(stage_name)
        if stage is None:
            continue
        total = 0
        for (yaml_stem, idx), prefix_to_waives in block_filters.items():
            if yaml_stem != stage.yaml_stem:
                continue
            block = block_by_key.get((yaml_stem, idx))
            if block is None or not block_matches_stage(block, stage):
                continue
            kept: list[str] = []
            for t in block.tests:
                target = _entry_target(t)
                matched_waives: set[str] = set()
                for prefix, waives in prefix_to_waives.items():
                    if _target_in_filter_subtree(target, prefix):
                        matched_waives |= waives
                if not matched_waives:
                    continue
                if any(_entry_applies_to_waive(t, w) for w in matched_waives):
                    kept.append(t)
            # Safety fallback mirrors write_filtered_test_db: when the
            # narrowing would empty the block, the original tests stay.
            total += len(kept) if kept else len(block.tests)
        counts[stage_name] = total
    return counts


# ---------------------------------------------------------------------------
# CBTS Layer 3: filtered test-db YAML generation
# ---------------------------------------------------------------------------


def write_filtered_test_db(
    src_dir: Path,
    output_dir: Path,
    block_filters: dict[tuple[str, int], dict[str, set[str]]],
) -> None:
    """Generate a tmp test-db dir narrowed by CBTS Layer 3.

    Each output YAML contains ONLY the blocks listed in `block_filters` for
    that stem, with each affected block's `tests:` array filtered to entries
    in the per-block filter prefix subtree. Unaffected blocks and unaffected
    YAML files are dropped entirely.

    Two narrowing checks per entry:
      1. Subtree match — entry's target lives under at least one filter prefix
      2. -k keyword guard — if the entry uses `func -k "K"`, drop it unless
         `K` would actually run at least one of the waives that resolved to
         the matching prefix(es). Without this guard, a parent-chain fallback
         (e.g. waive `func[CUTLASS-fp8-tp4]` -> prefix `func`) over-includes
         every `-k` variant of `func` even though only `-k "CUTLASS"` would
         pick up the waived test at runtime.

    Why drop unaffected blocks rather than write them through unchanged:
    Layer 3's contract is "only run tests touched by the affected blocks". If
    we kept post_merge / other-backend blocks that this PR never touched, a
    `/bot run --post-merge` could activate them on stages whose mako happens
    to match — running tests CBTS never selected. Dropping them keeps Layer
    3's narrowing semantically tight.

    `block_filters` keys: (yaml_stem, block_index) of affected blocks.
    Values: {filter_prefix: {waive_id, ...}} — prefix governs subtree match;
    waive ids are consulted by the -k keyword guard above.

    Safety: if filtering would empty an affected block's tests, the original
    tests are kept (prevents silent skip from typo'd waive ids or granularity
    mismatch). The block itself is still kept either way.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    affected_stems = {stem for stem, _ in block_filters}
    for stem in sorted(affected_stems):
        src = src_dir / f"{stem}.yml"
        if not src.exists():
            continue
        data = yaml.safe_load(src.read_text()) or {}

        for ctx_key, ctx_blocks in list(data.items()):
            if not isinstance(ctx_blocks, list):
                continue
            new_blocks = []
            for i, block_data in enumerate(ctx_blocks):
                if not isinstance(block_data, dict):
                    continue
                key = (stem, i)
                if key not in block_filters:
                    # Drop unaffected blocks (see docstring rationale).
                    continue
                original = block_data.get("tests") or []
                prefix_to_waives = block_filters[key]
                kept = []
                for t in original:
                    target = _entry_target(t)
                    matched_waives: set[str] = set()
                    for prefix, waives in prefix_to_waives.items():
                        if _target_in_filter_subtree(target, prefix):
                            matched_waives |= waives
                    if not matched_waives:
                        continue
                    # -k keyword guard: drop entries whose `-k` filter
                    # excludes every matching waive. Entries without `-k`
                    # are always kept (`_entry_applies_to_waive` returns
                    # True when no keyword is present).
                    if any(_entry_applies_to_waive(t, w) for w in matched_waives):
                        kept.append(t)
                # Safety: empty filter result → fallback to original (prevents
                # silent skip from typo'd waive ids or granularity mismatch).
                if kept:
                    block_data["tests"] = kept
                new_blocks.append(block_data)
            data[ctx_key] = new_blocks

        (output_dir / src.name).write_text(
            yaml.safe_dump(data, sort_keys=False, default_flow_style=False)
        )
