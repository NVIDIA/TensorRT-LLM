# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Extract bounded candidate lines for shared-library failure matching."""

import argparse
import json
from pathlib import Path

MAX_CANDIDATE_CHARS = 4096


def _query_terms(queries: list[dict]) -> list[str]:
    terms = []
    seen = set()
    for query in queries:
        query_terms = query.get("anyOf") or query.get("allOf") or []
        for term in query_terms:
            folded = term.casefold()
            if folded not in seen:
                seen.add(folded)
                terms.append(term)
    return terms


def _match_context(line: str, term: str) -> str:
    match_index = line.casefold().find(term.casefold())
    start = max(0, match_index - MAX_CANDIDATE_CHARS // 2)
    end = min(len(line), start + MAX_CANDIDATE_CHARS)
    if end - start < MAX_CANDIDATE_CHARS:
        start = max(0, end - MAX_CANDIDATE_CHARS)
    return line[start:end].strip().replace("\x00", "\ufffd")


def extract_candidates(log_path: Path, queries: list[dict]) -> list[str]:
    terms = _query_terms(queries)
    folded_terms = [(term, term.casefold()) for term in terms]
    matches = {}
    with log_path.open(encoding="utf-8", errors="replace") as log_file:
        for line in log_file:
            folded_line = line.casefold()
            for term, folded_term in folded_terms:
                if folded_term in folded_line:
                    matches[folded_term] = _match_context(line, term)

    candidates = []
    seen = set()
    for term in terms:
        candidate = matches.get(term.casefold())
        if candidate and candidate not in seen:
            seen.add(candidate)
            candidates.append(candidate)
    return candidates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("--queries", required=True, type=Path)
    args = parser.parse_args()

    queries = json.loads(args.queries.read_text(encoding="utf-8"))
    print("\n".join(extract_candidates(args.log, queries)))


if __name__ == "__main__":
    main()
