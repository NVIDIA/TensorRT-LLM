#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A count_tokens probe corpus from recorded Messages requests (check C-14, rule USG-05).

Each recorded ``POST /v1/messages`` request becomes one ``POST
/v1/messages/count_tokens`` entry with the same prompt-shaping fields. Replay the
corpus against the same server into the same trace directory
(``replay_requests.py``); ``verdict.py`` then pairs each count with the input
the matching turn was billed, by content.

    count_tokens_corpus.py --out corpus.jsonl [--sample N] <request shards or dirs>

The corpus holds request bodies: keep it with the trace it came from (same
permissions), never in the repository.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "validate"))
from tracelib import expand_shards, iter_lines, load_json_line  # noqa: E402
from usage import COUNTED_FIELDS, counted_digest  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)  # fmt: skip
    ap.add_argument("--out", required=True)
    ap.add_argument("--sample", type=int, default=0, help="at most N distinct bodies (0 = all)")
    ap.add_argument("paths", nargs="+")
    args = ap.parse_args(argv)
    seen, written = set(), 0
    with open(args.out, "w") as out:
        os.chmod(args.out, 0o600)
        for path in expand_shards(args.paths, "requests"):
            for _, raw in iter_lines(path):
                rec = load_json_line(raw)
                route = str((rec or {}).get("route") or "").split("?", 1)[0]
                body = (rec or {}).get("body")
                if route != "/v1/messages" or not isinstance(body, dict):
                    continue
                digest = counted_digest(body)
                if digest in seen:
                    continue
                seen.add(digest)
                entry = {"corpus_id": "count-" + digest[:16], "route": "/v1/messages/count_tokens",
                         "body": {k: body[k] for k in COUNTED_FIELDS if body.get(k) is not None}}  # fmt: skip
                out.write(json.dumps(entry, ensure_ascii=False) + "\n")
                written += 1
                if args.sample and written >= args.sample:
                    break
            if args.sample and written >= args.sample:
                break
    print(json.dumps({"entries": written, "out": args.out}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
