import argparse
import json
import sys

from utils.es import post_preapproved_deps

parser = argparse.ArgumentParser(
    description="Post manually approved license candidates to the preapproved index."
)
parser.add_argument(
    "candidates_file",
    help="JSON file containing preapproved candidates produced by main.py",
)
args = parser.parse_args()

with open(args.candidates_file) as f:
    candidates = json.load(f)

if not candidates:
    print("No candidates to index.")
    sys.exit(0)

# Convert candidate format {scan_type, package_name, ...} to the risk-doc format
# expected by post_preapproved_deps ({s_type, s_package_name, ...}).
risk_docs = [
    {
        "s_type": c["scan_type"],
        "s_package_name": c["package_name"],
        "s_package_version": c.get("package_version"),
        "s_package_type": c.get("package_type").lower(),
    }
    for c in candidates
]

ok = post_preapproved_deps(risk_docs)
if not ok:
    print("One or more preapproved candidates failed to index.", file=sys.stderr)
    sys.exit(1)

print(f"Successfully indexed {len(risk_docs)} preapproved candidate(s).")
