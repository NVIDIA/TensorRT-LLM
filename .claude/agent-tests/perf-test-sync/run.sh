#!/usr/bin/env bash
# Run promptfoo tests for perf-test-sync agent
# Usage: ./run.sh <ANTHROPIC_API_KEY> [ANTHROPIC_BASE_URL]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <ANTHROPIC_API_KEY> [ANTHROPIC_BASE_URL]"
    exit 1
fi

export ANTHROPIC_API_KEY="$1"
export OPENAI_API_KEY="$1"
if [[ $# -ge 2 ]]; then
    export ANTHROPIC_BASE_URL="$2"
    export OPENAI_BASE_URL="$2/v1"
fi

cd "$SCRIPT_DIR"
RESULTS_JSON="$SCRIPT_DIR/results.json"
REPORT="$SCRIPT_DIR/report.html"
set +e
npx promptfoo eval -c promptfooconfig.yaml --output "$RESULTS_JSON"
EVAL_RC=$?
set -e

if [[ -f "$RESULTS_JSON" ]]; then
    python3 "$SCRIPT_DIR/render_report.py" "$RESULTS_JSON" "$REPORT"
else
    echo "WARNING: $RESULTS_JSON not found; skipping render_report.py"
fi

if [[ $EVAL_RC -ne 0 ]]; then
    echo "NOTE: promptfoo eval exited with status $EVAL_RC (report was still rendered if results.json existed)"
fi

echo
echo "HTML report: $REPORT"
echo "Interactive viewer: npx promptfoo view  (opens http://localhost:15500)"
