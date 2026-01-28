#!/bin/bash
# cleanup_jobs.sh - Cancel all SLURM jobs tracked in jobs.txt
#
# This script is designed to run in GitLab CI after_script to ensure
# all SLURM jobs are cancelled when the pipeline is interrupted, cancelled,
# or times out.
#
# Usage:
#   bash cleanup_jobs.sh
#
# Environment variables:
#   OUTPUT_PATH: Directory containing jobs.txt and pytest.pid

set -e

OUTPUT_PATH="${OUTPUT_PATH:-/tmp}"
JOBS_FILE="${OUTPUT_PATH}/jobs.txt"
PID_FILE="${OUTPUT_PATH}/pytest.pid"

echo "=========================================="
echo "SLURM Job Cleanup Script"
echo "=========================================="
echo "Output path: $OUTPUT_PATH"
echo ""

# Terminate pytest process if still running
if [ -f "$PID_FILE" ]; then
    PYTEST_PID=$(cat "$PID_FILE" | tr -d '\n')
    echo "Pytest PID: $PYTEST_PID"

    # Check if pytest is still running and kill it
    if kill -0 "$PYTEST_PID" 2>/dev/null; then
        echo "Status: Still running - terminating..."
        if kill -9 "$PYTEST_PID" 2>/dev/null; then
            echo "       [OK] Process killed"
        else
            echo "       [WARN] Failed to kill process (may already be gone)"
        fi
    else
        echo "Status: Already terminated"
    fi
    echo ""
else
    echo "No pytest.pid found (test may not have started)"
    echo ""
fi

# Check if jobs.txt exists
if [ ! -f "$JOBS_FILE" ]; then
    echo "[WARN] No jobs.txt found"
    echo "       Nothing to cancel"
    echo "=========================================="
    exit 0
fi

echo "[INFO] Reading jobs from: $JOBS_FILE"

# Read, deduplicate, and filter empty lines
JOBS=$(sort -u "$JOBS_FILE" | grep -v '^$' || true)

if [ -z "$JOBS" ]; then
    echo "[WARN] jobs.txt is empty"
    echo "       Nothing to cancel"
    echo "=========================================="
    exit 0
fi

JOB_COUNT=$(echo "$JOBS" | wc -l)
echo "Found $JOB_COUNT job(s) to cancel"
echo ""

# Cancel each job
CANCELLED=0
ALREADY_DONE=0
FAILED=0

echo "Cancelling jobs..."
while IFS= read -r job_id; do
    if [ -n "$job_id" ]; then
        printf "  %-12s ... " "$job_id"

        # Try to cancel the job
        if scancel "$job_id" 2>/dev/null; then
            echo "[OK] Cancelled"
            CANCELLED=$((CANCELLED + 1))
        else
            # Check if job exists in squeue
            if squeue -j "$job_id" -h 2>/dev/null | grep -q "$job_id"; then
                echo "[FAIL] Failed to cancel"
                FAILED=$((FAILED + 1))
            else
                echo "[SKIP] Already finished"
                ALREADY_DONE=$((ALREADY_DONE + 1))
            fi
        fi
    fi
done <<< "$JOBS"

echo ""
echo "=========================================="
echo "[DONE] Cleanup completed"
echo "       Total:           $JOB_COUNT"
echo "       Cancelled:       $CANCELLED"
echo "       Already done:    $ALREADY_DONE"
echo "       Failed:          $FAILED"
echo "=========================================="

# Exit with error if any cancellation actually failed
if [ $FAILED -gt 0 ]; then
    exit 1
fi

exit 0
