#!/bin/bash

RUNNER_REPO=$RUNNER_REPO
RUNNER_PAT=$RUNNER_PAT
RUNNER_GROUP=$RUNNER_GROUP
RUNNER_LABELS=$RUNNER_LABELS
RUNNER_NAME=$(hostname)

cd /home/runner/actions-runner

./config.sh --unattended --replace --url https://github.com/${RUNNER_REPO} --pat ${RUNNER_PAT} --name ${RUNNER_NAME} --runnergroup ${RUNNER_GROUP} --labels ${RUNNER_LABELS} --work /home/runner/actions-runner/_work

cleanup() {
    echo "Removing runner..."
    ./config.sh remove --unattended --pat ${RUNNER_PAT}
}

trap 'cleanup; exit 130' INT
trap 'cleanup; exit 143' TERM

./run.sh & wait $!