#!/bin/bash
set -ex

cd /code/

function serve {
    export UCX_UD_TIMEOUT=120s
    export PMIX_MCA_gds=hash # Required

    /opt/tritonserver/bin/tritonserver --model-repo llmapi_repo
}

# task
nvidia-smi
serve
