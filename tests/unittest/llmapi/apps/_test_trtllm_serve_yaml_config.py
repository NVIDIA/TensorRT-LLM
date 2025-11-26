import os
import tempfile

import pytest
import requests
import yaml

from ..test_llm import get_model_path
from .openai_server import RemoteOpenAIServer

pytestmark = pytest.mark.threadleak(enabled=False)


@pytest.fixture(scope="module", ids=["TinyLlama-1.1B-Chat"])
def model_name():
    return "llama-models-v2/TinyLlama-1.1B-Chat-v1.0"


# Expected config values from YAML
EXPECTED_MAX_NUM_TOKENS = 512
EXPECTED_MAX_BATCH_SIZE = 64
EXPECTED_MAX_SEQ_LEN = 2048


@pytest.fixture(scope="module")
def temp_extra_llm_api_options_file():
    """Create YAML config with top-level keys only (no CLI overrides)."""
    temp_dir = tempfile.gettempdir()
    temp_file_path = os.path.join(temp_dir, "yaml_config_test.yaml")
    try:
        # Top-level config - same as would be specified via CLI flags
        extra_llm_api_options_dict = {
            "max_num_tokens": EXPECTED_MAX_NUM_TOKENS,
            "max_batch_size": EXPECTED_MAX_BATCH_SIZE,
            "max_seq_len": EXPECTED_MAX_SEQ_LEN,
        }

        with open(temp_file_path, "w") as f:
            yaml.dump(extra_llm_api_options_dict, f)

        yield temp_file_path
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


@pytest.fixture(scope="module")
def server(model_name: str, temp_extra_llm_api_options_file: str):
    """Start server with YAML config only (no CLI overrides).

    Note: backend defaults to pytorch, so no need to specify.
    """
    model_path = get_model_path(model_name)
    args = [
        "--extra_llm_api_options",
        temp_extra_llm_api_options_file,
        # NOTE: No --max_num_tokens, --max_batch_size, --max_seq_len CLI flags!
        # These values MUST come from the YAML file.
    ]
    with RemoteOpenAIServer(model_path, cli_args=args) as remote_server:
        yield remote_server


def test_serve_yaml_top_level_config_applied(server: RemoteOpenAIServer):
    """
    Verify that top-level YAML config (max_num_tokens, max_batch_size, max_seq_len)
    is correctly applied to trtllm-serve.

    This test explicitly verifies the runtime config values match the YAML file,
    ensuring YAML config is the single source of truth without needing CLI flags.

    Related: nvbug 5692109
    """
    # Query the /config endpoint to get actual runtime values
    config_url = server.url_for("config")
    response = requests.get(config_url)

    assert response.status_code == 200, f"Failed to get /config: {response.text}"
    config = response.json()

    # Verify YAML values are actually being used
    assert config["max_num_tokens"] == EXPECTED_MAX_NUM_TOKENS, (
        f"max_num_tokens mismatch: expected {EXPECTED_MAX_NUM_TOKENS}, got {config['max_num_tokens']}"
    )

    assert config["max_batch_size"] == EXPECTED_MAX_BATCH_SIZE, (
        f"max_batch_size mismatch: expected {EXPECTED_MAX_BATCH_SIZE}, got {config['max_batch_size']}"
    )

    assert config["max_seq_len"] == EXPECTED_MAX_SEQ_LEN, (
        f"max_seq_len mismatch: expected {EXPECTED_MAX_SEQ_LEN}, got {config['max_seq_len']}"
    )
