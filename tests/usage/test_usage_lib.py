# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import time
from pathlib import Path
from unittest.mock import MagicMock, patch

from tensorrt_llm.usage import is_usage_stats_enabled, report_usage
from tensorrt_llm.usage.usage_lib import UsageMessage, collect_system_info, detect_cloud_provider


class TestOptOut:
    """Test opt-out mechanisms."""

    def test_opt_out_via_env_var(self, monkeypatch):
        """Test opt-out via TRTLLM_NO_USAGE_STATS environment variable."""
        monkeypatch.setenv("TRTLLM_NO_USAGE_STATS", "1")
        assert not is_usage_stats_enabled()

    def test_opt_out_via_file(self, tmp_path, monkeypatch):
        """Test opt-out via config file."""
        # Mock OPT_OUT_FILE to use tmp_path
        opt_out_file = tmp_path / "do_not_track"
        opt_out_file.touch()

        with patch("tensorrt_llm.usage.usage_lib.OPT_OUT_FILE", opt_out_file):
            assert not is_usage_stats_enabled()

    def test_enabled_by_default(self, monkeypatch):
        """Test that usage stats are enabled by default."""
        # Clear opt-out environment variable
        monkeypatch.delenv("TRTLLM_NO_USAGE_STATS", raising=False)

        # Mock OPT_OUT_FILE to not exist
        with patch(
            "tensorrt_llm.usage.usage_lib.OPT_OUT_FILE", Path("/nonexistent/path/do_not_track")
        ):
            assert is_usage_stats_enabled()


class TestSystemInfo:
    """Test system information collection."""

    def test_collect_system_info(self):
        """Test system info collection."""
        info = collect_system_info()

        # Check that basic system info is present
        assert "platform" in info
        assert "python_version" in info
        assert "machine" in info
        assert "cpu_count" in info

        # GPU info may or may not be present depending on hardware
        # Just ensure no exceptions were raised


class TestCloudDetection:
    """Test cloud provider detection."""

    def test_detect_cloud_provider_timeout(self):
        """Test that cloud detection handles timeouts gracefully."""
        # This should return "unknown" quickly since we're not on a cloud instance
        provider = detect_cloud_provider()
        assert provider == "unknown"

    @patch("tensorrt_llm.usage.usage_lib.requests.get")
    def test_detect_aws(self, mock_get):
        """Test AWS detection."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        provider = detect_cloud_provider()
        assert provider == "aws"


class TestUsageMessage:
    """Test UsageMessage class."""

    def test_usage_message_initialization(self):
        """Test UsageMessage initialization."""
        message = UsageMessage()
        assert message.uuid is not None
        assert len(message.uuid) > 0

    def test_collect_heartbeat_data(self):
        """Test heartbeat data collection."""
        message = UsageMessage()
        heartbeat = message.collect_heartbeat_data()

        assert "uuid" in heartbeat
        assert "timestamp_ns" in heartbeat
        assert heartbeat["uuid"] == message.uuid

    def test_extract_trtllm_config(self):
        """Test TensorRT-LLM config extraction."""
        # Create a mock llm_args object
        mock_args = MagicMock()
        mock_args.backend = "tensorrt"
        mock_args.tensor_parallel_size = 2
        mock_args.pipeline_parallel_size = 1
        mock_args.context_parallel_size = 1
        mock_args.dtype = "float16"

        message = UsageMessage()
        config = message._extract_trtllm_config(mock_args)

        assert config["backend"] == "tensorrt"
        assert config["tensor_parallel_size"] == 2
        assert config["pipeline_parallel_size"] == 1
        assert config["context_parallel_size"] == 1
        assert config["dtype"] == "float16"

    def test_extract_model_info(self):
        """Test model info extraction."""
        # Create a mock pretrained_config object
        mock_config = MagicMock()
        mock_config.architectures = ["LlamaForCausalLM"]
        mock_config.model_type = "llama"
        mock_config.num_hidden_layers = 32
        mock_config.hidden_size = 4096
        mock_config.num_attention_heads = 32

        message = UsageMessage()
        model_info = message._extract_model_info(mock_config)

        assert model_info["architecture"] == "LlamaForCausalLM"
        assert model_info["model_type"] == "llama"
        assert model_info["num_layers"] == 32
        assert model_info["hidden_size"] == 4096
        assert model_info["num_attention_heads"] == 32

    def test_extract_model_info_none(self):
        """Test model info extraction with None config."""
        message = UsageMessage()
        model_info = message._extract_model_info(None)

        assert model_info["architecture"] == "unknown"

    def test_extract_model_info_architectures_only(self):
        """Test model info extraction with only architectures attribute."""
        mock_config = MagicMock()
        mock_config.architectures = ["Qwen2ForCausalLM", "GPT2LMHeadModel"]

        message = UsageMessage()
        model_info = message._extract_model_info(mock_config)

        # Should use the first architecture in the list
        assert model_info["architecture"] == "Qwen2ForCausalLM"

    def test_extract_model_info_empty_architectures(self):
        """Test model info extraction with empty architectures list."""
        mock_config = MagicMock()
        mock_config.architectures = []

        message = UsageMessage()
        model_info = message._extract_model_info(mock_config)

        assert model_info["architecture"] == "unknown"


class TestReportUsage:
    """Test report_usage function."""

    def test_report_usage_respects_opt_out(self, monkeypatch, tmp_path):
        """Test that reporting respects opt-out."""
        monkeypatch.setenv("TRTLLM_NO_USAGE_STATS", "1")

        # Create mock args
        mock_args = MagicMock()
        mock_args.backend = "tensorrt"

        # Should return immediately without starting thread
        report_usage(mock_args, pretrained_config=None)

        # Give time for any thread to start (shouldn't)
        time.sleep(0.2)

        # Thread shouldn't have started - test passes if no exceptions raised

    @patch("tensorrt_llm.usage.usage_lib.UsageMessage.report_once")
    def test_report_usage_starts_thread(self, mock_report_once, monkeypatch):
        """Test that report_usage starts a background thread."""
        # Ensure stats are enabled
        monkeypatch.delenv("TRTLLM_NO_USAGE_STATS", raising=False)

        with patch(
            "tensorrt_llm.usage.usage_lib.OPT_OUT_FILE", Path("/nonexistent/path/do_not_track")
        ):
            # Create mock args with required attributes
            mock_args = MagicMock()
            mock_args.backend = "tensorrt"

            # Call report_usage
            report_usage(mock_args, pretrained_config=None)

            # Give thread time to start and call report_once
            time.sleep(0.5)

            # Verify that report_once was called (initial report)
            assert mock_report_once.called
