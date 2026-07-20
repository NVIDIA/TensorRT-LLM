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

"""Unit tests for perf-sanity disaggregated-server config generation."""

import unittest

try:
    from .disagg_server_config import build_disagg_server_config
except ImportError:
    # Support direct execution without importing the GPU-heavy integration package.
    from disagg_server_config import build_disagg_server_config


class TestBuildDisaggServerConfig(unittest.TestCase):
    def test_no_extra_preserves_baseline_config(self):
        config = build_disagg_server_config(
            hostname="benchmark-host",
            port=8123,
            num_ctx_servers=1,
            num_gen_servers=2,
            ctx_hostnames=["ctx-host:8000"],
            gen_hostnames=["gen-host-0:8001", "gen-host-1:8001"],
        )

        self.assertEqual(
            config,
            {
                "hostname": "benchmark-host",
                "port": 8123,
                "backend": "pytorch",
                "context_servers": {
                    "num_instances": 1,
                    "urls": ["ctx-host:8000"],
                },
                "generation_servers": {
                    "num_instances": 2,
                    "urls": ["gen-host-0:8001", "gen-host-1:8001"],
                },
            },
        )

    def test_applies_schedule_style_from_server_config_extra(self):
        config = build_disagg_server_config(
            hostname="benchmark-host",
            port=8123,
            num_ctx_servers=1,
            num_gen_servers=1,
            ctx_hostnames=["ctx-host:8000"],
            gen_hostnames=["gen-host:8001"],
            server_config_extra={"schedule_style": "generation_first"},
        )

        self.assertEqual(config["schedule_style"], "generation_first")
        self.assertEqual(config["context_servers"]["urls"], ["ctx-host:8000"])
        self.assertEqual(config["generation_servers"]["urls"], ["gen-host:8001"])

    def test_rejects_unknown_or_reserved_extra_keys(self):
        for extra in (
            {"unexpected": "value"},
            {"hostname": "redirected-host"},
            {"port": 9000},
            {"backend": "other"},
            {"context_servers": {}},
            {"generation_servers": {}},
        ):
            with (
                self.subTest(extra=extra),
                self.assertRaisesRegex(ValueError, "supports only 'schedule_style'"),
            ):
                build_disagg_server_config(
                    hostname="benchmark-host",
                    port=8123,
                    num_ctx_servers=1,
                    num_gen_servers=1,
                    ctx_hostnames=["ctx-host:8000"],
                    gen_hostnames=["gen-host:8001"],
                    server_config_extra=extra,
                )

    def test_rejects_invalid_schedule_style(self):
        with self.assertRaisesRegex(ValueError, "schedule_style must be one of"):
            build_disagg_server_config(
                hostname="benchmark-host",
                port=8123,
                num_ctx_servers=1,
                num_gen_servers=1,
                ctx_hostnames=["ctx-host:8000"],
                gen_hostnames=["gen-host:8001"],
                server_config_extra={"schedule_style": "unsupported"},
            )


if __name__ == "__main__":
    unittest.main()
