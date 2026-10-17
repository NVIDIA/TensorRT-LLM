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
"""host_page_retention host policy: locating and setting mimalloc's purge delay in libc10.so.

CPU only. Tests that need PyTorch's statically linked mimalloc skip on builds
without it.
"""

import ctypes
import json
import os
import subprocess
import sys

import pytest

from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared import dispatch
from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.host_page_retention import (  # noqa: E501
    host_policy as hp,
)


def _options_or_skip():
    try:
        return hp.MimallocOptions(hp.torch_libc10_path())
    except hp.HostPolicyUnavailable as exc:
        pytest.skip(f"no mimalloc option table in this PyTorch build: {exc}")


def _run(code, env_extra):
    env = {k: v for k, v in os.environ.items() if not k.startswith("MIMALLOC_")}
    env.update(env_extra)
    out = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        check=True,
        capture_output=True,
        text=True,
        cwd="/tmp",
        timeout=300,
    )
    return json.loads(out.stdout.strip().splitlines()[-1])


_READ = """
import json, torch
from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.host_page_retention import host_policy as hp
opts = hp.MimallocOptions(hp.torch_libc10_path())
before = opts.get("purge_delay")
result = None
if {apply}:
    result = hp.apply_purge_policy()
print(json.dumps({{"before": before, "after": opts.get("purge_delay"), "result": result}}))
"""  # noqa: E501


def test_option_table_is_found_and_consistent():
    opts = _options_or_skip()
    assert opts.names.count(hp.PURGE_DELAY_OPTION) == 1
    assert len(opts.names) == len(set(opts.names))
    for name in opts.names:
        opts.get(name)  # mi_option_get agrees with the table for every option


def test_set_takes_effect_and_restores():
    opts = _options_or_skip()
    old = opts.get(hp.PURGE_DELAY_OPTION)
    try:
        opts.set(hp.PURGE_DELAY_OPTION, 1234)
        assert opts.get(hp.PURGE_DELAY_OPTION) == 1234
    finally:
        opts.set(hp.PURGE_DELAY_OPTION, old)
    assert opts.get(hp.PURGE_DELAY_OPTION) == old


def test_runtime_policy_equals_environment_setting():
    _options_or_skip()
    env_policy = _run(_READ.format(apply=False), {"MIMALLOC_PURGE_DELAY": str(hp.PURGE_DELAY_MS)})
    served = _run(_READ.format(apply=False), {"MIMALLOC_PURGE_DELAY": "0"})
    applied = _run(_READ.format(apply=True), {"MIMALLOC_PURGE_DELAY": "0"})
    assert env_policy["before"] == hp.PURGE_DELAY_MS
    assert served["before"] == 0 and served["after"] == 0
    assert applied["result"] == [0, hp.PURGE_DELAY_MS]
    assert applied["after"] == env_policy["before"]


def test_already_set_is_left_alone():
    _options_or_skip()
    out = _run(_READ.format(apply=True), {"MIMALLOC_PURGE_DELAY": str(hp.PURGE_DELAY_MS)})
    assert out["result"] == [hp.PURGE_DELAY_MS, hp.PURGE_DELAY_MS]


def test_bad_descriptor_table_is_rejected():
    opts = _options_or_skip()
    good = ctypes.string_at(opts._table, opts._count * hp._DESC.size)
    buf = ctypes.create_string_buffer(good, len(good))
    value, init, option, name, legacy = hp._DESC.unpack_from(buf.raw, hp._DESC.size)
    hp._DESC.pack_into(buf, hp._DESC.size, value, init, option + 7, name, legacy)
    fake = object.__new__(hp.MimallocOptions)
    fake._table = ctypes.addressof(buf)
    fake._count = opts._count
    with pytest.raises(hp.HostPolicyUnavailable, match="not a mimalloc option table"):
        fake._check_table()


@pytest.mark.parametrize("content", [b"", b"not an elf file at all" * 4])
def test_non_elf_library_is_rejected(tmp_path, content):
    path = tmp_path / "libc10.so"
    path.write_bytes(content)
    with pytest.raises(hp.HostPolicyUnavailable):
        hp.MimallocOptions(str(path))


def test_missing_library_is_rejected():
    with pytest.raises(hp.HostPolicyUnavailable, match="not found"):
        hp.MimallocOptions("/nonexistent/libc10.so")


def test_library_not_loaded_is_rejected(tmp_path):
    # A copy of libc10.so is an ELF file with the symbols but is not loaded
    # under that path: the policy must not dlopen a second copy.
    _options_or_skip()
    copy = tmp_path / "libc10.so"
    copy.write_bytes(open(hp.torch_libc10_path(), "rb").read())
    with pytest.raises(hp.HostPolicyUnavailable, match="not loaded"):
        hp.MimallocOptions(str(copy))


def test_install_declines_without_side_effect(monkeypatch):
    opts = _options_or_skip()
    old = opts.get(hp.PURGE_DELAY_OPTION)
    logged = []
    monkeypatch.setattr(hp, "torch_libc10_path", lambda: "/nonexistent/libc10.so")
    monkeypatch.setattr(dispatch, "_CAPABILITY_LOGGED", {})
    monkeypatch.setattr(dispatch, "_log_info", logged.append)
    assert hp.install_host_policy("test") is False
    assert opts.get(hp.PURGE_DELAY_OPTION) == old
    assert logged and "host_page_retention/host_policy: native fallback" in logged[0]


def test_install_engages_and_logs(monkeypatch):
    opts = _options_or_skip()
    old = opts.get(hp.PURGE_DELAY_OPTION)
    logged = []
    monkeypatch.setattr(dispatch, "_CAPABILITY_LOGGED", {})
    monkeypatch.setattr(dispatch, "_log_info", logged.append)
    try:
        assert hp.install_host_policy("test") is True
        assert opts.get(hp.PURGE_DELAY_OPTION) == hp.PURGE_DELAY_MS
        assert "host_page_retention/host_policy: engaged" in logged[0]
        assert f"{old} -> {hp.PURGE_DELAY_MS} ms" in logged[0]
    finally:
        opts.set(hp.PURGE_DELAY_OPTION, old)


def test_package_registers_entry_point():
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels import (
        host_page_retention,
    )

    assert (
        dispatch.get_impl("host_page_retention", "host_policy")
        is host_page_retention.install_host_policy
    )
