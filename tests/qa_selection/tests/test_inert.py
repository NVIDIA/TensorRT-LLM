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
"""AC-6: without a named machine, loading the plugin changes nothing.

    selection.run(...)              pytest --collect-only -p qa_selection.plugin ...
    selection.without_plugin(...)   pytest --collect-only ...

The second is the baseline: the command a CI job runs today. The mock module
carries `skip_less_device`, which only the plugin declares, so the
plugin-absent arm collects it as an unknown marker -- which is why this suite's
`pytest.ini` must not set `--strict-markers`.
"""

from mock_suite import DeviceCount


def test_a_loaded_but_untargeted_plugin_is_invisible(selection):
    """Same ids in the same order as a run without `-p`, and nothing written."""
    loaded = selection.run(DeviceCount.MODULE, "--selection-out-dir={out}")
    bare = selection.without_plugin(DeviceCount.MODULE)

    assert loaded.selected == bare.selected == [DeviceCount.UNMARKED, DeviceCount.NEEDS_EIGHT]

    # An output directory was named and nothing was written to it, not even
    # the directory itself.
    assert loaded.written == []
    assert not loaded.out_dir.exists()

    assert loaded.summary == []
    assert "deselected" not in loaded.result.stdout.str()


def test_options_are_not_validated_while_inert(selection):
    """Without `--machine` no other option is read, so none can be refused.

    `--ladder=999` is a usage error for any named machine.
    """
    run = selection.run(DeviceCount.MODULE, "--ladder=999")

    assert run.selected == [DeviceCount.UNMARKED, DeviceCount.NEEDS_EIGHT]
