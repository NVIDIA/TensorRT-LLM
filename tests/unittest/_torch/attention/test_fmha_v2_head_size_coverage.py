# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Static coverage test for the fmha v2 kernel generator's head-size allowlist.

``cpp/kernels/fmha_v2/setup.py::enumerate_kernels`` decides which FMHA kernels
get compiled. Its general clause admits ``sm in [80, 86, 89, 90, 120]`` -- sm100
is deliberately absent and is served only by narrow per-model clauses. A vision
encoder whose head dim has no such clause therefore silently gets *no* kernel:
``FmhaDispatcher`` logs "Fall back to unfused MHA", ``mEnableContextFMHA`` goes
false, and the quadratic ``qk_buf``/``qk_buf_float`` terms in
``AttentionOp::getWorkspaceSizeForContext`` size the attention workspace at
multiple TiB -- surfacing as an implausible OOM rather than as a missing kernel
(https://nvbugs/6665906).

Nothing else in the tree ties a model's head dim to that allowlist, so this test
asserts the coverage directly. It enumerates the generator in-process (no
compilation, no GPU, ~0.1s) by intercepting ``generate_files``.
"""

import importlib.util
import pathlib
import sys
import unittest.mock as mock

import pytest

_FMHA_V2_DIR = pathlib.Path(__file__).resolve().parents[4] / "cpp" / "kernels" / "fmha_v2"

# Head dims that sm100 vision encoders depend on, and the model that needs each.
# The generator emits one spec per (dtype, input layout, tiling) combination, so
# a covered head size lands many specs; assert only that it is nonempty.
_SM100_REQUIRED_HEAD_SIZES = {
    72: "Gemma3 VL",
    80: "Clip/SigLip",
    104: "Pixtral / Mistral-Large-3 (hidden 1664 / 16 heads)",
}


def _enumerate_specs():
    """Return ``(module, kernel_specs)`` for the generator, generating no files."""
    setup_py = _FMHA_V2_DIR / "setup.py"
    if not setup_py.is_file():
        pytest.skip(f"fmha_v2 generator not present at {setup_py}")

    spec = importlib.util.spec_from_file_location("fmha_v2_setup", setup_py)
    module = importlib.util.module_from_spec(spec)

    captured = {}

    def capture(specs_names):
        captured["specs"] = specs_names

    # ``setup.py`` generates on import as well as via ``enumerate_kernels``, and
    # reads its arch switches from the environment. Neutralize both, then call
    # ``enumerate_kernels`` explicitly with the sm100 kernels enabled.
    env = {"GENERATE_CUBIN": "1", "ENABLE_SM100": "1", "ENABLE_SM120": "1"}
    with mock.patch.dict("os.environ", env), mock.patch.object(sys, "argv", ["setup.py"]):
        # The import-time pass is redirected into ``capture`` too, so it cannot
        # write into the source tree.
        module.generate_files = capture
        spec.loader.exec_module(module)
        module.generate_files = capture
        captured.pop("specs", None)
        module.enumerate_kernels()

    assert "specs" in captured, "enumerate_kernels did not reach generate_files"
    return module, [entry[0] for entry in captured["specs"]]


@pytest.fixture(scope="module")
def generator():
    return _enumerate_specs()


@pytest.fixture(scope="module")
def sm100_head_sizes(generator):
    _, specs = generator
    return {kspec.head_size for kspec in specs if kspec.sm == 100}


@pytest.mark.parametrize(
    "head_size", sorted(_SM100_REQUIRED_HEAD_SIZES), ids=lambda hs: f"head_dim_{hs}"
)
def test_sm100_generates_required_vision_head_size(head_size, sm100_head_sizes):
    assert head_size in sm100_head_sizes, (
        f"No sm100 fmha v2 kernel is generated for head_size={head_size}, needed by "
        f"{_SM100_REQUIRED_HEAD_SIZES[head_size]}. Without it the dispatcher falls back "
        f"to unfused MHA and the attention workspace is sized in TiB. Add a clause for "
        f"this head size to the specs_names filter in "
        f"cpp/kernels/fmha_v2/setup.py::enumerate_kernels. Generated sm100 head sizes: "
        f"{sorted(sm100_head_sizes)}"
    )


def test_sm100_pixtral_head_size_supports_the_packed_qkv_context_query(generator):
    """The 104 kernels must cover the layout the Pixtral encoder actually asks for.

    ``PixtralAttention`` runs bf16 context attention over packed QKV with no KV
    cache, which is what the dispatcher's failed lookup reported
    (``dataType = bf16 ... attentionInputLayout = packed_qkv``). A clause that
    generated only, say, the separate-Q-K-V layout would satisfy the coverage
    test above while leaving the reported failure in place.
    """
    module, specs = generator
    matching = [
        kspec
        for kspec in specs
        if kspec.sm == 100
        and kspec.head_size == 104
        and kspec.dtype == "bf16"
        and kspec.input_layout == module.InputLayout.PACKED_QKV
        and kspec.flash_attention
    ]
    assert matching, (
        "sm100 head_size=104 has no bf16 packed-QKV flash-attention kernel, which is "
        "the exact configuration the Pixtral vision encoder queries."
    )
