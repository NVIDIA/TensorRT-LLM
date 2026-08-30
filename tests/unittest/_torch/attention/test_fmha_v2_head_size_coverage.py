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
import tempfile

import pytest

_FMHA_V2_DIR = pathlib.Path(__file__).resolve().parents[4] / "cpp" / "kernels" / "fmha_v2"

# Head dims that sm100 vision encoders depend on, and the model that needs each.
_SM100_REQUIRED_HEAD_SIZES = {
    72: "Gemma3 VL",
    80: "Clip/SigLip",
    104: "Pixtral / Mistral-Large-3 (hidden 1664 / 16 heads)",
}


@pytest.fixture(scope="module")
def sm100_specs():
    """``(generator_module, sm100_kernel_specs)``, generating no files.

    ``enumerate_kernels`` hands its result to ``generate_files``, so replacing
    that function captures the spec list without emitting or compiling anything.
    It does still ``mkdir`` ``./generated`` relative to the cwd, hence the
    scratch directory.
    """
    setup_py = _FMHA_V2_DIR / "setup.py"
    if not setup_py.is_file():
        pytest.skip(f"fmha_v2 generator not present at {setup_py}")

    spec = importlib.util.spec_from_file_location("fmha_v2_setup", setup_py)
    module = importlib.util.module_from_spec(spec)
    captured = []

    with pytest.MonkeyPatch.context() as patch, tempfile.TemporaryDirectory() as scratch:
        patch.chdir(scratch)
        # The only switch gating sm100 enumeration; build_wheel.py sets it too.
        patch.setenv("ENABLE_SM100", "1")
        spec.loader.exec_module(module)
        module.generate_files = captured.extend
        module.enumerate_kernels()

    assert captured, "enumerate_kernels did not reach generate_files"
    return module, [kspec for kspec, *_ in captured if kspec.sm == 100]


@pytest.mark.parametrize(
    "head_size", sorted(_SM100_REQUIRED_HEAD_SIZES), ids=lambda hs: f"head_dim_{hs}"
)
def test_sm100_generates_required_vision_head_size(head_size, sm100_specs):
    """Each required head dim must have the kernel the encoder actually asks for.

    ``PixtralAttention`` runs bf16 context attention over packed QKV with no KV
    cache, which is what the dispatcher's failed lookup reported (``dataType =
    bf16 ... attentionInputLayout = packed_qkv``). Asserting that combination
    rather than bare head-size membership is what makes the test bite: a clause
    generating only, say, the separate-Q-K-V layout would look like coverage
    while leaving the reported failure in place.
    """
    module, specs = sm100_specs
    # One spec per (dtype, layout, tiling) combination, so just assert nonempty.
    matching = [
        kspec
        for kspec in specs
        if kspec.head_size == head_size
        and kspec.dtype == "bf16"
        and kspec.input_layout == module.InputLayout.PACKED_QKV
        and kspec.flash_attention
    ]
    assert matching, (
        f"No sm100 bf16 packed-QKV fmha v2 kernel is generated for head_size={head_size}, "
        f"needed by {_SM100_REQUIRED_HEAD_SIZES[head_size]}. Without it the dispatcher falls "
        f"back to unfused MHA and the attention workspace is sized in TiB. Add a clause for "
        f"this head size to the specs_names filter in "
        f"cpp/kernels/fmha_v2/setup.py::enumerate_kernels. Generated sm100 head sizes: "
        f"{sorted({kspec.head_size for kspec in specs})}"
    )
