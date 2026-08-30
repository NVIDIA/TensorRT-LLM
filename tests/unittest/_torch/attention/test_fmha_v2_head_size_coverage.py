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
is deliberately absent and is served only for ``VISION_ENCODER_HEAD_SIZES``. A
vision encoder whose head dim is missing from that list therefore silently gets
*no* kernel, and ``selected_mask_types`` independently decides whether the
kernel that does exist keeps the ``PADDING`` mask variant the encoder asks for.
Both gates failed in turn for head dim 104 (https://nvbugs/6665906):

* no kernel at all -> ``FmhaDispatcher`` falls back to unfused MHA, and the
  quadratic ``qk_buf``/``qk_buf_float`` terms in
  ``AttentionOp::getWorkspaceSizeForContext`` size the attention workspace at
  multiple TiB, surfacing as an implausible OOM rather than a missing kernel.
* kernel without the padding mask -> the lookup in
  ``fused_multihead_attention_v2.cpp`` is an exact hash match with no fallback,
  so every rank aborts with "FMHA kernels are not found with these parameters".

Nothing else in the tree ties a model's head dim to that allowlist, so this test
asserts the coverage directly. It enumerates the generator in-process (no
compilation, no GPU, ~0.1s) by intercepting ``generate_files``.
"""

import importlib.util
import pathlib
import tempfile

import pytest

_SETUP_PY = pathlib.Path(__file__).resolve().parents[4] / "cpp" / "kernels" / "fmha_v2" / "setup.py"

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
    scratch directory. The env matches what ``build_wheel.py`` exports, since
    both ``ENABLE_SM100`` (which gates sm100 enumeration) and ``GENERATE_CUBIN``
    (which turns on the mask-variant pruning) change the result.
    """
    if not _SETUP_PY.is_file():
        pytest.skip(f"fmha_v2 generator not present at {_SETUP_PY}")

    spec = importlib.util.spec_from_file_location("fmha_v2_setup", _SETUP_PY)
    module = importlib.util.module_from_spec(spec)
    captured = []

    with pytest.MonkeyPatch.context() as patch, tempfile.TemporaryDirectory() as scratch:
        patch.chdir(scratch)
        patch.setenv("ENABLE_SM100", "1")
        patch.setenv("GENERATE_CUBIN", "1")
        spec.loader.exec_module(module)
        module.generate_files = captured.extend
        module.enumerate_kernels()

    assert captured, "enumerate_kernels did not reach generate_files"
    return module, [kspec for kspec, *_ in captured if kspec.sm == 100]


@pytest.mark.parametrize(
    "head_size", sorted(_SM100_REQUIRED_HEAD_SIZES), ids=lambda hs: f"head_dim_{hs}"
)
def test_sm100_vision_head_size_has_a_padding_mask_kernel(head_size, sm100_specs):
    """Each required head dim must get the kernel the encoder actually asks for.

    ``PixtralAttention`` runs bf16 context attention over packed QKV with no KV
    cache, which is what the dispatcher's failed lookup reported (``dataType =
    bf16 ... attentionInputLayout = packed_qkv``). Asserting that combination
    rather than bare head-size membership is what makes the test bite: a clause
    generating only, say, the separate-Q-K-V layout would look like coverage
    while leaving the reported failure in place.

    The two asserts are the two gates, in the order they fail: existing at all,
    then keeping the ``PADDING`` mask variant (flag 0 of ``selected_mask_types``)
    that a ViT needs, having no causal structure over a padded batch.
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
        f"back to unfused MHA and the attention workspace is sized in TiB. Add this head size "
        f"to VISION_ENCODER_HEAD_SIZES in cpp/kernels/fmha_v2/setup.py. Generated sm100 head "
        f"sizes: {sorted({kspec.head_size for kspec in specs})}"
    )

    assert any(module.selected_mask_types(kspec)[0] == "1" for kspec in matching), (
        f"Every sm100 bf16 packed-QKV kernel for head_size={head_size} "
        f"({_SM100_REQUIRED_HEAD_SIZES[head_size]}) is compiled with the padding mask "
        f"disabled, so the encoder's mask=PADDING lookup finds no kernel and aborts. Add this "
        f"head size to PACKED_QKV_PADDING_MASK_HEAD_SIZES in cpp/kernels/fmha_v2/setup.py."
    )


def test_required_head_sizes_match_the_generators_vision_list(sm100_specs):
    """This file's required set must track the generator's own vision list.

    Keeps a newly supported vision head size from being added to the generator
    without also declaring the model that needs it here.
    """
    module, _ = sm100_specs
    assert set(_SM100_REQUIRED_HEAD_SIZES) == set(module.VISION_ENCODER_HEAD_SIZES)
