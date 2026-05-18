# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""License and attribution data and generators for the standalone llmc package.

Separated from create_standalone_package.py to keep data and logic apart.
"""

import os
import re
import textwrap

# Vendored projects with code actually present in auto_deploy source.
# Only these should appear in the standalone LICENSE file.
VENDORED_PROJECTS = [
    {
        "name": "causal-conv1d",
        "url": "https://github.com/Dao-AILab/causal-conv1d",
        "copyright": "Copyright (c) 2024, Tri Dao.",
        "license_id": "BSD-3-Clause",
    },
    {
        "name": "flash-linear-attention",
        "url": "https://github.com/fla-org/flash-linear-attention",
        "copyright": "Copyright (c) 2023-2025 Songlin Yang",
        "license_id": "MIT",
    },
    {
        "name": "Mamba",
        "url": "https://github.com/state-spaces/mamba",
        "copyright": "Copyright 2023 Tri Dao, Albert Gu",
        "license_id": "Apache-2.0",
    },
    {
        "name": "SGLang",
        "url": "https://github.com/sgl-project/sglang",
        "copyright": "Copyright contributors to the SGLang project",
        "license_id": "Apache-2.0",
    },
    {
        "name": "vLLM",
        "url": "https://github.com/vllm-project/vllm",
        "copyright": "Copyright contributors to the vLLM project",
        "license_id": "Apache-2.0",
    },
    {
        "name": "Transformers",
        "url": "https://github.com/huggingface/transformers",
        "copyright": "Copyright 2018 The HuggingFace Team",
        "license_id": "Apache-2.0",
    },
]

# Direct dependency license mapping for ATTRIBUTIONS generation.
DIRECT_DEP_LICENSES = {
    "torch": ("PyTorch", "BSD-3-Clause", "https://github.com/pytorch/pytorch"),
    "transformers": ("Transformers", "Apache-2.0", "https://github.com/huggingface/transformers"),
    "pydantic": ("Pydantic", "MIT", "https://github.com/pydantic/pydantic"),
    "pydantic-settings": (
        "pydantic-settings",
        "MIT",
        "https://github.com/pydantic/pydantic-settings",
    ),
    "triton": ("Triton", "MIT", "https://github.com/triton-lang/triton"),
    "flashinfer-python": (
        "FlashInfer",
        "Apache-2.0",
        "https://github.com/flashinfer-ai/flashinfer",
    ),
    "safetensors": ("safetensors", "Apache-2.0", "https://github.com/huggingface/safetensors"),
    "accelerate": ("Accelerate", "Apache-2.0", "https://github.com/huggingface/accelerate"),
    "huggingface-hub": (
        "huggingface-hub",
        "Apache-2.0",
        "https://github.com/huggingface/huggingface_hub",
    ),
    "omegaconf": ("OmegaConf", "BSD-3-Clause", "https://github.com/omry/omegaconf"),
    "pyyaml": ("PyYAML", "MIT", "https://github.com/yaml/pyyaml"),
    "numpy": ("NumPy", "BSD-3-Clause", "https://github.com/numpy/numpy"),
    "pillow": ("Pillow", "HPND", "https://github.com/python-pillow/Pillow"),
    "einops": ("einops", "MIT", "https://github.com/arogozhnikov/einops"),
    "six": ("six", "MIT", "https://github.com/benjaminp/six"),
    "importlib-metadata": (
        "importlib-metadata",
        "Apache-2.0",
        "https://github.com/python/importlib_metadata",
    ),
    "werkzeug": ("Werkzeug", "BSD-3-Clause", "https://github.com/pallets/werkzeug"),
    "StrEnum": ("StrEnum", "MIT", "https://github.com/irgeek/StrEnum"),
    "graphviz": ("graphviz", "MIT", "https://github.com/xflr6/graphviz"),
}

# Full license text templates keyed by SPDX identifier.
LICENSE_TEXTS = {
    "Apache-2.0": textwrap.dedent("""\
                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

        TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

        1. Definitions.

           "License" shall mean the terms and conditions for use, reproduction,
           and distribution as defined by Sections 1 through 9 of this document.

           "Licensor" shall mean the copyright owner or entity authorized by
           the copyright owner that is granting the License.

           "Legal Entity" shall mean the union of the acting entity and all
           other entities that control, are controlled by, or are under common
           control with that entity. For the purposes of this definition,
           "control" means (i) the power, direct or indirect, to cause the
           direction or management of such entity, whether by contract or
           otherwise, or (ii) ownership of fifty percent (50%) or more of the
           outstanding shares, or (iii) beneficial ownership of such entity.

           "You" (or "Your") shall mean an individual or Legal Entity
           exercising permissions granted by this License.

           "Source" form shall mean the preferred form for making modifications,
           including but not limited to software source code, documentation
           source, and configuration files.

           "Object" form shall mean any form resulting from mechanical
           transformation or translation of a Source form, including but
           not limited to compiled object code, generated documentation,
           and conversions to other media types.

           "Work" shall mean the work of authorship, whether in Source or
           Object form, made available under the License, as indicated by a
           copyright notice that is included in or attached to the work
           (an example is provided in the Appendix below).

           "Derivative Works" shall mean any work, whether in Source or Object
           form, that is based on (or derived from) the Work and for which the
           editorial revisions, annotations, elaborations, or other modifications
           represent, as a whole, an original work of authorship. For the purposes
           of this License, Derivative Works shall not include works that remain
           separable from, or merely link (or bind by name) to the interfaces of,
           the Work and Derivative Works thereof.

           "Contribution" shall mean any work of authorship, including
           the original version of the Work and any modifications or additions
           to that Work or Derivative Works thereof, that is intentionally
           submitted to Licensor for inclusion in the Work by the copyright owner
           or by an individual or Legal Entity authorized to submit on behalf of
           the copyright owner. For the purposes of this definition, "submitted"
           means any form of electronic, verbal, or written communication sent
           to the Licensor or its representatives, including but not limited to
           communication on electronic mailing lists, source code control systems,
           and issue tracking systems that are managed by, or on behalf of, the
           Licensor for the purpose of discussing and improving the Work, but
           excluding communication that is conspicuously marked or otherwise
           designated in writing by the copyright owner as "Not a Contribution."

           "Contributor" shall mean Licensor and any individual or Legal Entity
           on behalf of whom a Contribution has been received by Licensor and
           subsequently incorporated within the Work.

        2. Grant of Copyright License. Subject to the terms and conditions of
           this License, each Contributor hereby grants to You a perpetual,
           worldwide, non-exclusive, no-charge, royalty-free, irrevocable
           copyright license to reproduce, prepare Derivative Works of,
           publicly display, publicly perform, sublicense, and distribute the
           Work and such Derivative Works in Source or Object form.

        3. Grant of Patent License. Subject to the terms and conditions of
           this License, each Contributor hereby grants to You a perpetual,
           worldwide, non-exclusive, no-charge, royalty-free, irrevocable
           (except as stated in this section) patent license to make, have made,
           use, offer to sell, sell, import, and otherwise transfer the Work,
           where such license applies only to those patent claims licensable
           by such Contributor that are necessarily infringed by their
           Contribution(s) alone or by combination of their Contribution(s)
           with the Work to which such Contribution(s) was submitted. If You
           institute patent litigation against any entity (including a
           cross-claim or counterclaim in a lawsuit) alleging that the Work
           or a Contribution incorporated within the Work constitutes direct
           or contributory patent infringement, then any patent licenses
           granted to You under this License for that Work shall terminate
           as of the date such litigation is filed.

        4. Redistribution. You may reproduce and distribute copies of the
           Work or Derivative Works thereof in any medium, with or without
           modifications, and in Source or Object form, provided that You
           meet the following conditions:

           (a) You must give any other recipients of the Work or
               Derivative Works a copy of this License; and

           (b) You must cause any modified files to carry prominent notices
               stating that You changed the files; and

           (c) You must retain, in the Source form of any Derivative Works
               that You distribute, all copyright, patent, trademark, and
               attribution notices from the Source form of the Work,
               excluding those notices that do not pertain to any part of
               the Derivative Works; and

           (d) If the Work includes a "NOTICE" text file as part of its
               distribution, then any Derivative Works that You distribute must
               include a readable copy of the attribution notices contained
               within such NOTICE file, excluding those notices that do not
               pertain to any part of the Derivative Works, in at least one
               of the following places: within a NOTICE text file distributed
               as part of the Derivative Works; within the Source form or
               documentation, if provided along with the Derivative Works; or,
               within a display generated by the Derivative Works, if and
               wherever such third-party notices normally appear. The contents
               of the NOTICE file are for informational purposes only and
               do not modify the License. You may add Your own attribution
               notices within Derivative Works that You distribute, alongside
               or as an addendum to the NOTICE text from the Work, provided
               that such additional attribution notices cannot be construed
               as modifying the License.

           You may add Your own copyright statement to Your modifications and
           may provide additional or different license terms and conditions
           for use, reproduction, or distribution of Your modifications, or
           for any such Derivative Works as a whole, provided Your use,
           reproduction, and distribution of the Work otherwise complies with
           the conditions stated in this License.

        5. Submission of Contributions. Unless You explicitly state otherwise,
           any Contribution intentionally submitted for inclusion in the Work
           by You to the Licensor shall be under the terms and conditions of
           this License, without any additional terms or conditions.
           Notwithstanding the above, nothing herein shall supersede or modify
           the terms of any separate license agreement you may have executed
           with Licensor regarding such Contributions.

        6. Trademarks. This License does not grant permission to use the trade
           names, trademarks, service marks, or product names of the Licensor,
           except as required for reasonable and customary use in describing the
           origin of the Work and reproducing the content of the NOTICE file.

        7. Disclaimer of Warranty. Unless required by applicable law or
           agreed to in writing, Licensor provides the Work (and each
           Contributor provides its Contributions) on an "AS IS" BASIS,
           WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
           implied, including, without limitation, any warranties or conditions
           of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
           PARTICULAR PURPOSE. You are solely responsible for determining the
           appropriateness of using or redistributing the Work and assume any
           risks associated with Your exercise of permissions under this License.

        8. Limitation of Liability. In no event and under no legal theory,
           whether in tort (including negligence), contract, or otherwise,
           unless required by applicable law (such as deliberate and grossly
           negligent acts) or agreed to in writing, shall any Contributor be
           liable to You for damages, including any direct, indirect, special,
           incidental, or consequential damages of any character arising as a
           result of this License or out of the use or inability to use the
           Work (including but not limited to damages for loss of goodwill,
           work stoppage, computer failure or malfunction, or any and all
           other commercial damages or losses), even if such Contributor
           has been advised of the possibility of such damages.

        9. Accepting Warranty or Additional Liability. While redistributing
           the Work or Derivative Works thereof, You may choose to offer,
           and charge a fee for, acceptance of support, warranty, indemnity,
           or other liability obligations and/or rights consistent with this
           License. However, in accepting such obligations, You may act only
           on Your own behalf and on Your sole responsibility, not on behalf
           of any other Contributor, and only if You agree to indemnify,
           defend, and hold each Contributor harmless for any liability
           incurred by, or claims asserted against, such Contributor by reason
           of your accepting any such warranty or additional liability.

        END OF TERMS AND CONDITIONS

        APPENDIX: How to apply the Apache License to your work.

           To apply the Apache License to your work, attach the following
           boilerplate notice, with the fields enclosed by brackets "[]"
           replaced with your own identifying information. (Don't include
           the brackets!)  The text should be enclosed in the appropriate
           comment syntax for the file format. We also recommend that a
           file or class name and description of purpose be included on the
           same "printed page" as the copyright notice for easier
           identification within third-party archives.

        Copyright [yyyy] [name of copyright owner]

        Licensed under the Apache License, Version 2.0 (the "License");
        you may not use this file except in compliance with the License.
        You may obtain a copy of the License at

            http://www.apache.org/licenses/LICENSE-2.0

        Unless required by applicable law or agreed to in writing, software
        distributed under the License is distributed on an "AS IS" BASIS,
        WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
        See the License for the specific language governing permissions and
        limitations under the License.
    """),
    "MIT": textwrap.dedent("""\
        Permission is hereby granted, free of charge, to any person obtaining a copy
        of this software and associated documentation files (the "Software"), to deal
        in the Software without restriction, including without limitation the rights
        to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
        copies of the Software, and to permit persons to whom the Software is
        furnished to do so, subject to the following conditions:

        The above copyright notice and this permission notice shall be included in all
        copies or substantial portions of the Software.

        THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
        IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
        FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
        AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
        LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
        OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
        SOFTWARE.
    """),
    "BSD-3-Clause": textwrap.dedent("""\
        Redistribution and use in source and binary forms, with or without
        modification, are permitted provided that the following conditions are met:

        * Redistributions of source code must retain the above copyright notice, this
          list of conditions and the following disclaimer.

        * Redistributions in binary form must reproduce the above copyright notice,
          this list of conditions and the following disclaimer in the documentation
          and/or other materials provided with the distribution.

        * Neither the name of the copyright holder nor the names of its
          contributors may be used to endorse or promote products derived from
          this software without specific prior written permission.

        THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
        AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
        IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
        DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
        FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
        DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
        SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
        CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
        OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
        OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
    """),
}


def generate_license(output_dir: str) -> None:
    """Generate a LICENSE file listing only vendored projects present in auto_deploy."""
    lines = []
    lines.append("Copyright (c) 2011-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.\n")
    lines.append(
        "This project is licensed under the Apache 2.0 license, whose full license text "
        "is available below.\n"
    )
    lines.append(
        "This project contains portions of code that are based on or derived from\n"
        "other open source projects, which may have different licenses whose text\n"
        "is available below.\n"
    )
    lines.append(
        "All modifications and additions to other projects are licensed under the\n"
        "Apache License 2.0 unless otherwise specified. Please refer to the individual\n"
        "file headers for specific copyright and license information.\n"
    )
    lines.append(
        "Below is a list of other projects that have portions contained by this project:\n"
    )

    for proj in VENDORED_PROJECTS:
        lines.append("-" * 80)
        lines.append(proj["name"])
        lines.append("-" * 80)
        lines.append(f"Original Source: {proj['url']}")
        lines.append(proj["copyright"])
        lines.append(f"Licensed under the {proj['license_id']} License")
        lines.append("")

    # Append full license texts (deduplicated)
    seen = set()
    for proj in VENDORED_PROJECTS:
        lid = proj["license_id"]
        if lid not in seen and lid in LICENSE_TEXTS:
            seen.add(lid)
            lines.append("=" * 80)
            lines.append(f"                              {lid} LICENSE")
            lines.append("=" * 80)
            lines.append("")
            lines.append(LICENSE_TEXTS[lid])

    with open(os.path.join(output_dir, "LICENSE"), "w") as f:
        f.write("\n".join(lines))


def generate_attributions(output_dir: str, dependencies: list) -> None:
    """Generate ATTRIBUTIONS-Python.md listing direct dependency licenses."""
    lines = []
    lines.append("<!--")
    lines.append(
        "SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. "
        "All rights reserved."
    )
    lines.append("SPDX-License-Identifier: Apache-2.0")
    lines.append("-->")
    lines.append("")
    lines.append("# Third-Party Software Attributions")
    lines.append("")
    lines.append(
        "This file lists the direct runtime dependencies of this package and their licenses."
    )
    lines.append(
        "For transitive dependencies, consult `uv.lock` or run `pip-licenses` after installation."
    )
    lines.append("")

    for dep_str in sorted(dependencies, key=lambda d: d.split("[")[0].lower()):
        # Extract base name (strip version spec)
        base = re.match(r"^([a-zA-Z0-9_-]+(?:\[[^\]]+\])?)", dep_str)
        if not base:
            continue
        pkg_name = base.group(1).split("[")[0]
        info = DIRECT_DEP_LICENSES.get(pkg_name) or DIRECT_DEP_LICENSES.get(pkg_name.lower())
        if not info:
            lines.append(f"## {dep_str}")
            lines.append("")
            lines.append("License: Unknown")
            lines.append("")
            continue
        display_name, license_id, url = info
        lines.append(f"## {display_name}")
        lines.append("")
        lines.append(f"- **PyPI package**: `{dep_str}`")
        lines.append(f"- **License**: {license_id}")
        lines.append(f"- **Source**: {url}")
        lines.append("")

    with open(os.path.join(output_dir, "ATTRIBUTIONS-Python.md"), "w") as f:
        f.write("\n".join(lines))
