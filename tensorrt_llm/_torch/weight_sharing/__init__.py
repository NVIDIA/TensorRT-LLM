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
"""Backend-agnostic weight-sharing utilities (MX, GMS, ...)."""

from tensorrt_llm._torch.weight_sharing.artifact_identity import (
    ARTIFACT_IDENTITY_FORMAT_VERSION,
    ArtifactIdentity,
)
from tensorrt_llm._torch.weight_sharing.post_transform_profiles import (
    LLAMA_POST_TRANSFORM_LAYOUT_ABI_V1,
    QWEN2_DENSE_POST_TRANSFORM_LAYOUT_ABI_V1,
    QWEN3_DENSE_POST_TRANSFORM_LAYOUT_ABI_V1,
    LazyRootModelIdentity,
    PostTransformConfigIdentity,
    PostTransformFeature,
    PostTransformProfile,
    PostTransformProfileRegistry,
    PostTransformQualificationDecision,
    PostTransformQualificationReason,
    PostTransformRuntimeConfig,
    PostTransformRuntimeConstraints,
    PostTransformTransferScope,
)
from tensorrt_llm._torch.weight_sharing.source_identity import (
    SOURCE_IDENTITY_FORMAT_VERSION,
    IdentityCheckDecision,
    IdentityCheckPolicy,
    IdentityMatchResult,
    SourceIdentity,
    SourceIdentityMismatchError,
    check_weight_sharing_compatibility,
)
from tensorrt_llm._torch.weight_sharing.weight_manifest import (
    WEIGHT_MANIFEST_DIR_ENV,
    WEIGHT_MANIFEST_FAMILIES,
    WEIGHT_MANIFEST_FILE_PATTERN,
    WEIGHT_MANIFEST_FORMAT_VERSION,
    WEIGHT_MANIFEST_KINDS,
    WEIGHT_MANIFEST_ROLE_ENV,
    SkippedTensor,
    WeightManifest,
    WeightManifestDiff,
    WeightManifestEntry,
    WeightManifestWriteResult,
    build_weight_manifest,
    canonical_tensor_bytes,
    compare_weight_manifests,
    load_weight_manifest,
    manifest_file_name,
    maybe_write_weight_manifest,
    serialize_weight_manifest,
    write_weight_manifest,
)

__all__ = [
    "ARTIFACT_IDENTITY_FORMAT_VERSION",
    "LLAMA_POST_TRANSFORM_LAYOUT_ABI_V1",
    "QWEN2_DENSE_POST_TRANSFORM_LAYOUT_ABI_V1",
    "QWEN3_DENSE_POST_TRANSFORM_LAYOUT_ABI_V1",
    "SOURCE_IDENTITY_FORMAT_VERSION",
    "WEIGHT_MANIFEST_DIR_ENV",
    "WEIGHT_MANIFEST_FAMILIES",
    "WEIGHT_MANIFEST_FILE_PATTERN",
    "WEIGHT_MANIFEST_FORMAT_VERSION",
    "WEIGHT_MANIFEST_KINDS",
    "WEIGHT_MANIFEST_ROLE_ENV",
    "ArtifactIdentity",
    "IdentityCheckDecision",
    "IdentityCheckPolicy",
    "IdentityMatchResult",
    "LazyRootModelIdentity",
    "PostTransformConfigIdentity",
    "PostTransformFeature",
    "PostTransformProfile",
    "PostTransformProfileRegistry",
    "PostTransformQualificationDecision",
    "PostTransformQualificationReason",
    "PostTransformRuntimeConfig",
    "PostTransformRuntimeConstraints",
    "PostTransformTransferScope",
    "SkippedTensor",
    "SourceIdentity",
    "SourceIdentityMismatchError",
    "WeightManifest",
    "WeightManifestDiff",
    "WeightManifestEntry",
    "WeightManifestWriteResult",
    "build_weight_manifest",
    "canonical_tensor_bytes",
    "check_weight_sharing_compatibility",
    "compare_weight_manifests",
    "load_weight_manifest",
    "manifest_file_name",
    "maybe_write_weight_manifest",
    "serialize_weight_manifest",
    "write_weight_manifest",
]
