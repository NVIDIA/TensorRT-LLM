# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DSA attention metadata of the default Kimi K3.1 Rubin target at an NVFP4 KV cache or FP4 index keys.

At the specialized recipe (FP8 KV cache, FP8 index keys) the default target builds the specialized
metadata classes themselves (``rubin_dsa_default_backend.default_dsa_backend_cls``). Its other
recipes build these two, each the specialized class of its layout plus two mixins:

* ``RubinDsaDefaultMetadata`` (dep16 and tep8) on ``RubinDSAMetadata``;
* ``RubinDsaDefaultHelixMetadata`` (helix8) on ``RubinDSAHelixMetadata``.

The mixins define hook methods only and each class calls them in a fixed order:
``RubinDsaDefaultFp4Mixin`` (``rubin_dsa_default_fp4``) for the index keys, FP8 or FP4, and
``RubinDsaDefaultNvfp4Mixin`` (``rubin_dsa_default_nvfp4``) for the latent KV cache, FP8 or NVFP4.
Each class defines the methods of its ``OVERRIDES``; every other method is its specialized class's:

* ``__post_init__``: the specialized post-init, then the index-key binding and the latent-KV
  allocations (outside capture). helix8 configures its key-sharded Top-K for FP8 or FP4 index keys
  in place of the specialized FP8-only rule;
* ``prepare``: the specialized step preparation, then the latent-KV step inputs;
* ``prepare_for_spec_decode``: the verify window against the index-key kernel's one-atom bound,
  then the specialized planning (one atom, no split);
* ``on_update_kv_lens``: the specialized decode metadata with the index-key slots at the cache's
  dtype, then the latent-KV part (device work only, replayed by a captured graph).
"""

from __future__ import annotations

from typing import ClassVar, Tuple

from .rubin_dsa_default_fp4 import RubinDsaDefaultFp4Mixin
from .rubin_dsa_default_nvfp4 import RubinDsaDefaultNvfp4Mixin
from .rubin_dsa_metadata import RubinDSAHelixMetadata, RubinDSAMetadata

_OVERRIDES = ("__post_init__", "prepare", "prepare_for_spec_decode", "on_update_kv_lens")


class RubinDsaDefaultMetadata(RubinDsaDefaultFp4Mixin, RubinDsaDefaultNvfp4Mixin, RubinDSAMetadata):
    """The default target's DSA metadata of dep16 and tep8."""

    # The methods this class defines; every other method is the specialized class's.
    OVERRIDES: ClassVar[Tuple[str, ...]] = _OVERRIDES

    def __post_init__(self):
        RubinDSAMetadata.__post_init__(self)
        self._fp4_post_init()
        self._nvfp4_post_init()

    def prepare(self):
        RubinDSAMetadata.prepare(self)
        self._nvfp4_prepare()

    def prepare_for_spec_decode(self, kv_lens):
        self._fp4_check_verify_window()
        RubinDSAMetadata.prepare_for_spec_decode(self, kv_lens)

    def on_update_kv_lens(self) -> None:
        self._fp4_decode_metadata()
        self._nvfp4_on_update_kv_lens()


class RubinDsaDefaultHelixMetadata(
    RubinDsaDefaultFp4Mixin, RubinDsaDefaultNvfp4Mixin, RubinDSAHelixMetadata
):
    """The default target's DSA metadata of helix8."""

    OVERRIDES: ClassVar[Tuple[str, ...]] = _OVERRIDES

    def __post_init__(self):
        RubinDSAMetadata.__post_init__(self)
        self._fp4_post_init()
        self.key_shard_world, self.key_shard_rows_max, self.key_shard_max_local_len = (
            self._fp4_key_shard_config() if not self.is_cross else (1, 0, 0)
        )
        self._nvfp4_post_init()

    # The same functions as dep16 / tep8: the specialized helix8 class takes both from its base.
    prepare = RubinDsaDefaultMetadata.prepare
    prepare_for_spec_decode = RubinDsaDefaultMetadata.prepare_for_spec_decode

    def on_update_kv_lens(self) -> None:
        self._fp4_helix_decode_metadata()
        self._nvfp4_on_update_kv_lens()


__all__ = ["RubinDsaDefaultHelixMetadata", "RubinDsaDefaultMetadata"]
