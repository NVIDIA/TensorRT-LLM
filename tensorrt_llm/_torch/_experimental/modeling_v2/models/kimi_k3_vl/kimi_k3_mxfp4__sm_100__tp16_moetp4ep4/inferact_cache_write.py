# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Default-off, fixed-shape Inferact draft KV producer/store fusion.

The worker keeps the exact stock MLA class. The existing page placement owns
valid, unique (page, offset) pairs for all seven rows; these are draft rows,
not accepted-context writes. No embedding, mask, frequency or collective changes.
"""

import torch

from .inferact_boundary import _model_supported
from .inferact_cache_write_kernel import is_compiled, is_supported, paged_cache_write


class InferactPagedCacheWrite:
    """Own only static model admission; destination placement stays with the caller."""

    def __init__(self, draft_model, mapping) -> None:
        self.model_supported = _model_supported(draft_model, mapping)

    def try_write(self, latent, weight, eps, freqs_cis, cache, pages, offsets) -> bool:
        """Write all seven draft rows, or return False before mutation for stock fallback.

        ``latent`` is BF16 [1, 7, 576], ``freqs_cis`` is complex64 [1, 7, 32],
        and ``pages``/``offsets`` are the stock int64 [1, 7] placement tensors.
        The lower-level op checks all dtypes, layouts, physical aliases and device.
        The stock page builder owns destination bounds and uniqueness. Views here
        allocate no tensor storage. Compilation failures propagate without retry.
        """
        if not self.model_supported:
            return False
        if (
            tuple(latent.shape) != (1, 7, 576)
            or not latent.is_contiguous()
            or freqs_cis.dtype != torch.complex64
            or tuple(freqs_cis.shape) != (1, 7, 32)
            or not freqs_cis.is_contiguous()
            or freqs_cis.is_conj()
            or tuple(pages.shape) != (1, 7)
            or not pages.is_contiguous()
            or tuple(offsets.shape) != (1, 7)
            or not offsets.is_contiguous()
        ):
            return False
        x = latent.view(7, 576)
        freqs = torch.view_as_real(freqs_cis).view(7, 32, 2)
        flat_pages, flat_offsets = pages.view(7), offsets.view(7)
        if not is_supported(x, weight, freqs, cache, flat_pages, flat_offsets, eps):
            return False
        # No compilation or allocation on a first encounter during capture.
        # Warmup executes the exact same signature before native graph capture.
        with torch.cuda.device(x.device):
            if torch.cuda.is_current_stream_capturing() and not is_compiled(eps, x.device.index):
                return False
        paged_cache_write(x, weight, freqs, cache, flat_pages, flat_offsets, eps)
        return True
