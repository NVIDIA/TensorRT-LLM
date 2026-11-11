# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The mHC (manifold hyper-connection) boundaries around GLM-5.3-Flash's sublayers, one op per kernel.

Every sublayer reads and writes four residual streams ``[M, 4, hidden]``. With ``hc`` a sublayer's
mixer weights (``fn [24, 4 * hidden]``, ``scale [3]``, ``base [24]``, all FP32), ``norm`` its input
RMSNorm weight (BF16) and ``scalars`` the tuple ``(sinkhorn_iters, norm_eps, hc_eps, sinkhorn_eps,
post_mult, layer_input_eps)`` (GLM-5.3-Flash's values are ``mhc_big_fuse.GLM_SCALARS``):

* The first sublayer's pre-mapping, on the embedding's expanded streams::

      mixes, sqrsum = mhc_gemm_sqrsum(streams.view(M, -1), hc.fn, gemm_sqrsum_tile_n(M))
      bs = big_fuse_block_size(M)
      post, comb, x = mhc_big_fuse(mixes, sqrsum, streams, hc.scale, hc.base, norm, *scalars, bs)

* Every boundary between two sublayers, ``out`` being the previous one's reduced output::

      args = (hc.fn, hc.scale, hc.base, norm, *scalars)
      tactic = fused_hc_tactic(out, streams, post, comb, *args)
      scratch = fused_hc_scratch(M, tactic, out.device)
      streams, post, comb, x = mhc_fused_hc(out, streams, post, comb, *args, tactic, *scratch)

* After the last sublayer::

      streams = mhc_post_mapping(out, streams, post, comb)

Expanding the embedding into streams and collapsing them at the end (an unweighted mean, then the
final RMSNorm) are plain torch.
"""
