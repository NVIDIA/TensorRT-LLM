"""Hopper forward operators."""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute

from .mainloop import SolAttnMainloopSm90
from .split_combine import BlockSparseAttnForwardCombine


class SolAttnSplitForwardSm90(SolAttnMainloopSm90):
    """Run one CTA per KV split and merge the partial outputs."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._combine = BlockSparseAttnForwardCombine(
            dtype=cutlass.BFloat16,
            head_dim=self.tile_hdimv,
            tile_m=16,
            k_block_size=64,
            log_max_splits=1 if self.sol_attn_num_splits == 2 else 2,
            num_threads=128,
            stages=4,
            partial_dtype=cutlass.BFloat16,
        )

    @cute.jit
    def __call__(
        self,
        q: cute.Tensor,
        k: cute.Tensor,
        v: cute.Tensor,
        o: cute.Tensor,
        kc: cute.Tensor,
        vc: cute.Tensor,
        threshold: cute.Tensor,
        lse: cute.Tensor,
        o_partial: cute.Tensor,
        lse_partial: cute.Tensor,
        softmax_scale: cutlass.Float32,
        sink_range: cutlass.Int32,
        stream: cuda.CUstream = None,
    ):
        SolAttnMainloopSm90.__call__(
            self,
            q,
            k,
            v,
            o_partial,
            kc,
            vc,
            threshold,
            lse_partial,
            softmax_scale,
            sink_range,
            stream=stream,
        )

        splits = self.sol_attn_num_splits
        heads = q.shape[2]
        o_partial = cute.make_tensor(
            o_partial.iterator,
            cute.make_layout(
                (
                    splits,
                    o_partial.shape[0],
                    o_partial.shape[1],
                    heads,
                    o_partial.shape[3],
                ),
                stride=(
                    heads * o_partial.stride[2],
                    o_partial.stride[0],
                    o_partial.stride[1],
                    o_partial.stride[2],
                    o_partial.stride[3],
                ),
            ),
        )
        lse_partial = cute.make_tensor(
            lse_partial.iterator,
            cute.make_layout(
                (
                    splits,
                    lse_partial.shape[0],
                    lse_partial.shape[1],
                    heads,
                ),
                stride=(
                    heads * lse_partial.stride[2],
                    lse_partial.stride[0],
                    lse_partial.stride[1],
                    lse_partial.stride[2],
                ),
            ),
        )
        self._combine(
            o_partial,
            lse_partial,
            o,
            lse,
            None,
            None,
            None,
            None,
            None,
            stream,
        )


__all__ = ["SolAttnSplitForwardSm90"]
