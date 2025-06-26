import torch
import triton

from .triton_kernels.rope import rope_fwd_flattened_kernel, rope_fwd_kernel


@torch.library.custom_op("auto_deploy::triton_rope_with_input_pos", mutates_args=())
def apply_rope_with_input_pos(
    x: torch.Tensor, freqs_cis: torch.Tensor, input_pos: torch.Tensor, layout: str
) -> torch.Tensor:
    """Embeds the input using RoPE (https://arxiv.org/abs/2104.09864).

    Supports 6 different layouts of ``x``.:
      - ``'bsnd'``:  (batch, seq_len, n_head, d_head)
      - ``'bnsd'``:  (batch, n_head, seq_len, d_head)
      - ``'sbnd'``:  (seq_len, batch, n_head, d_head)
      - ``'snbd'``:  (seq_len, n_head, batch, d_head)
      - ``'nbsd'``:  (n_head, batch, seq_len, d_head)
      - ``'nsbd'``:  (n_head, seq_len, batch, d_head)


    Args:
        x: key or query Tensor to be embedded
        freqs_cis: contains interleaved cos and sin frequencies.
        input_pos: Tensor of size `b` containing the input offsets.
        layout: string of layout above
    """
    assert set(layout) == {"b", "n", "s", "d"}, "invalid layout."
    assert layout[3] == "d"
    assert x.shape[3] % 2 == 0, "RoPE requires an even number as hidden size."

    y = torch.empty_like(x)

    batch_dim = layout.find("b")
    seq_dim = layout.find("s")
    nhead_dim = layout.find("n")
    if input_pos is None:
        input_pos = torch.tensor([0] * x.shape[batch_dim], device=x.device, dtype=torch.int32)
    N = x.shape[batch_dim]
    L = x.shape[seq_dim]
    H = x.shape[nhead_dim]
    D = x.shape[3]

    stride_n = x.stride(batch_dim)
    stride_l = x.stride(seq_dim)
    stride_h = x.stride(nhead_dim)
    stride_d = x.stride(3)

    BLOCK_SIZE_H = 32
    BLOCK_SIZE_L = min(triton.next_power_of_2(L), 32)
    grid = (
        N,
        triton.cdiv(H, BLOCK_SIZE_H),
        triton.cdiv(L, BLOCK_SIZE_L),
    )
    rope_fwd_kernel[grid](
        x,
        input_pos,
        freqs_cis,
        y,
        N,
        L,
        H,
        D,
        stride_n,
        stride_l,
        stride_h,
        stride_d,
        BLOCK_SIZE_H=BLOCK_SIZE_H,
        BLOCK_SIZE_L=BLOCK_SIZE_L,
    )
    return y


@apply_rope_with_input_pos.register_fake
def apply_rope_with_input_pos_fake(x, freqs_cis, input_pos, layout):
    return torch.empty_like(x)


@torch.library.custom_op("auto_deploy::triton_rope_on_flattened_inputs", mutates_args=())
def apply_rope_on_flattened_inputs(
    x: torch.Tensor,
    freqs_cis: torch.Tensor,
    input_pos: torch.Tensor,
    seq_lens: torch.Tensor,
    seq_start_indices: torch.Tensor,
) -> torch.Tensor:
    """Embeds the input using RoPE (https://arxiv.org/abs/2104.09864).

    Assumes input is flattened as [B*S, N, D]m

    Args:
        x: key or query Tensor to be embedded
        freqs_cis: contains interleaved cos and sin frequencies.
        input_pos: Tensor of size `B` containing the input offsets.
        seq_lens: Tensor of size `B` containing the length of sequences.
        seq_start_indices: Tensor of size `B` containing the start indices of sequences in the
            flattened representation.
    """
    y = torch.empty_like(x)

    B = len(input_pos)  # number of sequences
    assert seq_start_indices.shape[0] == seq_lens.shape[0]

    H = x.shape[1]
    D = x.shape[2]

    L = seq_lens.max().item()

    BLOCK_SIZE_H = 32
    BLOCK_SIZE_L = min(max(triton.next_power_of_2(L), 1), 32)
    grid = (
        B,
        triton.cdiv(H, BLOCK_SIZE_H),
        triton.cdiv(L, BLOCK_SIZE_L),
    )
    rope_fwd_flattened_kernel[grid](
        x,
        seq_lens,
        seq_start_indices,
        input_pos,
        freqs_cis,
        y,
        H,
        D,
        BLOCK_SIZE_H=BLOCK_SIZE_H,
        BLOCK_SIZE_L=BLOCK_SIZE_L,
    )
    return y


@apply_rope_on_flattened_inputs.register_fake
def apply_rope_on_flattened_inputs_fake(x, freqs_cis, input_pos, seq_lens, seq_start_indices):
    return torch.empty_like(x)
