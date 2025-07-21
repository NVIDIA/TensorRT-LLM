from typing import Optional, Tuple

import torch

from tensorrt_llm._utils import nvtx_range


def ceil_div(x: int, y: int) -> int:
    """
    Perform ceiling division of two integers.

    Args:
        x: the dividend.
        y: the divisor.

    Returns:
        The result of the ceiling division.
    """
    return (x + y - 1) // y


def align(x: int, y: int) -> int:
    return ceil_div(x, y) * y


def ceil_to_ue8m0(x: torch.Tensor):
    return torch.pow(2.0, torch.ceil(torch.log2(x.abs())))


@nvtx_range("[DG] quantization")
@torch.compile(dynamic=True)
def per_token_cast_to_fp8_e8m0(
        x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if x.dim() == 2:
        assert x.size(1) % 128 == 0
        m, n = x.shape
        x_view = x.view(m, -1, 128)
        x_amax = x_view.abs().float().amax(dim=2).view(m, -1).clamp(1e-4)
        sf = ceil_to_ue8m0(x_amax / 448.0)
        return (x_view * (1.0 / sf.unsqueeze(2))).to(torch.float8_e4m3fn).view(
            m, n), sf
    else:
        assert x.size(2) % 128 == 0
        g, m, n = x.shape
        x_view = x.view(g, m, -1, 128)
        x_amax = x_view.abs().float().amax(dim=3).view(g, m, -1).clamp(1e-4)
        sf = ceil_to_ue8m0(x_amax / 448.0)
        return (x_view * (1.0 / sf.unsqueeze(3))).to(torch.float8_e4m3fn).view(
            g, m, n), sf


def per_block_cast_to_fp8_e8m0(
        x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if x.dim() == 2:
        m, n = x.shape
        x_padded = torch.zeros((align(m, 128), align(n, 128)),
                               dtype=x.dtype,
                               device=x.device)
        x_padded[:m, :n] = x
        x_view = x_padded.view(-1, 128, x_padded.size(1) // 128, 128)
        x_amax = x_view.abs().float().amax(dim=(1, 3), keepdim=True).clamp(1e-4)
        sf = ceil_to_ue8m0(x_amax / 448.0)
        x_scaled = (x_view * (1.0 / sf)).to(torch.float8_e4m3fn)
        return x_scaled.view_as(x_padded)[:m, :n].contiguous(), sf.view(
            x_view.size(0), x_view.size(2))
    else:
        g, m, n = x.shape
        x_padded = torch.zeros((g, align(m, 128), align(n, 128)),
                               dtype=x.dtype,
                               device=x.device)
        x_padded[:, :m, :n] = x
        x_view = x_padded.view(g, -1, 128, x_padded.size(-1) // 128, 128)
        x_amax = x_view.abs().float().amax(dim=(2, 4), keepdim=True).clamp(1e-4)
        sf = ceil_to_ue8m0(x_amax / 448.0)
        x_scaled = (x_view * (1.0 / sf)).to(torch.float8_e4m3fn)
        return x_scaled.view_as(x_padded)[:, :m, :n].contiguous(), sf.view(
            x_view.size(0), x_view.size(1), x_view.size(3))


def resmooth_to_fp8_e8m0(weight: torch.Tensor,
                         sf: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    weight = weight.cuda()
    sf = sf.cuda()
    if weight.dim() == 2:
        x = weight.float() * sf.repeat_interleave(128, dim=0).repeat_interleave(
            128, dim=1)[:weight.shape[0], :weight.shape[1]]
    else:
        x = weight.float() * sf.repeat_interleave(128, dim=1).repeat_interleave(
            128, dim=2)[:weight.shape[0], :weight.shape[1], :weight.shape[2]]
    return per_block_cast_to_fp8_e8m0(x)


def get_m_alignment_for_contiguous_layout():
    return 128


def get_tma_aligned_size(x: int, element_size: int) -> int:
    tma_alignment_bytes = 16
    assert tma_alignment_bytes % element_size == 0
    alignment = tma_alignment_bytes // element_size
    return align(x, alignment)


def get_col_major_tma_aligned_packed_tensor(x: torch.Tensor) -> torch.Tensor:
    # NOTES: for the extreme performance, you may rewrite/fuse this function in CUDA
    assert x.dtype == torch.float and x.dim() in (2, 3)

    # First, convert into UE8M0 `uint8_t`
    ue8m0_tensor = (x.view(torch.int) >> 23).to(torch.uint8)

    # Second, make padded packed tensors
    mn, k = x.shape[-2], x.shape[-1]
    remove_dim = False
    if x.dim() == 2:
        x, remove_dim = x.unsqueeze(0), True
    b = x.shape[0]
    aligned_mn = get_tma_aligned_size(mn, 4)
    aligned_k = align(k, 4)
    padded = torch.zeros((b, aligned_mn, aligned_k),
                         device=x.device,
                         dtype=torch.uint8)
    padded[:, :mn, :k] = ue8m0_tensor
    padded = padded.view(-1).view(dtype=torch.int).view(b, aligned_mn,
                                                        aligned_k // 4)

    # Finally, transpose
    transposed = torch.transpose(
        torch.empty((b, aligned_k // 4, aligned_mn),
                    device=x.device,
                    dtype=torch.int), 1, 2)
    transposed[:, :, :] = padded
    aligned_x = transposed[:, :mn, :]
    return aligned_x.squeeze(0) if remove_dim else aligned_x


def check_sf_layout(sf: torch.Tensor,
                    mn: int,
                    k: int,
                    gran: Tuple[int, int],
                    num_groups: Optional[int],
                    tma_stride_check: bool = False,
                    type_check: Optional[torch.dtype] = None) -> torch.Tensor:
    # Type check
    if type_check is not None:
        assert sf.dtype == type_check

    # Always do shape checks
    assert sf.dtype in (torch.float, torch.int)
    assert sf.dim() == int(num_groups is not None) + 2
    if num_groups is not None:
        assert sf.size(-3) == num_groups
    assert sf.size(-2) == ceil_div(mn, gran[0])
    assert sf.size(-1) == ceil_div(
        k, gran[1] * (1 if sf.dtype == torch.float else 4))

    # TMA stride checks: TMA aligned and MN-major
    if tma_stride_check:
        if num_groups is not None:
            assert sf.stride(-3) == sf.stride(-1) * sf.size(-1)
        assert sf.stride(-2) == 1
        assert sf.stride(-1) == get_tma_aligned_size(mn, sf.element_size())

    return sf


@nvtx_range("[DG] transform_sf_into_required_layout")
@torch.compile(dynamic=True)
def transform_sf_into_required_layout(sf: torch.Tensor,
                                      mn: int,
                                      k: int,
                                      recipe: Tuple[int, int, int],
                                      num_groups: Optional[int] = None,
                                      is_sfa: bool = False):
    gran = (recipe[0 if is_sfa else 1], recipe[2])

    should_skip_transform = ((sf.dtype == torch.int and gran == (1, 128))
                             or (sf.dtype == torch.int and gran == (128, 128)))

    if not should_skip_transform:
        # Pre-transform checks
        check_sf_layout(sf, mn=mn, k=k, gran=gran, num_groups=num_groups)

    # (FP32, 1, 128) on SM100: transform to (INT, 1, 128), TMA-aligned and MN-major
    if sf.dtype == torch.float and gran == (1, 128):
        sf = get_col_major_tma_aligned_packed_tensor(sf)
        return check_sf_layout(sf,
                               mn=mn,
                               k=k,
                               gran=(1, 128),
                               num_groups=num_groups,
                               tma_stride_check=True,
                               type_check=torch.int)

    # (FP32, 128, 128) on SM100: transform to (INT, 1, 128), TMA-aligned and MN-major
    if sf.dtype == torch.float and gran == (128, 128):
        sf = sf.index_select(-2, torch.arange(mn, device=sf.device) // 128)
        sf = get_col_major_tma_aligned_packed_tensor(sf)
        return check_sf_layout(sf,
                               mn=mn,
                               k=k,
                               gran=(1, 128),
                               num_groups=num_groups,
                               tma_stride_check=True,
                               type_check=torch.int)

    if should_skip_transform:
        # TODO: add transpose kernel if SF layout is not satisfied
        return check_sf_layout(sf,
                               mn=mn,
                               k=k,
                               gran=(1, 128),
                               num_groups=num_groups,
                               tma_stride_check=True,
                               type_check=torch.int)

    assert False, f'Unknown cases: {sf.dtype=}, {gran=}'
