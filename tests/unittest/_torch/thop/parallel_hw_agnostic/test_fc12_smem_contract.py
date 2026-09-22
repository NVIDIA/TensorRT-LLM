# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host-side contracts of the Rubin fused FC1+FC2 MoE kernel.

Every check runs on the host: task-space derivation, shared-memory stage
selection and byte layout, and the CuTe views the kernel builds from them.
The last test traces the kernel's layout setup under ``cute.compile`` with an
explicit ``sm_107a`` target, so no Rubin GPU is required.
"""

import inspect

import pytest

cutlass = pytest.importorskip("cutlass")
cute = pytest.importorskip("cutlass.cute")

from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_RUBIN_AVAILABLE  # noqa: E402

pytestmark = pytest.mark.skipif(
    not IS_CUTLASS_DSL_RUBIN_AVAILABLE,
    reason="the installed CuTe DSL package has no Rubin (SM107) helpers",
)

if IS_CUTLASS_DSL_RUBIN_AVAILABLE:
    from tensorrt_llm._torch.cute_dsl_kernels.rubin.moe import (
        rubin_contiguous_grouped_blockscaled_gemm_fused_fc12 as fused_fc12,
    )
    from tensorrt_llm._torch.cute_dsl_kernels.rubin.moe.rubin_contiguous_grouped_blockscaled_gemm_fused_fc12 import (  # noqa: E501
        END_PHASE,
        FC1_PHASE,
        FC2_PHASE,
        Fc12Task,
        FusedSmemStageBytes,
        FusedSmemStageConfig,
        SeparatePhaseSmemStageBytes,
        SeparatePhaseSmemStageConfig,
        Sm107BlockScaledContiguousGroupedGemmFusedFc12Kernel,
        derive_fc12_cta_task_stream,
        derive_fc12_task_shape,
        derive_fused_ab_mbarrier_array_count,
        select_fused_smem_stages,
        select_separate_phase_smem_stages,
    )

SMEM_CAPACITY = 334_848

# DS-V4-Pro FC1 (N = 2 * intermediate) and FC2 (N = hidden) problem sizes used
# by every task-space check below.
FC1_GEMM_N = 6_144
FC2_N = 7_168


def _supported_problem() -> dict:
    return dict(
        a_dtype=cutlass.Float4E2M1FN,
        b_dtype=cutlass.Float4E2M1FN,
        sf_dtype=cutlass.Float8E4M3FN,
        sf_vec_size=16,
        fc1_c_dtype=cutlass.Float4E2M1FN,
        fc2_c_dtype=cutlass.BFloat16,
        mma_inst_shape=(128, 128, 128),
        mma_tiler=(128, 128, 256),
        cluster_shape_mn=(1, 1),
        fc1_gemm_shape=(1_536, FC1_GEMM_N, 7_168, 12),
        fc2_gemm_shape=(1_536, FC2_N, 3_072, 12),
        a_major="k",
        b_major="k",
        fc1_c_major="n",
        fc2_c_major="n",
    )


def _common_kernel_args() -> dict:
    return dict(
        sf_vec_size=16,
        mma_inst_shape=(128, 128, 128),
        mma_tiler=(128, 128, 256),
        cluster_shape_mn=(1, 1),
        vectorized_f32=True,
        topk=1,
    )


def test_constructor_and_scheduler_drop_legacy_parameters():
    constructor_parameters = inspect.signature(
        Sm107BlockScaledContiguousGroupedGemmFusedFc12Kernel.__init__
    ).parameters
    assert "a_path" not in constructor_parameters
    assert "fc2_n" not in constructor_parameters
    assert "raster_along_m" not in constructor_parameters

    scheduler_parameters = inspect.signature(
        Sm107BlockScaledContiguousGroupedGemmFusedFc12Kernel._compute_tile_sched_params
    ).parameters
    assert "raster_along_m" not in scheduler_parameters

    kernel_1cta = Sm107BlockScaledContiguousGroupedGemmFusedFc12Kernel(**_common_kernel_args())
    assert not hasattr(kernel_1cta, "raster_along_m")
    with pytest.raises(TypeError, match="raster_along_m"):
        Sm107BlockScaledContiguousGroupedGemmFusedFc12Kernel(
            **_common_kernel_args(), raster_along_m=True
        )


def test_l2_atomic_descriptor_bounds():
    coordinate_count_limit = 1 << 16
    fused_fc12.validate_l2_atomic_descriptor_bounds(
        phase="FC1",
        gemm_shape=(
            coordinate_count_limit * 128,
            coordinate_count_limit * 128,
            coordinate_count_limit,
        ),
        cta_tile_shape_mnk=(128, 128, 256),
    )
    for mode, out_of_bounds_shape in (
        ("M", ((coordinate_count_limit + 1) * 128, 128, 1)),
        ("N", (128, (coordinate_count_limit + 1) * 128, 1)),
        ("L", (128, 128, coordinate_count_limit + 1)),
    ):
        with pytest.raises(ValueError, match=mode):
            fused_fc12.validate_l2_atomic_descriptor_bounds(
                phase="FC1",
                gemm_shape=out_of_bounds_shape,
                cta_tile_shape_mnk=(128, 128, 256),
            )


def test_can_implement_accepts_the_supported_problem_and_rejects_variants():
    supported_problem = _supported_problem()
    can_implement = Sm107BlockScaledContiguousGroupedGemmFusedFc12Kernel.can_implement
    assert can_implement(**supported_problem)
    # FC2 N not a multiple of the tile.
    assert not can_implement(**(supported_problem | {"fc2_gemm_shape": (1_536, 7_040, 3_072, 12)}))
    # Unsupported FC2 output dtype.
    assert not can_implement(**(supported_problem | {"fc2_c_dtype": cutlass.Float16}))
    # 2-CTA tile with a 1x1 cluster.
    assert not can_implement(**(supported_problem | {"mma_tiler": (256, 128, 256)}))


@pytest.mark.parametrize(
    "mma_n,fc1_tiles_per_m,fc2_tiles_per_m",
    [(128, 48, 56), (256, 24, 28)],
    ids=["1cta", "2cta"],
)
def test_task_shape(mma_n, fc1_tiles_per_m, fc2_tiles_per_m):
    task_shape = derive_fc12_task_shape(fc1_gemm_n=FC1_GEMM_N, fc2_n=FC2_N, mma_n=mma_n)
    assert task_shape.fc1_tiles_per_m == fc1_tiles_per_m
    assert task_shape.fc2_tiles_per_m == fc2_tiles_per_m


def test_task_shape_rejects_non_divisible_fc1_n():
    with pytest.raises(ValueError, match="FC1 GEMM N"):
        derive_fc12_task_shape(fc1_gemm_n=FC1_GEMM_N + 6, fc2_n=FC2_N, mma_n=128)


def test_cta_task_streams_cover_every_tile_once_and_finish_fc1_first():
    task_shape = derive_fc12_task_shape(fc1_gemm_n=FC1_GEMM_N, fc2_n=FC2_N, mma_n=128)
    num_m_tiles = 12
    num_resident_ctas = 148
    cta_streams = tuple(
        derive_fc12_cta_task_stream(
            num_m_tiles=num_m_tiles,
            task_shape=task_shape,
            num_resident_ctas=num_resident_ctas,
            cta_idx=cta_idx,
        )
        for cta_idx in range(num_resident_ctas)
    )

    for stream in cta_streams:
        assert stream[-1] == Fc12Task(phase=END_PHASE, m_tile=-1, n_tile=-1)
        seen_fc2 = False
        for task in stream[:-1]:
            assert task.phase in (FC1_PHASE, FC2_PHASE)
            if task.phase == FC2_PHASE:
                seen_fc2 = True
            else:
                assert not seen_fc2, "FC1 task published after FC2 phase started"

    fc1_task_list = [
        (task.m_tile, task.n_tile)
        for stream in cta_streams
        for task in stream
        if task.phase == FC1_PHASE
    ]
    fc2_task_list = [
        (task.m_tile, task.n_tile)
        for stream in cta_streams
        for task in stream
        if task.phase == FC2_PHASE
    ]
    expected_fc1 = {
        (m_tile, n_tile)
        for m_tile in range(num_m_tiles)
        for n_tile in range(task_shape.fc1_tiles_per_m)
    }
    expected_fc2 = {
        (m_tile, n_tile)
        for m_tile in range(num_m_tiles)
        for n_tile in range(task_shape.fc2_tiles_per_m)
    }
    assert len(fc1_task_list) == len(expected_fc1)
    assert len(fc2_task_list) == len(expected_fc2)
    assert set(fc1_task_list) == expected_fc1
    assert set(fc2_task_list) == expected_fc2


def test_2cta_readiness_and_mbarrier_array_counts():
    kernel_2cta = Sm107BlockScaledContiguousGroupedGemmFusedFc12Kernel(
        **(
            _common_kernel_args()
            | {
                "mma_inst_shape": (256, 256, 128),
                "mma_tiler": (256, 256, 256),
                "cluster_shape_mn": (2, 1),
            }
        )
    )
    assert kernel_2cta.use_2cta_instrs

    ready_expected = fused_fc12.derive_fc1_ready_expected
    assert ready_expected(fc1_gemm_n=FC1_GEMM_N, mma_n=256, mma_cta_group_size=2) == 48
    with pytest.raises(ValueError, match="must be 1 or 2"):
        ready_expected(fc1_gemm_n=FC1_GEMM_N, mma_n=256, mma_cta_group_size=3)

    assert derive_fused_ab_mbarrier_array_count(1) == 4
    assert derive_fused_ab_mbarrier_array_count(2) == 5


_COMMON_STAGE_BYTES = dict(
    a_per_ab_stage=16_384,
    sfa_per_ab_stage=2_048,
    fc1_c_per_stage=4_096,
    metadata=2_048,
    header_fixed=156,
    header_per_ab_stage=64,
    alignment=1_024,
)


def test_common_smem_stage_selection():
    stages_128x128 = FusedSmemStageBytes(
        b_per_ab_stage=16_384,
        sfb_per_ab_stage=2_048,
        fc2_c=34_800,
        **_COMMON_STAGE_BYTES,
    )
    assert stages_128x128.total(ab_stages=8, fc1_c_stages=9) == SMEM_CAPACITY
    assert select_fused_smem_stages(
        capacity=SMEM_CAPACITY,
        preferred_ab=8,
        preferred_fc1_c=9,
        minimum_fc1_c=2,
        stage_bytes=stages_128x128,
    ) == FusedSmemStageConfig(ab=8, fc1_c=9)

    stages_128x256 = FusedSmemStageBytes(
        b_per_ab_stage=32_768,
        sfb_per_ab_stage=4_096,
        fc2_c=67_568,
        **_COMMON_STAGE_BYTES,
    )
    assert stages_128x256.total(ab_stages=5, fc1_c_stages=14) == 347_136
    assert stages_128x256.total(ab_stages=5, fc1_c_stages=2) == 347_136
    assert stages_128x256.total(ab_stages=4, fc1_c_stages=14) == 291_840
    assert select_fused_smem_stages(
        capacity=SMEM_CAPACITY,
        preferred_ab=5,
        preferred_fc1_c=14,
        minimum_fc1_c=2,
        stage_bytes=stages_128x256,
    ) == FusedSmemStageConfig(ab=4, fc1_c=14)

    with pytest.raises(ValueError, match="no fused FC12"):
        select_fused_smem_stages(
            capacity=1_024,
            preferred_ab=1,
            preferred_fc1_c=2,
            minimum_fc1_c=2,
            stage_bytes=stages_128x256,
        )


def _separate_stage_bytes(**overrides):
    values = dict(
        a_per_stage=16_384,
        b_per_stage=16_384,
        sfa_per_stage=2_048,
        sfb_per_stage=2_048,
        fc1_c_per_stage=4_096,
        fc2_c=34_800,
        metadata=2_048,
        header_fixed=156,
        fc1_header_per_stage=32,
        fc2_header_per_stage=32,
        alignment=1_024,
    )
    values.update(overrides)
    return SeparatePhaseSmemStageBytes(**values)


def test_separate_phase_stages_256x256_overlay_fc2_c_on_the_fc1_operand_tail():
    stage_bytes = _separate_stage_bytes(sfb_per_stage=4_096, fc2_c=67_568, fc1_header_per_stage=48)
    selected = select_separate_phase_smem_stages(
        capacity=SMEM_CAPACITY,
        preferred_fc1_ab=8,
        preferred_fc2_ab=6,
        preferred_fc1_c=5,
        minimum_fc1_c=2,
        stage_bytes=stage_bytes,
    )
    assert selected == SeparatePhaseSmemStageConfig(fc1_ab=8, fc2_ab=6, fc1_c=5, overlay_fc2_c=True)
    assert (
        stage_bytes.overlay_bytes(fc1_ab_stages=selected.fc1_ab, fc2_ab_stages=selected.fc2_ab)
        == 73_728
    )
    assert (
        stage_bytes.total(
            fc1_ab_stages=selected.fc1_ab,
            fc2_ab_stages=selected.fc2_ab,
            fc1_c_stages=selected.fc1_c,
            overlay_fc2_c=selected.overlay_fc2_c,
        )
        == SMEM_CAPACITY
    )

    layout = fused_fc12.derive_separate_phase_smem_layout(
        stage_config=selected, stage_bytes=stage_bytes
    )
    # A occupies one contiguous low-address allocation. B and SFB are paired
    # per physical stage after A; FC2 skips the first two B/SFB slots because
    # they are part of the contiguous 72 KiB FC2 C alias beginning at A6.
    assert layout.a_smem_offset_bytes == 0
    assert layout.a_smem_alloc_bytes == 131_072
    assert layout.fc1_b_smem_offset_bytes == 131_072
    assert layout.fc1_sfb_smem_offset_bytes == 147_456
    assert layout.fc2_b_smem_offset_bytes == 172_032
    assert layout.fc2_sfb_smem_offset_bytes == 188_416
    assert layout.fc2_bsfb_stage_offset == 2
    assert layout.fc2_c_smem_offset_bytes == 98_304
    assert layout.fc2_c_overlay_bytes == 73_728
    assert layout.mainloop_smem_alloc_bytes == 294_912
    # The 8/6-stage path keeps its paired B/SFB strides.
    assert layout.b_stage_stride_bytes == 20_480
    assert layout.sfb_stage_stride_bytes == 20_480


@pytest.mark.parametrize(
    "stage_overrides,preferred,expected_stages,expected_layout",
    [
        pytest.param(
            {},
            dict(preferred_fc1_ab=8, preferred_fc2_ab=8, preferred_fc1_c=9),
            SeparatePhaseSmemStageConfig(fc1_ab=8, fc2_ab=8, fc1_c=9, overlay_fc2_c=False),
            dict(
                expected_b_offset=131_072,
                expected_sfb_offset=262_144,
                expected_mainloop_bytes=278_528,
            ),
            id="128x128",
        ),
        pytest.param(
            # One FC1-only tail stage holds 52 KiB, which cannot fit the
            # 67,568-byte FC2 C tile, so the common four-stage depth and the
            # standalone epilogue allocation are kept.
            dict(b_per_stage=32_768, sfb_per_stage=4_096, fc2_c=67_568),
            dict(preferred_fc1_ab=5, preferred_fc2_ab=4, preferred_fc1_c=14),
            SeparatePhaseSmemStageConfig(fc1_ab=4, fc2_ab=4, fc1_c=14, overlay_fc2_c=False),
            dict(
                expected_b_offset=65_536,
                expected_sfb_offset=196_608,
                expected_mainloop_bytes=212_992,
            ),
            id="128x256",
        ),
        pytest.param(
            dict(b_per_stage=8_192, header_fixed=220, fc1_header_per_stage=48),
            dict(preferred_fc1_ab=11, preferred_fc2_ab=10, preferred_fc1_c=4),
            SeparatePhaseSmemStageConfig(fc1_ab=10, fc2_ab=10, fc1_c=4, overlay_fc2_c=False),
            dict(
                expected_b_offset=163_840,
                expected_sfb_offset=245_760,
                expected_mainloop_bytes=266_240,
            ),
            id="256x128",
        ),
    ],
)
def test_separate_phase_stages_without_overlay(
    stage_overrides, preferred, expected_stages, expected_layout
):
    stage_bytes = _separate_stage_bytes(**stage_overrides)
    selected = select_separate_phase_smem_stages(
        capacity=SMEM_CAPACITY, minimum_fc1_c=2, stage_bytes=stage_bytes, **preferred
    )
    assert selected == expected_stages
    _check_non_overlay_smem_layout(
        stage_bytes=stage_bytes, ab_stages=expected_stages.fc1_ab, **expected_layout
    )


def test_separate_phase_128x128_fills_the_smem_capacity_exactly():
    assert (
        _separate_stage_bytes().total(
            fc1_ab_stages=8, fc2_ab_stages=8, fc1_c_stages=9, overlay_fc2_c=False
        )
        == SMEM_CAPACITY
    )


def test_separate_phase_layout_keeps_alignment_padding():
    # Byte counts that are not multiples of the buffer alignment must still
    # produce aligned, non-interleaved operand regions.
    stage_bytes = _separate_stage_bytes(
        a_per_stage=768,
        b_per_stage=1_536,
        sfa_per_stage=256,
        sfb_per_stage=128,
        fc1_c_per_stage=512,
        fc2_c=1_024,
        metadata=64,
        header_fixed=96,
    )
    _check_non_overlay_smem_layout(
        stage_bytes=stage_bytes,
        ab_stages=3,
        expected_b_offset=3_072,
        expected_sfb_offset=8_192,
        expected_mainloop_bytes=9_216,
    )
    assert (
        stage_bytes.total(fc1_ab_stages=3, fc2_ab_stages=3, fc1_c_stages=2, overlay_fc2_c=False)
        == 13_312
    )


def _check_non_overlay_smem_layout(
    *,
    stage_bytes,
    ab_stages: int,
    expected_b_offset: int,
    expected_sfb_offset: int,
    expected_mainloop_bytes: int,
) -> None:
    """Without FC2 C aliasing, SFB must follow all B stages, not interleave them."""
    layout = fused_fc12.derive_separate_phase_smem_layout(
        stage_config=SeparatePhaseSmemStageConfig(
            fc1_ab=ab_stages, fc2_ab=ab_stages, fc1_c=2, overlay_fc2_c=False
        ),
        stage_bytes=stage_bytes,
    )
    assert layout.fc1_sfb_smem_offset_bytes == expected_sfb_offset
    assert layout.a_smem_offset_bytes == 0
    assert layout.fc1_b_smem_offset_bytes == expected_b_offset
    assert layout.fc2_b_smem_offset_bytes == expected_b_offset
    assert layout.fc2_sfb_smem_offset_bytes == expected_sfb_offset
    assert layout.b_stage_stride_bytes == stage_bytes.b_per_stage
    assert layout.sfb_stage_stride_bytes == stage_bytes.sfb_per_stage
    assert layout.mainloop_smem_alloc_bytes == expected_mainloop_bytes
    assert layout.fc2_bsfb_stage_offset == 0
    assert layout.fc2_c_overlay_bytes == 0


# (tile_m, tile_n) -> (B stride in FP4 elements, SFB stride in FP8 elements)
_EXPECTED_STAGED_STRIDES = {
    (128, 128): (32_768, 2_048),
    (128, 256): (65_536, 4_096),
    (256, 128): (16_384, 2_048),
    (256, 256): (40_960, 20_480),
}


def _check_constructed_smem_views() -> None:
    """The staged B/SFB CuTe views must carry the byte layout's strides."""
    for (tile_m, tile_n), (b_stride, sfb_stride) in _EXPECTED_STAGED_STRIDES.items():
        kernel = Sm107BlockScaledContiguousGroupedGemmFusedFc12Kernel(
            sf_vec_size=16,
            mma_inst_shape=(tile_m, tile_n, 128),
            mma_tiler=(tile_m, tile_n, 256),
            cluster_shape_mn=(tile_m // 128, 1),
            vectorized_f32=True,
            topk=6,
            use_pdl=False,
            scheduler="l2_atomic",
        )
        kernel.a_dtype = cutlass.Float4E2M1FN
        kernel.b_dtype = cutlass.Float4E2M1FN
        kernel.sf_dtype = cutlass.Float8E4M3FN
        kernel.fc1_c_dtype = cutlass.Float4E2M1FN
        kernel.fc2_c_dtype = cutlass.BFloat16
        kernel.a_major_mode = cute.nvgpu.OperandMajorMode.K
        kernel.b_major_mode = cute.nvgpu.OperandMajorMode.K
        kernel.fc1_c_layout = cutlass.tensor_utils.LayoutEnum.ROW_MAJOR
        kernel.fc2_c_layout = cutlass.tensor_utils.LayoutEnum.ROW_MAJOR
        kernel._setup_attributes()
        assert kernel.fc1_b_smem_layout_staged.outer.stride[-1] == b_stride
        assert kernel.fc2_b_smem_layout_staged.outer.stride[-1] == b_stride
        assert kernel.fc1_sfb_smem_layout_staged.stride[-1] == sfb_stride
        assert kernel.fc2_sfb_smem_layout_staged.stride[-1] == sfb_stride


@pytest.mark.timeout(600)
def test_constructed_smem_views_in_dsl_context():
    @cute.jit
    def check_smem_views_in_dsl_context() -> None:
        _check_constructed_smem_views()

    # The layout target is pinned so this traces without a visible Rubin GPU;
    # every assertion runs during tracing and no kernel is launched.
    cute.compile(check_smem_views_in_dsl_context, options="--gpu-arch=sm_107a")
