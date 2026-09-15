# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import subprocess
import sys
import textwrap
from importlib import import_module

import pytest
import torch

cute = pytest.importorskip("cutlass.cute")

from tensorrt_llm._torch.visual_gen.attention_backend import flash_attn4, parallel  # noqa: E402
from tensorrt_llm._torch.visual_gen.attention_backend.flash_attn4 import (  # noqa: E402
    _install_cutlass_dsl_compatibility,
    _install_flash_attn_tile_scheduler_compatibility,
)


def test_cutlass_dsl_47_moved_names_are_restored(monkeypatch):
    monkeypatch.delattr(cute.core, "ThrCopy", raising=False)
    monkeypatch.delattr(cute.core, "ThrMma", raising=False)
    monkeypatch.delattr(cute, "make_fragment", raising=False)

    _install_cutlass_dsl_compatibility()

    assert cute.core.ThrCopy is cute.ThrCopy
    assert cute.core.ThrMma is cute.ThrMma
    assert cute.make_fragment is cute.make_rmem_tensor


def test_cutlass_dsl_existing_names_are_preserved(monkeypatch):
    existing_thr_copy = object()
    existing_thr_mma = object()
    existing_make_fragment = object()
    monkeypatch.setattr(cute.core, "ThrCopy", existing_thr_copy)
    monkeypatch.setattr(cute.core, "ThrMma", existing_thr_mma)
    monkeypatch.setattr(cute, "make_fragment", existing_make_fragment)

    _install_cutlass_dsl_compatibility()

    assert cute.core.ThrCopy is existing_thr_copy
    assert cute.core.ThrMma is existing_thr_mma
    assert cute.make_fragment is existing_make_fragment


def test_cutlass_dsl_47_aliases_allow_fa4_interface_import() -> None:
    _install_cutlass_dsl_compatibility()
    interface = import_module("flash_attn.cute.interface")

    assert callable(interface._flash_attn_fwd)
    assert callable(interface.flash_attn_combine)
    assert callable(flash_attn4._flash_attn_fwd)
    assert callable(parallel._flash_attn_combine)


def test_fa4_work_tile_info_survives_cutlass_task_scheduling_import() -> None:
    task_scheduling = pytest.importorskip("cutlass.experimental.task_scheduling")
    tile_scheduler = pytest.importorskip("flash_attn.cute.tile_scheduler")
    import cutlass
    from cutlass.cutlass_dsl import Boolean
    from cutlass.utils.static_persistent_tile_scheduler import WorkTileInfo as CutlassWorkTileInfo

    del task_scheduling
    _install_flash_attn_tile_scheduler_compatibility()
    tile_idx = (cutlass.Int32(1), cutlass.Int32(2), cutlass.Int32(3), cutlass.Int32(0))

    # FA4's four-axis coordinate must construct whether or not the task-scheduling
    # import rewrote the shared CUTLASS parent class.
    fa4_tile = tile_scheduler.WorkTileInfo(tile_idx, Boolean(True))

    assert "__init__" in vars(tile_scheduler.WorkTileInfo)
    assert fa4_tile.tile_idx == tile_idx
    assert bool(fa4_tile.is_valid_tile)
    assert issubclass(tile_scheduler.WorkTileInfo, CutlassWorkTileInfo)


def test_fa4_work_tile_info_survives_task_scheduling_imported_first() -> None:
    pytest.importorskip("cutlass.experimental.task_scheduling")
    pytest.importorskip("flash_attn.cute.tile_scheduler")
    script = textwrap.dedent(
        """
        import cutlass
        import cutlass.experimental.task_scheduling  # noqa: F401
        from cutlass.cutlass_dsl import Boolean

        import tensorrt_llm._torch.visual_gen.attention_backend.flash_attn4  # noqa: F401
        from flash_attn.cute.tile_scheduler import WorkTileInfo

        tile_idx = (cutlass.Int32(1), cutlass.Int32(2), cutlass.Int32(3), cutlass.Int32(0))
        tile = WorkTileInfo(tile_idx, Boolean(True))
        assert tile.tile_idx == tile_idx
        assert bool(tile.is_valid_tile)
        print("fa4-work-tile-info-ok")
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=900, check=False
    )

    assert result.returncode == 0, result.stderr[-4000:]
    assert result.stdout.strip().endswith("fa4-work-tile-info-ok")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FA4 kernel tracing requires CUDA")
def test_fa4_kernel_traces_after_task_scheduling_imported_first() -> None:
    """Compile and run FA4 after the CUTLASS task-scheduling import.

    Construction alone does not reach ``__extract_mlir_values__`` or FA4's
    ``__new_from_mlir_values__``; only kernel tracing does. The forced split-KV
    case makes the fourth (split) coordinate non-zero, and the graph replay
    covers the captured path used by VisualGen.
    """
    pytest.importorskip("cutlass.experimental.task_scheduling")
    pytest.importorskip("flash_attn.cute.tile_scheduler")
    script = textwrap.dedent(
        """
        import cutlass.experimental.task_scheduling  # noqa: F401
        import torch

        from tensorrt_llm._torch.visual_gen.attention_backend.flash_attn4 import _flash_attn_fwd

        torch.manual_seed(0)
        # S_kv stays above FA4's short-KV short-circuit so num_splits=8 reaches the
        # split-KV kernel; S_q is short so the split path is worth taking.
        B, S_q, S_kv, H, d_h = 1, 64, 4096, 8, 128
        q, k, v = (
            torch.randn(B, s, H, d_h, dtype=torch.bfloat16, device="cuda")
            for s in (S_q, S_kv, S_kv)
        )

        def fa4(num_splits):
            out, lse, *_ = _flash_attn_fwd(
                q,
                k,
                v,
                seqused_k=None,
                softmax_scale=d_h**-0.5,
                causal=False,
                window_size_left=None,
                window_size_right=None,
                learnable_sink=None,
                softcap=0.0,
                pack_gqa=None,
                mask_mod=None,
                block_sparse_tensors=None,
                return_lse=True,
                num_splits=num_splits,
            )
            return out, lse

        ref_out, ref_lse = fa4(1)
        split_out, split_lse = fa4(8)
        torch.testing.assert_close(split_out, ref_out, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(split_lse, ref_lse, rtol=1e-3, atol=1e-3)

        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            fa4(8)
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_out, graph_lse = fa4(8)
        for t in (q, k, v):
            t.copy_(torch.randn_like(t))
        graph.replay()
        ref_out, ref_lse = fa4(1)
        torch.testing.assert_close(graph_out, ref_out, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(graph_lse, ref_lse, rtol=1e-3, atol=1e-3)
        print("fa4-kernel-trace-ok")
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=900, check=False
    )

    assert result.returncode == 0, result.stderr[-4000:]
    assert result.stdout.strip().endswith("fa4-kernel-trace-ok")
