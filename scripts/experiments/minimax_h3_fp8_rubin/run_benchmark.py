# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in uniform static-FP8 H3 fusion benchmark on SM107, with50-step requests."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PRESET = {
    "attention_fmha": "trtllm-gen-fp16sm",
    "skip_corr_threshold": 0.0,
    "fp8_scale_fusion": "qkstatic",
    "fp8_fusion_block_h": 8,
    "fp8_fusion_warps": 2,
    "grouped_cast": True,
    "vamax_fold": False,
    "fused_quant": False,
    "out_quant": False,
    "out_a2a_fp4": False,
    "vae_fp8": True,
    "vae_fp8_tiled": True,
    "vae_fused_norm": True,
    "vae_graph_max_shapes": 2,
    "text_encoder_graph": True,
    "preloop_device_plan": True,
    "preloop_async_copies": True,
    "preloop_noise_early": True,
}
REQUEST_SHAS = {
    "waiter": "d6781453c91097a294becf64b5b1a37262212aae77b27067e2156310a9007435",
    "cycling": "adada0baf8f7ec70af2d775ac23f682e4e1aebf8589dbb63f17d89cdcf2090bc",
}


def _sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _write(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, default=str, allow_nan=False) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--scene", choices=REQUEST_SHAS, default="waiter")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    request_path = ROOT / "requests" / (args.scene + ".json")
    if _sha(request_path) != REQUEST_SHAS[args.scene]:
        raise ValueError("The benchmark requires an unchanged50-step fixture")
    request = json.loads(request_path.read_text())
    if request["num_inference_steps"] != 50:
        raise ValueError("The benchmark requires50 steps")
    pins = json.loads((ROOT / "source-pins.json").read_text())
    for relative, expected in pins["files"].items():
        if _sha(ROOT / relative) != expected:
            raise ValueError("Benchmark source changed: " + relative)
    if args.validate_only:
        print(
            json.dumps(
                {
                    "CPU_only": True,
                    "GPU_calls": 0,
                    "scene": args.scene,
                    "requested_steps": 50,
                    "expected_forward_calls": 49,
                }
            )
        )
        return
    os.environ.setdefault("NVIDIA_TF32_OVERRIDE", "0")
    os.environ.setdefault("TLLM_DISABLE_MPI", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "8")
    os.environ.setdefault("MKL_NUM_THREADS", "8")
    if os.environ["NVIDIA_TF32_OVERRIDE"] != "0":
        raise ValueError("The reproducible benchmark requires NVIDIA_TF32_OVERRIDE=0")
    args.output.mkdir(parents=True, exist_ok=False)
    for name in ("fusion_round1", "hotspot_round2", "long_round3", "long_round4", "long_round5"):
        sys.path.insert(0, str(ROOT / name))
    import torch
    from phase_timer import EventTimer

    from tensorrt_llm._torch.modules.gated_mlp import GatedMLP
    from tensorrt_llm._torch.visual_gen.models.minimax_h3 import (
        pipeline_minimax_h3,
        scenarios,
        transformer_minimax_h3,
    )
    from tensorrt_llm._torch.visual_gen.models.minimax_h3.minimax_h3_nvfp4__sm_107__u1_u4 import (
        config,
    )
    from tensorrt_llm._torch.visual_gen.pipeline_loader import PipelineLoader
    from tensorrt_llm.quantization.mode import QuantAlgo
    from tensorrt_llm.visual_gen.args import VisualGenArgs

    torch.set_num_threads(8)
    if (
        torch.cuda.get_device_capability() != (10, 7)
        or torch.cuda.get_device_properties(0).multi_processor_count != 212
    ):
        raise ValueError("The fused implementation requires SM107 with212 SMs")
    proof = json.loads((ROOT / "long_round5/round5_shared_forward_proof.json").read_text())
    if _sha(Path(transformer_minimax_h3.__file__)) != proof["native_file_sha256"]:
        raise ValueError("Use the H3 Python runtime matching this integration base; see README")
    original_resolve, original_init = scenarios.resolve_scenario, GatedMLP.__init__
    original_build = pipeline_minimax_h3.build_packed_sequence
    layouts = []

    def logged_build(*values: object, **kwargs: object) -> object:
        layout = original_build(*values, **kwargs)
        layouts.append(layout.sequence_length)
        return layout

    def resolve(family: type, cfg: object) -> type:
        if scenarios._transformer_quant_algo(cfg) != QuantAlgo.FP8 or cfg.dynamic_weight_quant:
            raise ValueError("Use the provided uniform static per-tensor FP8 checkpoint")
        if scenarios._ulysses_size(cfg) != 1 or cfg.mapping.tp_size != 1:
            raise ValueError("The full fusion stack currently supports Ulysses=TP=1")
        qac = cfg.attention.quant_attention_config
        if cfg.attention.backend != "CUDNN" or (qac.qk_dtype, qac.v_dtype) != ("fp8", "fp8"):
            raise ValueError("The FP8 scenario requires the cuDNN adapter with FP8 QKV")
        config.apply(PRESET)
        from tensorrt_llm._torch.visual_gen.models.minimax_h3.minimax_h3_nvfp4__sm_107__u1_u4.pipeline import (
            MiniMaxH3Nvfp4Sm107U1U4Pipeline,
        )

        return MiniMaxH3Nvfp4Sm107U1U4Pipeline

    def compatible_init(
        self: object, *values: object, fuse_bf16_gate_up_swiglu: bool = False, **kwargs: object
    ) -> None:
        if fuse_bf16_gate_up_swiglu:
            raise ValueError("This runtime compatibility shim only supports the FP8 configuration")
        original_init(self, *values, **kwargs)

    runtime = None
    record = {
        "scene": args.scene,
        "request_sha256": REQUEST_SHAS[args.scene],
        "preset": PRESET,
        "requested_steps": 50,
        "finished": False,
        "model": str(args.model.resolve()),
        "checkpoint_config_sha256": _sha(args.model / "transformer/config.json"),
        "GPU_UUID": str(torch.cuda.get_device_properties(0).uuid),
        "torch": str(torch.__version__),
        "CUDA": torch.version.cuda,
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "generations": [],
        "semantic_quality": "pending",
        "source_pins": pins,
    }
    try:
        pipeline_minimax_h3.build_packed_sequence = logged_build
        scenarios.resolve_scenario = resolve
        if "fuse_bf16_gate_up_swiglu" not in inspect.signature(original_init).parameters:
            GatedMLP.__init__ = compatible_init
        vg = VisualGenArgs(
            model=str(args.model),
            attention_config={
                "backend": "CUDNN",
                "quant_attention_config": {"qk_dtype": "fp8", "v_dtype": "fp8"},
            },
            parallel_config={"ulysses_size": 1, "tp_size": 1, "parallel_vae_size": 1},
            torch_compile_config={"enable": True},
            cuda_graph_config={"enable": False},
            compilation_config={
                "skip_warmup": False,
                "resolutions": [[768, 1344]],
                "num_frames": [124],
            },
            cache_config=None,
        )
        pipeline = PipelineLoader(vg).load(skip_warmup=False)
        scenarios.resolve_scenario, GatedMLP.__init__ = original_resolve, original_init
        model = pipeline.transformer
        groups = []
        for block in model.transformer_blocks:
            for linear in (
                block.attn.qkv_proj,
                block.attn.to_out[0],
                block.ff.gate_up_proj,
                block.ff.down_proj,
            ):
                if (
                    not linear.has_fp8_qdq
                    or linear.weight.dtype != torch.float8_e4m3fn
                    or linear.input_scale is None
                ):
                    raise ValueError(
                        "The full fusion entry requires all200 main Linears to use static FP8"
                    )
                groups.append(list(linear.weight.shape))
        if len(groups) != 200:
            raise ValueError("Unexpected H3 main Linear count")
        timer = EventTimer()
        timer.wrap(model, "forward", "transformer_step", nvtx="h3/transformer")
        import fp8_producers
        import gemm_runtime
        import patch_runtime
        import residual_kernel
        import residual_runtime
        import round4_attention_runtime as attention
        from round4_runtime import Runtime as Round4Runtime
        from round5_runtime import Runtime as Round5Runtime

        fp8_producers.STATE.update(adaln=True, swiglu=True, tile=2048, warps=4)
        patch_runtime.install()
        gemm_runtime.install()
        gemm_runtime.STATE.update(mode=3, validate=False, fast_accum=True)
        residual_runtime.install(install_residual=True)
        residual_kernel.STATE.update(enabled=False, validate=False)
        base = Round4Runtime(pipeline)
        runtime = Round5Runtime(pipeline, timer=timer, cache=args.output / "build", base=base)
        runtime.configure("candidate", validate=False)
        pipeline.vae.enable_tiling(
            tile_sample_min_height=288,
            tile_sample_min_width=272,
            tile_sample_min_overlap_height=32,
            tile_sample_min_overlap_width=32,
        )
        torch._dynamo.reset()
        call = {
            key: request[key]
            for key in (
                "prompt",
                "seed",
                "height",
                "width",
                "num_frames",
                "frame_rate",
                "num_inference_steps",
            )
        }
        canonical = None
        for kind, index in [("warmup", 0)] + [("timed", i) for i in range(3)]:
            timer.records = []
            before = (
                base.g2.gateup.calls,
                base.g2.down.calls,
                attention.STATE["fused_calls"],
                runtime.quant.counts["fast_calls"],
                runtime.cast.counts["fast_calls"],
                base.g2.down.fallback_calls,
                attention.STATE["fallback_calls"],
                runtime.quant.counts["fallback_calls"],
                runtime.cast.counts["fallback_calls"],
                runtime.cast.counts["original_input_projection_calls"],
            )
            graphs = int(torch._dynamo.utils.counters.get("stats", {}).get("unique_graphs", 0))
            torch.cuda.synchronize()
            start = time.perf_counter()
            value = pipeline.forward(**call)
            torch.cuda.synchronize()
            row = {"kind": kind, "index": index, "wall_s": time.perf_counter() - start}
            record["generations"].append(row)
            _write(args.output / "results.json", record)
            events = timer.collect()
            delta = [
                after - prior
                for after, prior in zip(
                    (
                        base.g2.gateup.calls,
                        base.g2.down.calls,
                        attention.STATE["fused_calls"],
                        runtime.quant.counts["fast_calls"],
                        runtime.cast.counts["fast_calls"],
                        base.g2.down.fallback_calls,
                        attention.STATE["fallback_calls"],
                        runtime.quant.counts["fallback_calls"],
                        runtime.cast.counts["fallback_calls"],
                        runtime.cast.counts["original_input_projection_calls"],
                    ),
                    before,
                )
            ]
            row.update(
                actual_transformer_calls=len(events.get("transformer_step", [])),
                actual_packed_tokens=layouts[-1] if layouts else None,
                dispatch=delta,
                no_new_graphs=graphs
                == int(torch._dynamo.utils.counters.get("stats", {}).get("unique_graphs", 0)),
            )
            video, audio = (
                value.video.detach().contiguous().cpu(),
                value.audio.detach().contiguous().cpu(),
            )
            if (
                tuple(video.shape) != (1, 124, 768, 1344, 3)
                or video.dtype != torch.uint8
                or tuple(audio.shape) != (1, 2, 165600)
                or audio.dtype != torch.float32
                or not bool(torch.isfinite(audio).all())
                or int(value.audio_sample_rate) != 32000
            ):
                raise ValueError("Invalid H3 output")
            logical = {
                "video": hashlib.sha256(video.numpy().tobytes()).hexdigest(),
                "audio": hashlib.sha256(audio.numpy().tobytes()).hexdigest(),
            }
            if canonical is None:
                canonical = logical
                torch.save(
                    {
                        "video": video,
                        "audio": audio,
                        "frame_rate": 24,
                        "audio_sample_rate": int(value.audio_sample_rate),
                    },
                    args.output / "output.pt",
                )
                record["canonical_media_sha256"] = logical
                record["raw_output_sha256"] = _sha(args.output / "output.pt")
            row["matches_warmup"] = logical == canonical
            _write(args.output / "results.json", record)
            expected_tokens = 38169 if args.scene == "waiter" else 37820
            if (
                row["actual_transformer_calls"] != 49
                or row["actual_packed_tokens"] != expected_tokens
                or delta
                != [
                    2450,
                    2450,
                    2450,
                    2450,
                    49,
                    0,
                    0,
                    0,
                    0,
                    98,
                ]
            ):
                raise RuntimeError("Unexpected fusion dispatch; timing and raw output retained")
            del video, audio, value
        formal = [row for row in record["generations"] if row["kind"] == "timed"]
        record.update(
            finished=True,
            median_wall_s=statistics.median(row["wall_s"] for row in formal),
            stable_compilation=all(row["no_new_graphs"] for row in formal),
            bitwise_repeat_equal=all(row["matches_warmup"] for row in formal),
        )
        print(
            json.dumps(
                {
                    key: record[key]
                    for key in (
                        "finished",
                        "median_wall_s",
                        "stable_compilation",
                        "bitwise_repeat_equal",
                    )
                }
            )
        )
    finally:
        pipeline_minimax_h3.build_packed_sequence = original_build
        scenarios.resolve_scenario, GatedMLP.__init__ = original_resolve, original_init
        if runtime is not None:
            record["runtime_report"] = runtime.report()
            runtime.configure("native")
        _write(args.output / "results.json", record)


if __name__ == "__main__":
    main()
