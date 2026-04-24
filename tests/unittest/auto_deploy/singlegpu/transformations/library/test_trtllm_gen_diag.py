"""Diagnostic: compare TRTLLM-Gen accuracy with Cutlass and larger dimensions."""

import torch
from test_moe_fusion import MoEOpModelNVFP4
from utils.util import skip_pre_blackwell

import tensorrt_llm._torch.auto_deploy.custom_ops  # noqa: F401
from tensorrt_llm._torch.auto_deploy.export import torch_export_to_gm
from tensorrt_llm._torch.auto_deploy.transform.optimizer import InferenceOptimizer
from tensorrt_llm._torch.auto_deploy.utils.node_utils import is_op


def _run_three_paths(model, x, label):
    """Run baseline, cutlass, and trtllm-gen, return diffs."""
    gm_b = torch_export_to_gm(model, args=(x,), clone=True)
    with torch.inference_mode():
        out_base = gm_b(x)

    gm_c = torch_export_to_gm(model, args=(x,), clone=True)
    gm_c = InferenceOptimizer(None, {"fuse_nvfp4_moe": {"stage": "post_load_fusion"}})(None, gm_c)
    with torch.inference_mode():
        out_cut = gm_c(x)

    gm_t = torch_export_to_gm(model, args=(x,), clone=True)
    gm_t = InferenceOptimizer(
        None,
        {"fuse_nvfp4_moe": {"stage": "post_load_fusion", "backend": "trtllm_gen"}},
    )(None, gm_t)
    with torch.inference_mode():
        out_trt = gm_t(x)

    bs = out_base.float().abs().max().item()
    cut_scale = out_cut.float().abs().max().item()
    dc = (out_cut.float() - out_base.float()).abs()
    dt = (out_trt.float() - out_base.float()).abs()
    d_trt_cut = (out_trt.float() - out_cut.float()).abs()
    print(f"\n=== {label} ===")
    print(f"  baseline_scale={bs:.6e}, cutlass_scale={cut_scale:.6e}")
    print(
        f"  Cutlass vs base:    max={dc.max().item():.6e}, mean={dc.mean().item():.6e}, rel={dc.max().item() / bs:.4f}"
    )
    print(
        f"  TRTLLM-Gen vs base: max={dt.max().item():.6e}, mean={dt.mean().item():.6e}, rel={dt.max().item() / bs:.4f}"
    )
    print(
        f"  TRTLLM-Gen vs CUT:  max={d_trt_cut.max().item():.6e}, mean={d_trt_cut.mean().item():.6e},"
        f" rel={d_trt_cut.max().item() / cut_scale:.4f}"
    )
    return out_base, out_cut, out_trt


@skip_pre_blackwell
def test_trtllm_gen_diagnostic():
    device = "cuda"
    dtype = torch.bfloat16

    for hsz, isz in [(256, 256), (512, 256), (1024, 512), (2048, 1024)]:
        torch.manual_seed(1234)
        torch.cuda.manual_seed(1234)
        model = MoEOpModelNVFP4(
            hidden_size=hsz,
            intermediate_size=isz,
            num_experts=4,
            top_k=2,
            dtype=dtype,
            is_gated_mlp=True,
        ).to(device)
        x = torch.randn(2, hsz, device=device, dtype=dtype) * 0.1
        _run_three_paths(model, x, f"h={hsz}, i={isz}")


@skip_pre_blackwell
def test_compare_module_vs_ad_weight_processing():
    """Compare processed weights/scales from AD fusion with module processing."""
    import torch

    from tensorrt_llm._torch.modules.fused_moe.quantization import (
        trtllmgen_maybe_get_cached_w3_w1_permute_indices,
    )
    from tensorrt_llm.quantization.utils.fp4_utils import (
        get_reorder_rows_for_gated_act_gemm_row_indices,
        get_shuffle_matrix_a_row_indices,
        get_shuffle_matrix_sf_a_row_indices,
    )

    device = "cuda"
    dtype = torch.bfloat16
    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)

    model = MoEOpModelNVFP4(
        hidden_size=256,
        intermediate_size=256,
        num_experts=4,
        top_k=2,
        dtype=dtype,
        is_gated_mlp=True,
    ).to(device)

    EPILOGUE_TILE_M = 128
    num_experts = 4
    cache = {}

    for e in range(num_experts):
        # Get raw weights
        w1 = model.w1_weight[e].data  # (I, K/2) uint8
        w3 = model.w3_weight[e].data  # (I, K/2) uint8
        w1_bs = getattr(model, f"w1_weight_scale_{e}")  # (I, K/16) uint8
        w3_bs = getattr(model, f"w3_weight_scale_{e}")  # (I, K/16) uint8

        # --- Weight processing: MODULE path ---
        fc1_module = torch.cat([w3, w1], dim=0)  # (2I, K/2)
        perm_w = trtllmgen_maybe_get_cached_w3_w1_permute_indices(
            fc1_module, cache, EPILOGUE_TILE_M
        )
        fc1_module_shuffled = torch.ops.trtllm.shuffle_matrix(fc1_module, perm_w)

        # --- Weight processing: AD path ---
        fc1_ad = torch.cat([w3, w1], dim=0)  # (2I, K/2)
        perm0_w = get_reorder_rows_for_gated_act_gemm_row_indices(fc1_ad).to(device)
        perm1_w = get_shuffle_matrix_a_row_indices(fc1_ad, epilogue_tile_m=EPILOGUE_TILE_M).to(
            device
        )
        combined_w = perm0_w[perm1_w]
        fc1_ad_shuffled = torch.ops.trtllm.shuffle_matrix(fc1_ad, combined_w)

        w_match = torch.equal(fc1_module_shuffled, fc1_ad_shuffled)
        if not w_match:
            diff_rows = (fc1_module_shuffled != fc1_ad_shuffled).any(dim=1).sum().item()
            print(f"Expert {e}: WEIGHT MISMATCH! {diff_rows} rows differ")
        else:
            print(f"Expert {e}: weights match ✓")

        # --- Scale processing: MODULE path ---
        fc1_bs_module = torch.cat([w3_bs, w1_bs], dim=0).view(torch.uint8)  # (2I, K/16)
        perm_s = trtllmgen_maybe_get_cached_w3_w1_permute_indices(
            fc1_bs_module, cache, EPILOGUE_TILE_M, num_elts_per_sf=16
        )
        fc1_bs_module_shuffled = torch.ops.trtllm.shuffle_matrix(fc1_bs_module, perm_s)
        fc1_bs_module_interleaved = torch.ops.trtllm.block_scale_interleave(
            fc1_bs_module_shuffled.reshape(fc1_bs_module.shape)
        )

        # --- Scale processing: AD path ---
        fc1_bs_ad = torch.cat([w3_bs, w1_bs], dim=0).view(torch.uint8)  # (2I, K/16)
        perm0_s = get_reorder_rows_for_gated_act_gemm_row_indices(fc1_bs_ad.float()).to(device)
        perm1_s = get_shuffle_matrix_sf_a_row_indices(
            fc1_bs_ad, epilogue_tile_m=EPILOGUE_TILE_M, num_elts_per_sf=16
        ).to(device)
        combined_s = perm0_s[perm1_s]
        fc1_bs_ad_shuffled = torch.ops.trtllm.shuffle_matrix(fc1_bs_ad, combined_s)
        fc1_bs_ad_interleaved = torch.ops.trtllm.block_scale_interleave(fc1_bs_ad_shuffled)

        # Compare interleaved scales
        mod_flat = fc1_bs_module_interleaved.flatten()
        ad_flat = fc1_bs_ad_interleaved.flatten()
        if mod_flat.shape != ad_flat.shape:
            print(f"Expert {e}: SCALE SHAPE MISMATCH! module={mod_flat.shape}, ad={ad_flat.shape}")
        else:
            s_match = torch.equal(mod_flat, ad_flat)
            if not s_match:
                diff_count = (mod_flat != ad_flat).sum().item()
                print(
                    f"Expert {e}: SCALE MISMATCH! {diff_count} out of {mod_flat.numel()} bytes differ"
                )
            else:
                print(f"Expert {e}: scales match ✓")

        # Also compare permutation indices
        perm_match = torch.equal(perm_w, combined_w)
        print(f"  weight permute match: {perm_match}")
        perm_s_match = torch.equal(perm_s, combined_s)
        print(f"  scale permute match: {perm_s_match}")


@skip_pre_blackwell
def test_direct_kernel_call():
    """Call fp4_block_scale_moe_runner directly with module-processed weights."""
    import torch

    from tensorrt_llm._torch.modules.fused_moe.quantization import (
        trtllmgen_maybe_get_cached_w2_permute_indices,
        trtllmgen_maybe_get_cached_w3_w1_permute_indices,
    )
    from tensorrt_llm._torch.modules.fused_moe.routing import RoutingMethodType

    device = "cuda"
    dtype = torch.bfloat16
    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)

    model = MoEOpModelNVFP4(
        hidden_size=256,
        intermediate_size=256,
        num_experts=4,
        top_k=2,
        dtype=dtype,
        is_gated_mlp=True,
    ).to(device)
    x = torch.randn(2, 256, device=device, dtype=dtype) * 0.1

    # Get baseline
    gm = torch_export_to_gm(model, args=(x,), clone=True)
    with torch.inference_mode():
        out_base = gm(x)

    # Prepare weights using module's exact functions
    EPILOGUE_TILE_M = 128
    num_experts = 4
    intermediate_size = 256
    hidden_size = 256
    cache = {}

    fc1_w_list = []
    fc1_bs_list = []
    fc2_w_list = []
    fc2_bs_list = []
    fc1_alpha_list = []
    fc2_alpha_list = []
    fc1_scale_c_list = []

    for e in range(num_experts):
        w1 = model.w1_weight[e].data
        w3 = model.w3_weight[e].data
        w2 = model.w2_weight[e].data
        # AD load path stores block scales in interleaved layout. TRTLLM-Gen fusion
        # normalizes by reversing once, then applies shuffle+interleave.
        w1_bs = torch.ops.trtllm.block_scale_interleave_reverse(
            getattr(model, f"w1_weight_scale_{e}").unsqueeze(0)
        ).view(getattr(model, f"w1_weight_scale_{e}").shape)
        w3_bs = torch.ops.trtllm.block_scale_interleave_reverse(
            getattr(model, f"w3_weight_scale_{e}").unsqueeze(0)
        ).view(getattr(model, f"w3_weight_scale_{e}").shape)
        w2_bs = torch.ops.trtllm.block_scale_interleave_reverse(
            getattr(model, f"w2_weight_scale_{e}").unsqueeze(0)
        ).view(getattr(model, f"w2_weight_scale_{e}").shape)
        w1_alpha = getattr(model, f"w1_alpha_{e}").item()
        w3_alpha = getattr(model, f"w3_alpha_{e}").item()
        w2_alpha = getattr(model, f"w2_alpha_{e}").item()
        w1_input_scale = getattr(model, f"w1_input_scale_{e}")
        w2_input_scale = getattr(model, f"w2_input_scale_{e}")

        # FC1: concat [w3, w1], shuffle
        fc1_w = torch.cat([w3, w1], dim=0)
        perm_w = trtllmgen_maybe_get_cached_w3_w1_permute_indices(fc1_w, cache, EPILOGUE_TILE_M)
        fc1_w_shuffled = torch.ops.trtllm.shuffle_matrix(fc1_w, perm_w)
        fc1_w_list.append(fc1_w_shuffled)

        # FC1 block scales: concat [w3_bs, w1_bs], shuffle, interleave
        fc1_bs = torch.cat([w3_bs, w1_bs], dim=0).view(torch.uint8)
        perm_s = trtllmgen_maybe_get_cached_w3_w1_permute_indices(
            fc1_bs, cache, EPILOGUE_TILE_M, num_elts_per_sf=16
        )
        fc1_bs_shuffled = torch.ops.trtllm.shuffle_matrix(fc1_bs, perm_s)
        fc1_bs_interleaved = torch.ops.trtllm.block_scale_interleave(
            fc1_bs_shuffled.reshape(fc1_bs.shape)
        )
        fc1_bs_list.append(fc1_bs_interleaved.reshape(fc1_bs.shape))

        # FC2: shuffle
        perm_w2 = trtllmgen_maybe_get_cached_w2_permute_indices(w2, cache, EPILOGUE_TILE_M)
        fc2_w_shuffled = torch.ops.trtllm.shuffle_matrix(w2, perm_w2)
        fc2_w_list.append(fc2_w_shuffled)

        # FC2 block scales: shuffle, interleave
        w2_bs_u8 = w2_bs.view(torch.uint8)
        perm_s2 = trtllmgen_maybe_get_cached_w2_permute_indices(
            w2_bs_u8, cache, EPILOGUE_TILE_M, num_elts_per_sf=16
        )
        w2_bs_shuffled = torch.ops.trtllm.shuffle_matrix(w2_bs_u8, perm_s2)
        w2_bs_interleaved = torch.ops.trtllm.block_scale_interleave(
            w2_bs_shuffled.reshape(w2_bs_u8.shape)
        )
        fc2_bs_list.append(w2_bs_interleaved.reshape(w2_bs_u8.shape))

        # Compute scales following MODULE convention
        fc1_act_global = w1_input_scale  # same for all experts
        fc2_input_scale_global = w2_input_scale

        gate_alpha = w1_alpha * w1_input_scale.item() / fc1_act_global.item()
        up_alpha = w3_alpha * w1_input_scale.item() / fc1_act_global.item()
        fc1_alpha_list.append(gate_alpha)
        fc1_scale_c_list.append(fc2_input_scale_global.item() * up_alpha)
        fc2_alpha_list.append(w2_alpha)

    fc1_w_stacked = torch.stack(fc1_w_list, dim=0)
    fc1_bs_stacked = torch.stack(fc1_bs_list, dim=0)
    fc2_w_stacked = torch.stack(fc2_w_list, dim=0)
    fc2_bs_stacked = torch.stack(fc2_bs_list, dim=0)
    fc1_alpha_t = torch.tensor(fc1_alpha_list, dtype=torch.float32, device=device)
    fc1_scale_c_t = torch.tensor(fc1_scale_c_list, dtype=torch.float32, device=device)
    fc2_alpha_t = torch.tensor(fc2_alpha_list, dtype=torch.float32, device=device)
    fc1_act_global = getattr(model, "w1_input_scale_0")

    # Compute routing
    with torch.inference_mode():
        router_logits = model.gate(x)
    routing_weights = torch.nn.functional.softmax(router_logits, dim=1, dtype=torch.float)
    routing_weights, selected_experts = torch.topk(routing_weights, 2, dim=-1)
    routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)

    # Quantize input
    x2d = x.view(-1, hidden_size)
    hidden_fp4, hidden_scale = torch.ops.trtllm.fp4_quantize(x2d, fc1_act_global, 16, False, False)

    topk_ids = selected_experts.to(torch.int32)
    topk_weights = routing_weights.to(torch.bfloat16)

    with torch.inference_mode():
        outputs = torch.ops.trtllm.fp4_block_scale_moe_runner(
            None,
            None,
            hidden_fp4,
            hidden_scale.view(torch.float8_e4m3fn),
            fc1_w_stacked,
            fc1_bs_stacked.view(torch.float8_e4m3fn),
            None,
            None,
            None,
            None,
            fc2_w_stacked,
            fc2_bs_stacked.view(torch.float8_e4m3fn),
            None,
            fc1_scale_c_t,
            fc1_alpha_t,
            fc2_alpha_t,
            num_experts,
            2,
            1,
            1,
            intermediate_size,
            0,
            num_experts,
            1.0,
            int(RoutingMethodType.DeepSeekV3),
            True,
            0,
            topk_weights,
            topk_ids,
        )

    out_direct = outputs[0]
    diff = (out_direct.float() - out_base.float()).abs()
    bs = out_base.float().abs().max().item()
    print("\nDirect kernel call:")
    print(f"  max_abs_diff={diff.max().item():.6e}, mean={diff.mean().item():.6e}")
    print(f"  max_rel_err={diff.max().item() / bs:.4f}")

    # Also run through AD custom op for comparison
    gm_t = torch_export_to_gm(model, args=(x,), clone=True)
    gm_t = InferenceOptimizer(
        None,
        {"fuse_nvfp4_moe": {"stage": "post_load_fusion", "backend": "trtllm_gen"}},
    )(None, gm_t)
    with torch.inference_mode():
        out_ad = gm_t(x)

    diff_ad = (out_ad.float() - out_base.float()).abs()
    print("\nAD custom op:")
    print(f"  max_abs_diff={diff_ad.max().item():.6e}, mean={diff_ad.mean().item():.6e}")
    print(f"  max_rel_err={diff_ad.max().item() / bs:.4f}")

    # Compare direct vs AD
    diff_d_vs_ad = (out_direct.float() - out_ad.float()).abs()
    print("\nDirect vs AD custom op:")
    print(f"  max_abs_diff={diff_d_vs_ad.max().item():.6e}")

    # Print values
    b = out_base.float().flatten()[:10].tolist()
    d = out_direct.float().flatten()[:10].tolist()
    a = out_ad.float().flatten()[:10].tolist()
    print(f"\n{'idx':>3} {'baseline':>12} {'direct':>12} {'ad_op':>12}")
    for j in range(10):
        print(f"{j:3d} {b[j]:12.3e} {d[j]:12.3e} {a[j]:12.3e}")


@skip_pre_blackwell
def test_verify_actual_ad_tensors():
    """Extract and verify the actual tensors the AD fusion produces."""
    import torch

    from tensorrt_llm._torch.modules.fused_moe.quantization import (
        trtllmgen_maybe_get_cached_w2_permute_indices,
        trtllmgen_maybe_get_cached_w3_w1_permute_indices,
    )

    device = "cuda"
    dtype = torch.bfloat16
    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)

    model = MoEOpModelNVFP4(
        hidden_size=256,
        intermediate_size=256,
        num_experts=4,
        top_k=2,
        dtype=dtype,
        is_gated_mlp=True,
    ).to(device)
    x = torch.randn(2, 256, device=device, dtype=dtype) * 0.1

    # Fuse with TRTLLM-Gen
    gm = torch_export_to_gm(model, args=(x,), clone=True)
    gm = InferenceOptimizer(
        None,
        {"fuse_nvfp4_moe": {"stage": "post_load_fusion", "backend": "trtllm_gen"}},
    )(None, gm)

    # Extract actual tensors from graph
    actual_tensors = {}
    for node in gm.graph.nodes:
        if is_op(node, torch.ops.auto_deploy.trtllm_nvfp4_trtllm_gen_moe_fused):
            for i, arg in enumerate(node.args):
                if isinstance(arg, torch.fx.Node) and arg.op == "get_attr":
                    tensor = getattr(gm, arg.target)
                    actual_tensors[i] = (arg.target, tensor)
            break

    print("\n=== AD Fusion Tensor Inventory ===")
    for i, (name, t) in sorted(actual_tensors.items()):
        print(f"  arg[{i}]: {name}, shape={list(t.shape)}, dtype={t.dtype}")

    # Now build expected tensors using module's exact processing
    cache = {}
    EPILOGUE_TILE_M = 128
    expected_fc1_w_list = []
    expected_fc1_bs_list = []
    expected_fc2_w_list = []
    expected_fc2_bs_list = []

    for e in range(4):
        w1 = model.w1_weight[e].data
        w3 = model.w3_weight[e].data
        w2 = model.w2_weight[e].data
        w1_bs = getattr(model, f"w1_weight_scale_{e}")
        w3_bs = getattr(model, f"w3_weight_scale_{e}")
        w2_bs = getattr(model, f"w2_weight_scale_{e}")

        # FC1 weights
        fc1_w = torch.cat([w3, w1], dim=0)
        perm_w = trtllmgen_maybe_get_cached_w3_w1_permute_indices(fc1_w, cache, EPILOGUE_TILE_M)
        expected_fc1_w_list.append(torch.ops.trtllm.shuffle_matrix(fc1_w, perm_w))

        # FC1 block scales
        fc1_bs = torch.cat([w3_bs, w1_bs], dim=0).view(torch.uint8)
        perm_s = trtllmgen_maybe_get_cached_w3_w1_permute_indices(
            fc1_bs, cache, EPILOGUE_TILE_M, num_elts_per_sf=16
        )
        fc1_bs_shuffled = torch.ops.trtllm.shuffle_matrix(fc1_bs, perm_s)
        fc1_bs_interleaved = torch.ops.trtllm.block_scale_interleave(
            fc1_bs_shuffled.reshape(fc1_bs.shape)
        )
        expected_fc1_bs_list.append(fc1_bs_interleaved.reshape(fc1_bs.shape))

        # FC2 weights
        perm_w2 = trtllmgen_maybe_get_cached_w2_permute_indices(w2, cache, EPILOGUE_TILE_M)
        expected_fc2_w_list.append(torch.ops.trtllm.shuffle_matrix(w2, perm_w2))

        # FC2 block scales
        w2_bs_u8 = w2_bs.view(torch.uint8)
        perm_s2 = trtllmgen_maybe_get_cached_w2_permute_indices(
            w2_bs_u8, cache, EPILOGUE_TILE_M, num_elts_per_sf=16
        )
        w2_bs_shuffled = torch.ops.trtllm.shuffle_matrix(w2_bs_u8, perm_s2)
        w2_bs_interleaved = torch.ops.trtllm.block_scale_interleave(
            w2_bs_shuffled.reshape(w2_bs_u8.shape)
        )
        expected_fc2_bs_list.append(w2_bs_interleaved.reshape(w2_bs_u8.shape))

    expected_fc1_w = torch.stack(expected_fc1_w_list)
    expected_fc1_bs = torch.stack(expected_fc1_bs_list).view(torch.float8_e4m3fn)
    expected_fc2_w = torch.stack(expected_fc2_w_list)
    expected_fc2_bs = torch.stack(expected_fc2_bs_list).view(torch.float8_e4m3fn)

    # Compare actual vs expected
    # arg 3 = fc1_weights, arg 4 = fc2_weights, arg 5 = fc1_blockscale, arg 6 = fc2_blockscale
    for i, (name, actual) in sorted(actual_tensors.items()):
        au8 = actual.contiguous().view(torch.uint8).flatten()
        for label, expected in [
            ("fc1_w", expected_fc1_w),
            ("fc1_bs", expected_fc1_bs),
            ("fc2_w", expected_fc2_w),
            ("fc2_bs", expected_fc2_bs),
        ]:
            eu8 = expected.contiguous().view(torch.uint8).flatten()
            if au8.shape == eu8.shape:
                if torch.equal(au8, eu8):
                    print(f"  arg[{i}] ({name}) MATCHES {label} ✓")
                else:
                    diff = (au8.int() - eu8.int()).abs().sum().item()
                    print(f"  arg[{i}] ({name}) vs {label}: {diff} byte diffs out of {au8.numel()}")


@skip_pre_blackwell
def test_shuffle_matrix_is_row_permutation():
    """Verify that torch.ops.trtllm.shuffle_matrix == simple row indexing."""
    from tensorrt_llm.quantization.utils.fp4_utils import (
        get_shuffle_matrix_a_row_indices,
        get_shuffle_matrix_sf_a_row_indices,
    )

    device = "cuda"
    # Weight-like tensor: (512, 128) uint8
    w = torch.randint(0, 256, (512, 128), dtype=torch.uint8, device=device)
    indices = get_shuffle_matrix_a_row_indices(w, epilogue_tile_m=128).to(device)
    shuffled = torch.ops.trtllm.shuffle_matrix(w, indices)
    expected = w[indices]
    matches = torch.equal(shuffled, expected)
    print(f"\nWeight shuffle: shuffle_matrix == row_index: {matches}")
    if not matches:
        diff_rows = (shuffled != expected).any(dim=1).sum().item()
        print(f"  {diff_rows} out of {w.shape[0]} rows differ")
    assert matches, "shuffle_matrix is NOT a pure row permutation for weights!"

    # Scale-like tensor: (512, 16) uint8
    s = torch.randint(0, 256, (512, 16), dtype=torch.uint8, device=device)
    indices_s = get_shuffle_matrix_sf_a_row_indices(s, epilogue_tile_m=128, num_elts_per_sf=16).to(
        device
    )
    shuffled_s = torch.ops.trtllm.shuffle_matrix(s, indices_s)
    expected_s = s[indices_s]
    matches_s = torch.equal(shuffled_s, expected_s)
    print(f"Scale shuffle: shuffle_matrix == row_index: {matches_s}")
    if not matches_s:
        diff_rows_s = (shuffled_s != expected_s).any(dim=1).sum().item()
        print(f"  {diff_rows_s} out of {s.shape[0]} rows differ")
    assert matches_s, "shuffle_matrix is NOT a pure row permutation for scales!"


@skip_pre_blackwell
def test_trtllm_gen_determinism():
    """Run TRTLLM-Gen twice on same input to check determinism."""
    device = "cuda"
    dtype = torch.bfloat16
    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)
    model = MoEOpModelNVFP4(
        hidden_size=256,
        intermediate_size=256,
        num_experts=4,
        top_k=2,
        dtype=dtype,
        is_gated_mlp=True,
    ).to(device)
    x = torch.randn(2, 256, device=device, dtype=dtype) * 0.1

    gm = torch_export_to_gm(model, args=(x,), clone=True)
    gm = InferenceOptimizer(
        None,
        {"fuse_nvfp4_moe": {"stage": "post_load_fusion", "backend": "trtllm_gen"}},
    )(None, gm)
    with torch.inference_mode():
        out1 = gm(x)
        out2 = gm(x)
    diff = (out1.float() - out2.float()).abs().max().item()
    print(f"\nDeterminism check: max diff between two runs = {diff:.6e}")
    assert diff == 0.0, f"Non-deterministic: diff={diff}"
