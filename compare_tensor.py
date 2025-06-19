import torch
import os
import numpy as np

def compare_tensors_meta_info(dir1_name, dir2_name, filename, filename_opt):
    # Read original tensor
    orig_path = os.path.join(dir1_name, filename)
    opt_path = os.path.join(dir2_name, filename_opt)
    
    # Check if files exist
    if not os.path.exists(orig_path) or not os.path.exists(opt_path):
        print(f"Files do not exist: {orig_path} or {opt_path}")
        return
        
    # Read and parse tensor info
    with open(orig_path, "r") as f:
        orig_lines = f.readlines()
    with open(opt_path, "r") as f:
        opt_lines = f.readlines()
        
    # Extract tensor basic info
    print("Original tensor info:")
    for line in orig_lines[:4]:
        print(line.strip())
    print("\nOptimized tensor info:")    
    for line in opt_lines[:4]:
        print(line.strip())
        
    # Extract values and compare
    orig_mean = float(orig_lines[3].split(": ")[1])
    opt_mean = float(opt_lines[3].split(": ")[1])
    orig_std = float(orig_lines[4].split(": ")[1]) 
    opt_std = float(opt_lines[4].split(": ")[1])
    orig_min = float(orig_lines[5].split(": ")[1])
    opt_min = float(opt_lines[5].split(": ")[1])
    orig_max = float(orig_lines[6].split(": ")[1])
    opt_max = float(opt_lines[6].split(": ")[1])
    
    print("\nValue comparison:")
    print(f"Mean difference: {abs(orig_mean - opt_mean)}")
    print(f"Std difference: {abs(orig_std - opt_std)}")
    print(f"Min difference: {abs(orig_min - opt_min)}")
    print(f"Max difference: {abs(orig_max - opt_max)}")
    print("\n")

def compare_tensors_data(dir1_name, dir2_name, filename, filename_opt):
    print(f"\n\nComparing tensors: {filename} and {filename_opt}")
    # Read original tensor
    orig_path = os.path.join(dir1_name, filename)
    opt_path = os.path.join(dir2_name, filename_opt)
    
    # Check if files exist
    if not os.path.exists(orig_path) or not os.path.exists(opt_path):
        print(f"Files do not exist: {orig_path} or {opt_path}")
        return
        
    # Read and parse tensor info
    orig_data = np.load(orig_path)
    opt_data = np.load(opt_path)
    
    # Compare data
    print(f"Original tensor data: {orig_data.shape}, {orig_data}")
    print(f"Optimized tensor data: {opt_data.shape}, {opt_data}")
    
    # Calculate difference
    diff = np.abs(orig_data - opt_data)
    print(f"Difference: {diff}")

    # 计算相对和绝对误差
    rel_diff = np.abs(orig_data - opt_data) / (np.abs(orig_data) + 1e-7)  # 添加小值避免除0
    abs_diff = np.abs(orig_data - opt_data)
    
    # 设置容差阈值
    rtol = 1e-2  # 相对容差
    atol = 0.1   # 绝对容差
    
    # 检查是否在容差范围内
    within_tol = (abs_diff <= atol + rtol * np.abs(orig_data))
    
    # 输出结果
    print("\n逐元素对比结果:")
    print(f"最大相对误差: {np.max(rel_diff)}")
    print(f"最大绝对误差: {np.max(abs_diff)}")
    print(f"超出容差的元素数量: {np.sum(~within_tol)}")
    print(f"超出容差的元素比例: {np.mean(~within_tol)*100:.2f}%")
    
    # # 如果有超出容差的元素,打印其位置和具体值
    # if np.any(~within_tol):
    #     print("\n超出容差的元素详情:")
    #     fail_idx = np.where(~within_tol)
    #     for idx in zip(*fail_idx):
    #         print(f"位置 {idx}:")
    #         print(f"  原始值: {orig_data[idx]}")
    #         print(f"  优化值: {opt_data[idx]}")
    #         print(f"  相对误差: {rel_diff[idx]:.4f}")
    #         print(f"  绝对误差: {abs_diff[idx]:.4f}")

def compare_tensors_data(dir1_name, dir2_name, filename, filename_opt):
    print(f"\n\nComparing tensors: {filename} and {filename_opt}")
    # Read original tensor
    orig_path = os.path.join(dir1_name, filename)
    opt_path = os.path.join(dir2_name, filename_opt)
    
    # Check if files exist
    if not os.path.exists(orig_path) or not os.path.exists(opt_path):
        print(f"Files do not exist: {orig_path} or {opt_path}")
        return
        
    def read_tensor_data(file_path):
        data = []
        with open(file_path, 'r') as f:
            for line in f:
                # 将每行按空格分割，并转换为float类型
                row = [float(x) for x in line.strip().split()]
                data.append(row)
        
        # 转换为numpy数组
        return np.array(data)

    # 使用示例
    data = read_tensor_data(orig_path).reshape(-1)
    # print(data)

    data_opt = np.load(opt_path).reshape(-1)
    # print(data_opt)

    diff = np.abs(data - data_opt)
    # print(diff)

    rel_diff = np.abs(data - data_opt) / (np.abs(data) + 1e-7)
    # print(rel_diff)
    print(f"max rel_diff: {np.max(rel_diff)}")
    print(f"max diff: {np.max(diff)}")

# Usage example
if __name__ == "__main__":
    # compare_tensors_meta_info("tensor_inputs", "tensor_inputs_opt", "inputs_embeds4.txt")
    # compare_tensors_data("tensor_inputs", "tensor_inputs_opt", "inputs_embeds4.npy")
    # print("output-layer-0: --------------------------------")
    # compare_tensors_meta_info("tensor_outputs", "tensor_outputs_opt", "hidden_states_output4_0.txt")
    # compare_tensors_data("tensor_outputs", "tensor_outputs_opt", "hidden_states_output4_0.npy")
    # print("output-layer-1: --------------------------------")
    # compare_tensors_meta_info("tensor_outputs", "tensor_outputs_opt", "hidden_states_output4_1.txt")
    # compare_tensors_data("tensor_outputs", "tensor_outputs_opt", "hidden_states_output4_1.npy")
    # print("\n\n")
    # print("   ffn-output, layer-0: --------------------------------")
    # compare_tensors_meta_info("ffn_input", "ffn_input_opt", "ffn_input4_0.txt")
    # compare_tensors_data("ffn_input", "ffn_input_opt", "ffn_input4_0.npy")
    # print("   ffn-output, layer-1: --------------------------------")
    # compare_tensors_meta_info("ffn_input", "ffn_input_opt", "ffn_input4_1.txt")
    # compare_tensors_data("ffn_input", "ffn_input_opt", "ffn_input4_1.npy")

    # print("\n\n")
    # print("   moe-input, layer-1: --------------------------------")
    # compare_tensors_meta_info("moe_input", "moe_input_opt", "moe_input4_1.txt")
    # compare_tensors_data("moe_input", "moe_input_opt", "moe_input4_1.npy")
    # print("   moe-output, layer-1: --------------------------------")
    # compare_tensors_meta_info("moe_output", "moe_output_opt", "moe_output4_1.txt")
    # compare_tensors_data("moe_output", "moe_output_opt", "moe_output4_1.npy")

    # print("\n\n")
    # print("   moe-routed-output, layer-1: --------------------------------")
    # compare_tensors_meta_info("moe_routed_output", "moe_routed_output_opt", "moe_routed_output4_1.txt")
    # compare_tensors_data("moe_routed_output", "moe_routed_output_opt", "moe_routed_output4_1.npy")
    # print("   moe-shared-output, layer-1: --------------------------------")
    # compare_tensors_meta_info("moe_shared_output", "moe_shared_output_opt", "moe_shared_output4_1.txt")
    # compare_tensors_data("moe_shared_output", "moe_shared_output_opt", "moe_shared_output4_1.npy")

    # print("\n\n")
    # print("   moe-router-logits, layer-1: --------------------------------")
    # compare_tensors_meta_info("moe_router_logits", "moe_router_logits_opt", "moe_router_logits4_1.txt")
    # compare_tensors_data("moe_router_logits", "moe_router_logits_opt", "moe_router_logits4_1.npy")
    # print("   moe-hidden-states, layer-1: --------------------------------")
    # compare_tensors_meta_info("moe_hidden_states", "moe_hidden_states_opt", "moe_hidden_states4_1.txt")
    # compare_tensors_data("moe_hidden_states", "moe_hidden_states_opt", "moe_hidden_states4_1.npy")

    # print("\n\n")
    # print("   forward-chunk-output, layer-1: --------------------------------")
    # compare_tensors_meta_info("forward_chunk_output", "forward_chunk_output_opt", "forward_chunk_output4_1.txt")
    # compare_tensors_data("forward_chunk_output", "forward_chunk_output_opt", "forward_chunk_output4_1.npy")

    # print("\n\n")
    # print("   forward-chunk-moe-input, layer-1: --------------------------------")
    # compare_tensors_meta_info("forward_chunk_moe_input", "forward_chunk_moe_input_opt", "forward_chunk_moe_input4_1.txt")
    # compare_tensors_data("forward_chunk_moe_input", "forward_chunk_moe_input_opt", "forward_chunk_moe_input4_1.npy")
    # print("   forward-chunk-moe-selected-slots, layer-1: --------------------------------")
    # compare_tensors_meta_info("forward_chunk_moe_selected_slots", "forward_chunk_moe_selected_slots_opt", "forward_chunk_moe_selected_slots4_1.txt")
    # compare_tensors_data("forward_chunk_moe_selected_slots", "forward_chunk_moe_selected_slots_opt", "forward_chunk_moe_selected_slots4_1.npy")
    # print("   forward-chunk-moe-final-scales, layer-1: --------------------------------")
    # compare_tensors_meta_info("forward_chunk_moe_final_scales", "forward_chunk_moe_final_scales_opt", "forward_chunk_moe_final_scales4_1.txt")
    # compare_tensors_data("forward_chunk_moe_final_scales", "forward_chunk_moe_final_scales_opt", "forward_chunk_moe_final_scales4_1.npy")
    # print("   forward-chunk-moe-w3-w1-weight, layer-1: --------------------------------")
    # compare_tensors_meta_info("forward_chunk_moe_w3_w1_weight", "forward_chunk_moe_w3_w1_weight_opt", "forward_chunk_moe_w3_w1_weight4_1.txt")
    # compare_tensors_data("forward_chunk_moe_w3_w1_weight", "forward_chunk_moe_w3_w1_weight_opt", "forward_chunk_moe_w3_w1_weight4_1.npy")
    # print("   forward-chunk-moe-w2-weight, layer-1: --------------------------------")
    # compare_tensors_meta_info("forward_chunk_moe_w2_weight", "forward_chunk_moe_w2_weight_opt", "forward_chunk_moe_w2_weight4_1.txt")
    # compare_tensors_data("forward_chunk_moe_w2_weight", "forward_chunk_moe_w2_weight_opt", "forward_chunk_moe_w2_weight4_1.npy")
    # print("   forward-chunk-moe-quant-scales-0, layer-1: --------------------------------")
    # compare_tensors_meta_info("forward_chunk_moe_quant_scales_0", "forward_chunk_moe_quant_scales_0_opt", "forward_chunk_moe_quant_scales_04_1.txt")
    # compare_tensors_data("forward_chunk_moe_quant_scales_0", "forward_chunk_moe_quant_scales_0_opt", "forward_chunk_moe_quant_scales_04_1.npy")
    # print("   forward-chunk-moe-quant-scales-1, layer-1: --------------------------------")
    # compare_tensors_meta_info("forward_chunk_moe_quant_scales_1", "forward_chunk_moe_quant_scales_1_opt", "forward_chunk_moe_quant_scales_14_1.txt")
    # compare_tensors_data("forward_chunk_moe_quant_scales_1", "forward_chunk_moe_quant_scales_1_opt", "forward_chunk_moe_quant_scales_14_1.npy")



    ###################for test_fused_moe.py compare tensor###################
    print("forward_chunk_moe_input: --------------------------------")
    compare_tensors_meta_info("./", "forward_chunk_moe_input_opt", "input_activations_void.txt", "forward_chunk_moe_input.txt")
    compare_tensors_data("./", "forward_chunk_moe_input_opt", "input_activations_void.npy", "forward_chunk_moe_input.npy")
    compare_tensors_meta_info("./", "forward_chunk_moe_selected_slots_opt", "token_selected_experts.txt", "forward_chunk_moe_selected_slots.txt")
    compare_tensors_data("./", "forward_chunk_moe_selected_slots_opt", "token_selected_experts.npy", "forward_chunk_moe_selected_slots.npy")
    compare_tensors_meta_info("./", "forward_chunk_moe_final_scales_opt", "token_final_scales.txt", "forward_chunk_moe_final_scales.txt")
    compare_tensors_data("./", "forward_chunk_moe_final_scales_opt", "token_final_scales.npy", "forward_chunk_moe_final_scales.npy")

    print("\n\nforward_chunk_moe_permuted_data_tensor: --------------------------------")
    compare_tensors_meta_info("./", "forward_chunk_moe_permuted_data_tensor_opt", "permuted_data_.txt", "forward_chunk_moe_permuted_data_tensor.txt", )
    compare_tensors_data("./", "forward_chunk_moe_permuted_data_tensor_opt", "permuted_data_.npy", "forward_chunk_moe_permuted_data_tensor.npy")

    # print("\n\nforward_chunk_moe_unpermuted_token_selected_experts_tensor: --------------------------------")
    # compare_tensors_meta_info("./", "forward_chunk_moe_unpermuted_token_selected_experts_tensor_opt", "unpermuted_token_selected_experts_.txt", "forward_chunk_moe_unpermuted_token_selected_experts_tensor.txt")
    # compare_tensors_data("./", "forward_chunk_moe_unpermuted_token_selected_experts_tensor_opt", "unpermuted_token_selected_experts_.npy", "forward_chunk_moe_unpermuted_token_selected_experts_tensor.npy")

    # print("\n\nforward_chunk_moe_unpermuted_source_token_ids_tensor: --------------------------------")
    # compare_tensors_meta_info("./", "forward_chunk_moe_unpermuted_source_token_ids_tensor_opt", "unpermuted_source_token_ids_.txt", "forward_chunk_moe_unpermuted_source_token_ids_tensor.txt")
    # compare_tensors_data("./", "forward_chunk_moe_unpermuted_source_token_ids_tensor_opt", "unpermuted_source_token_ids_.npy", "forward_chunk_moe_unpermuted_source_token_ids_tensor.npy")

    # print("\n\nforward_chunk_moe_permuted_source_token_ids_tensor: --------------------------------")
    # compare_tensors_meta_info("./", "forward_chunk_moe_permuted_source_token_ids_tensor_opt", "permuted_source_token_ids_.txt", "forward_chunk_moe_permuted_source_token_ids_tensor.txt")
    # compare_tensors_data("./", "forward_chunk_moe_permuted_source_token_ids_tensor_opt", "permuted_source_token_ids_.npy", "forward_chunk_moe_permuted_source_token_ids_tensor.npy")

    # print("\n\nforward_chunk_moe_permuted_token_selected_experts_tensor: --------------------------------")
    # compare_tensors_meta_info("./", "forward_chunk_moe_permuted_token_selected_experts_tensor_opt", "permuted_token_selected_experts_.txt", "forward_chunk_moe_permuted_token_selected_experts_tensor.txt")
    # compare_tensors_data("./", "forward_chunk_moe_permuted_token_selected_experts_tensor_opt", "permuted_token_selected_experts_.npy", "forward_chunk_moe_permuted_token_selected_experts_tensor.npy")

    # print("\n\nforward_chunk_moe_expert_first_token_offset_tensor: --------------------------------")
    # compare_tensors_meta_info("./", "forward_chunk_moe_expert_first_token_offset_tensor_opt", "expert_first_token_offset_.txt", "forward_chunk_moe_expert_first_token_offset_tensor.txt")
    # compare_tensors_data("./", "forward_chunk_moe_expert_first_token_offset_tensor_opt", "expert_first_token_offset_.npy", "forward_chunk_moe_expert_first_token_offset_tensor.npy")

    # print("\n\nforward_chunk_moe_src_to_dest_map_tensor: --------------------------------")
    # compare_tensors_meta_info("./", "forward_chunk_moe_src_to_dest_map_tensor_opt", "expanded_source_row_to_expanded_dest_row.txt", "forward_chunk_moe_src_to_dest_map_tensor.txt")
    # compare_tensors_data("./", "forward_chunk_moe_src_to_dest_map_tensor_opt", "expanded_source_row_to_expanded_dest_row.npy", "forward_chunk_moe_src_to_dest_map_tensor.npy")
    
    print("\n\nforward_chunk_moe_grouped_gemm1_result_tensor: --------------------------------")
    compare_tensors_meta_info("./", "forward_chunk_moe_group_gemm_1_opt", "grouped_gemm1_result_.txt", "forward_chunk_moe_group_gemm_1.txt")
    compare_tensors_data("./", "forward_chunk_moe_group_gemm_1_opt", "grouped_gemm1_result_.npy", "forward_chunk_moe_group_gemm_1.npy")

    print("\n\nforward_chunk_moe_fc1_result_tensor: --------------------------------")
    compare_tensors_meta_info("./", "forward_chunk_moe_swiglu_output_opt", "fc1_result_.txt", "forward_chunk_moe_swiglu_output.txt")
    compare_tensors_data("./", "forward_chunk_moe_swiglu_output_opt", "fc1_result_.npy", "forward_chunk_moe_swiglu_output.npy")

    print("\n\nforward_chunk_moe_grouped_gemm2_result_tensor: --------------------------------")
    compare_tensors_meta_info("./", "forward_chunk_moe_group_gemm_2_output_opt", "grouped_gemm2_result_.txt", "forward_chunk_moe_group_gemm_2_output.txt")
    compare_tensors_data("./", "forward_chunk_moe_group_gemm_2_output_opt", "grouped_gemm2_result_.npy", "forward_chunk_moe_group_gemm_2_output.npy")

    print("\n\nforward_chunk_moe_fc2_result_tensor: --------------------------------")
    compare_tensors_meta_info("./", "forward_chunk_moe_final_hidden_states_opt", "final_output_.txt", "forward_chunk_moe_final_hidden_states.txt")
    compare_tensors_data("./", "forward_chunk_moe_final_hidden_states_opt", "final_output_.npy", "forward_chunk_moe_final_hidden_states.npy")