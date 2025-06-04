# import numpy as np

# # Load tensors
# a = np.load('no_fused.npy')
# b = np.load('/tmp/tllm_debug/PP_1/TP_2/iteration_0/GemmaForCausalLM_transformer_layers_0_attention_dense_multiply_collect_L518_multiply_collect_L286_multiply_and_lora_L248__gemm_plugin_L139_PLUGIN_V2_Gemm_0_output_0_debug_final_output_NO_FUSION.npy')

# # Flatten
# a_flat = a.flatten()
# b_flat = b.flatten()

# # Ensure same shape
# assert a_flat.shape == b_flat.shape, "Tensors must have the same shape!"

# # Print header
# print(f"{'Index':>6} | {'Tensor1 (not fused)':>25} | {'Tensor2 (fused)':>25} | Diff")
# print("-" * 70)

# # Print values side by side, highlight differences
# for i, (x, y) in enumerate(zip(a_flat, b_flat)):
#     diff = "DIFF" if x != y else ""
#     print(f"{i:6d} | {x:34} | {y:34} | {diff}")

# # Optionally, print summary
# num_diff = np.sum(a_flat != b_flat)
# print(f"\nTotal differences: {num_diff} / {a_flat.size}")

import numpy as np

# Load tensors
a = np.load(
    'NO_FUSION_GemmaForCausalLM_transformer_layers_0___add___L322_elementwise_binary_L3011_ELEMENTWISE_SUM_0_output_0_debug_residual_NO_FUSION.npy'
)
b = np.load(
    'FUSION_ONLY_GemmaForCausalLM_transformer_layers_0_attention_dense_multiply_collect_L518_multiply_collect_L296_collect_and_bias_L537_allreduce_L4085_PLUGIN_V2_AllReduce_0_output_1_debug_inter_output_FUSION_ONLY_debug_residual_FUSION.npy'
)

# Flatten
a_flat = a.flatten()
b_flat = b.flatten()

# Ensure same shape
assert a_flat.shape == b_flat.shape, "Tensors must have the same shape!"

# Print header
print(
    f"{'Index':>6} | {'Tensor1 (not fused)':>25} | {'Tensor2 (fused)':>25} | Close"
)
print("-" * 70)

# Use np.isclose for elementwise comparison with default tolerances
is_close = np.isclose(a_flat, b_flat, atol=0.15, rtol=0.05)

for i, (x, y, close) in enumerate(zip(a_flat, b_flat, is_close)):
    diff = "" if close else "DIFF"
    print(f"{i:6d} | {x:34} | {y:34} | {diff}")

# Optionally, print summary
num_diff = np.sum(~is_close)
print(f"\nTotal differences (not close): {num_diff} / {a_flat.size}")

# Direct comparison (strict equality) - commented out:
# num_diff_strict = np.sum(a_flat != b_flat)
# print(f"\nTotal strict differences: {num_diff_strict} / {a_flat.size}")

# Use np.allclose for overall check
all_close = np.allclose(a_flat, b_flat)
print(f"\nnp.allclose result: {all_close}")
