# Debug script - Will be removed
import sys
import types
from unittest.mock import MagicMock

import torch
from torch.export import Dim
from transformers import AutoModelForCausalLM, AutoTokenizer

import tensorrt_llm._torch.auto_deploy.custom_ops  # noqa: F401
from tensorrt_llm._torch.auto_deploy.export import torch_export_to_gm


def _mock_gptqmodel_extensions():
    """Mock the broken Marlin kernel extension to allow gptqmodel to import.

    The issue is:
    1. gptqmodel/utils/marlin.py calls load_extension_module("gptqmodel_marlin_kernels")
    2. This fails because the extension is not installed properly
    3. The ImportError is caught but gptqmodel_marlin_kernels is never defined
    4. marlin_awq.py tries to import gptqmodel_marlin_kernels from marlin.py and fails

    Solution: Pre-populate sys.modules with a mock that has gptqmodel_marlin_kernels = None
    """
    # Create a mock kernel module that will be returned by load_extension_module
    mock_marlin_kernels = MagicMock()
    mock_marlin_kernels.__name__ = "gptqmodel_marlin_kernels"

    # Pre-register the mock in sys.modules so importlib.import_module finds it
    sys.modules["gptqmodel_marlin_kernels"] = mock_marlin_kernels

    print("[DEBUG] Mocked gptqmodel_marlin_kernels extension")


_mock_gptqmodel_extensions()


def _patch_gptq_backend():
    """Force GPTQ to use TorchQuantLinear backend instead of optimized kernels."""
    try:
        from gptqmodel.utils.backend import BACKEND
        from optimum.gptq import quantizer as gptq_quantizer

        _original_select_quant_linear = gptq_quantizer.GPTQQuantizer.select_quant_linear

        def _patched_select_quant_linear(self, device_map, pack=False):
            self.backend = BACKEND.TORCH
            return _original_select_quant_linear(self, device_map, pack)

        gptq_quantizer.GPTQQuantizer.select_quant_linear = _patched_select_quant_linear
        print("[DEBUG] GPTQ backend patched to use BACKEND.TORCH")
    except ImportError as e:
        print(f"[DEBUG] Failed to patch GPTQ backend: {e}")


_patch_gptq_backend()


def _torch_quant_linear_forward_patched(self, x: torch.Tensor):
    out = torch.ops.auto_deploy.torch_fake_quant_int4_gptq_linear(
        x, self.qweight, self.bias, [], [self.scales], [], [self.qzeros]
    )
    if self.adapter:
        out = self.adapter.apply(x=x, out=out)
    return out


def patch_torch_quant_linear(module_or_model):
    for _, m in getattr(module_or_model, "named_modules", lambda: [])():
        if type(m).__name__ == "TorchQuantLinear":
            if not hasattr(m, "__original_forward"):
                m.__original_forward = m.forward
                m.forward = types.MethodType(_torch_quant_linear_forward_patched, m)


def extract_logits(output):
    """Extract logits from model output (handles different output formats)."""
    if hasattr(output, "logits"):
        return output.logits
    return output[0] if isinstance(output, tuple) else output


def generate_tokens(model_or_gm, input_ids, num_tokens=5):
    """Generate tokens one by one (simulating decode phase)."""
    tokens = []
    current_ids = input_ids.clone()

    for _ in range(num_tokens):
        with torch.no_grad():
            output = model_or_gm(current_ids)
            logits = extract_logits(output)
            next_token = logits[0, -1].argmax().item()
            tokens.append(next_token)
            current_ids = torch.cat(
                [current_ids, torch.tensor([[next_token]], device=current_ids.device)], dim=1
            )

    return tokens


# ============================================================
# Load Model
# ============================================================
model_name = "Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int4"
print(f"Loading {model_name}...")
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", use_cache=False).to(
    "cuda"
)
patch_torch_quant_linear(model)
model.eval()
print("Model loaded with TorchQuantLinear patched to use custom GPTQ op")

tokenizer = AutoTokenizer.from_pretrained(model_name)

# Test prompt
prompt = "The capital of France is"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
print(f"\nPrompt: '{prompt}'")
print(f"Input shape: {input_ids.shape}")

num_gen_tokens = 10

# ============================================================
# Test 1: Original model (before export)
# ============================================================
print("\n" + "=" * 60)
print("=== TEST 1: ORIGINAL MODEL (before export) ===")
print("=" * 60)

pre_export_tokens = generate_tokens(model, input_ids, num_gen_tokens)
pre_export_text = tokenizer.decode(pre_export_tokens)
print(f"Generated tokens: {pre_export_tokens}")
print(f"Generated text: '{pre_export_text}'")

# ============================================================
# Export with dynamic shapes
# ============================================================
print("\n" + "=" * 60)
print("=== EXPORTING MODEL ===")
print("=" * 60)

dummy_input = torch.randint(0, 1000, input_ids.shape, device="cuda")
seq_len_dim = Dim("seq_len", min=1, max=2048)
dynamic_shapes = ({1: seq_len_dim},)

try:
    gm = torch_export_to_gm(model, (dummy_input,), dynamic_shapes=dynamic_shapes)
    print("Export with dynamic shapes successful!")
    use_dynamic = True
except Exception as e:
    print(f"Export with dynamic shapes failed: {e}")
    print("Falling back to static shapes...")
    gm = torch_export_to_gm(model, (dummy_input,))
    print("Export with static shapes successful!")
    use_dynamic = False

# ============================================================
# Test 2: Original model (after export) - check if export mutated the model
# ============================================================
print("\n" + "=" * 60)
print("=== TEST 2: ORIGINAL MODEL (after export) ===")
print("=" * 60)

post_export_tokens = generate_tokens(model, input_ids, num_gen_tokens)
post_export_text = tokenizer.decode(post_export_tokens)
print(f"Generated tokens: {post_export_tokens}")
print(f"Generated text: '{post_export_text}'")

# ============================================================
# Test 3: Exported GraphModule
# ============================================================
print("\n" + "=" * 60)
print("=== TEST 3: EXPORTED GRAPHMODULE ===")
print("=" * 60)

if use_dynamic:
    # With dynamic shapes, we can do full token generation
    gm_tokens = generate_tokens(gm, input_ids, num_gen_tokens)
    gm_text = tokenizer.decode(gm_tokens)
    print(f"Generated tokens: {gm_tokens}")
    print(f"Generated text: '{gm_text}'")
else:
    # With static shapes, only test context phase (single forward pass)
    print("(Static shapes - testing context phase only)")
    with torch.no_grad():
        gm_logits = extract_logits(gm(input_ids))
        gm_first_token = gm_logits[0, -1].argmax().item()
    gm_tokens = [gm_first_token]
    gm_text = tokenizer.decode(gm_tokens)
    print(f"First token: {gm_tokens}")
    print(f"Generated text: '{gm_text}'")
