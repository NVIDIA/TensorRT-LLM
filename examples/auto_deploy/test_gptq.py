# A tmp example file for GPTQ model patch
import types

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import tensorrt_llm._torch.auto_deploy.custom_ops  # noqa: F401
from tensorrt_llm._torch.auto_deploy.export import torch_export_to_gm


def _torch_quant_linear_forward_patched(self, x: torch.Tensor, out_shape):
    out_flat = torch.ops.auto_deploy.torch_fake_quant_int4_gptq_linear(
        x,
        self.qweight,
        self.qzeros,
        self.scales,
    )

    out = out_flat.reshape(out_shape)

    if self.bias is not None:
        out.add_(self.bias)

    if self.adapter:
        out = self.adapter.apply(x=x, out=out)

    return out


def patch_torch_quant_linear(module_or_model):
    for _, m in getattr(module_or_model, "named_modules", lambda: [])():
        if type(m).__name__ == "TorchQuantLinear":
            # Skip if already patched
            if not hasattr(m, "__original__forward"):
                m.__original__forward = m._forward
                m._forward = types.MethodType(_torch_quant_linear_forward_patched, m)
    # Support direct class instance patch as well
    if type(module_or_model).__name__ == "TorchQuantLinear":
        m = module_or_model
        if not hasattr(m, "__original__forward"):
            m.__original__forward = m._forward
            m._forward = types.MethodType(_torch_quant_linear_forward_patched, m)


model_name = "Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int4"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    use_cache=False,
).to("cuda")
patch_torch_quant_linear(model)
print(model)


# model is now exportable
x = torch.randn(2, 10, device="cuda")
gm = torch_export_to_gm(model, (x,))
# gm.print_readable()

tokenizer = AutoTokenizer.from_pretrained(model_name)
prompt = "How big is the universe?"
messages = [
    {
        "role": "system",
        "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
    },
    {"role": "user", "content": prompt},
]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
generated_ids = model.generate(**model_inputs, max_new_tokens=512)
generated_ids = [
    output_ids[len(input_ids) :]
    for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)
# Output:
# The universe is an enormous and complex concept that has puzzled humans for centuries. The size of the universe can be
# measured in billions or even trillions of light-years.
# The observable universe is about 138 billion light-years across, which is roughly equivalent to the distance from our
# planet to the Sun. This measurement includes everything we can see with our eyes,...
