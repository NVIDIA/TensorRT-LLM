import torch
from transformers import WhisperForConditionalGeneration, WhisperTokenizer

tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-tiny.en")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")

# input_ids = tokenizer("translate English to German: The house is wonderful.",
#                       return_tensors="pt").input_ids
# outputs = model.generate(input_ids, decoder_input_ids=torch.IntTensor([[
#     0,
# ]]))
# print("input", input_ids, "\noutput", outputs)
# print(tokenizer.decode(outputs[0], skip_special_tokens=True))

torch.save(model.state_dict(), './models/whisper-tiny-en.ckpt')

for k, v in model.state_dict().items():
    if k.startswith('model.encoder'):
        print(k)
    # if k in ["model.encoder.layers.0.self_attn.v_proj.bias", "model.encoder.layers.0.self_attn.q_proj.bias"]:
    #     print(v.shape, k)