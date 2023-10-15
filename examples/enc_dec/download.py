import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")

input_ids = tokenizer("translate English to German: The house is wonderful.",
                      return_tensors="pt").input_ids
outputs = model.generate(input_ids, decoder_input_ids=torch.IntTensor([[
    0,
]]))
print("input", input_ids, "\noutput", outputs)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

torch.save(model.state_dict(), './models/t5_small.ckpt')

for k, v in model.state_dict().items():
    print(k)
