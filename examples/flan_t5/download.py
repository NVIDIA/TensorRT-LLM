import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import os

model_name = os.environ.get("MODEL_NAME") or "google/flan-t5-xl"
checkpoint_name = os.environ.get("CHECKPOINT_NAME") or "test"
# model_name = os.environ.get("MODEL_NAME") or "GotItAI/distilled_elmar_qna_flan_ul2_ft_on_rocketbook_xl"

tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)
model = T5ForConditionalGeneration.from_pretrained(model_name)

input_ids = tokenizer("translate English to German: The house is wonderful.",
                      return_tensors="pt").input_ids
outputs = model.generate(input_ids, decoder_input_ids=torch.IntTensor([[
    0,
]]))
print("input", input_ids, "\noutput", outputs)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

torch.save(model.state_dict(), f"./models/{checkpoint_name}.ckpt")

for k, v in model.state_dict().items():
    print(k)
