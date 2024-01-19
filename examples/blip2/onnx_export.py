import os

import requests
import torch
from PIL import Image
from transformers import Blip2ForConditionalGeneration, Blip2Processor

img_url = 'https://storage.googleapis.com/sfr-vision-language-research/LAVIS/assets/merlion.png'
raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')

device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16)
model.to(device)
prompt = "Question: which city is this? Answer:"
inputs = processor(images=raw_image, text=prompt,
                   return_tensors="pt").to(device, torch.float16)
image = inputs['pixel_values']
for k in inputs.keys():
    print(k, inputs[k].shape)
generated_ids = model.generate(**inputs)
generated_text = processor.batch_decode(generated_ids,
                                        skip_special_tokens=True)[0].strip()
print(generated_text)

if not os.path.exists('query_tokens.pt'):
    torch.save(model.query_tokens, 'query_tokens.pt')

if not os.path.exists('image.pt'):
    torch.save(image, 'image.pt')

visual_wrapper = model.vision_model

vision_outputs = visual_wrapper(image)
image_embeds = vision_outputs[0]
print('image_embeds: ', image_embeds.shape)

os.system('mkdir -p ./onnx/visual_encoder')
torch.onnx.export(visual_wrapper,
                  image,
                  './onnx/visual_encoder/visual_encoder.onnx',
                  opset_version=17,
                  input_names=['input'],
                  output_names=['output'],
                  dynamic_axes={'input': {
                      0: 'batch'
                  }})

image_atts = torch.ones(image_embeds.size()[:-1],
                        dtype=torch.long).to(image_embeds.device)
query_tokens = model.query_tokens.expand(image_embeds.shape[0], -1, -1)


class Qformer_wrapper(torch.nn.Module):

    def __init__(self, Qformer, opt_proj):
        super().__init__()
        self.model = Qformer
        self.opt_proj = opt_proj

    def forward(self, query_tokens, image_embeds, image_atts):
        query_output = self.model(query_embeds=query_tokens,
                                  encoder_hidden_states=image_embeds,
                                  encoder_attention_mask=image_atts,
                                  return_dict=True)
        return self.opt_proj(query_output.last_hidden_state)


q_wrapper = Qformer_wrapper(model.qformer, model.language_projection)
inputs_opt = q_wrapper(query_tokens, image_embeds, image_atts)
# torch.save(inputs_opt, 'inputs_opt.pt')
os.system('mkdir -p ./onnx/Qformer')
torch.onnx.export(q_wrapper, (query_tokens, image_embeds, image_atts),
                  './onnx/Qformer/Qformer.onnx',
                  opset_version=17,
                  input_names=['query_tokens', 'image_embeds', 'image_atts'],
                  output_names=['query_output'],
                  dynamic_axes={
                      'query_tokens': {
                          0: 'batch'
                      },
                      'image_embeds': {
                          0: 'batch'
                      },
                      'image_atts': {
                          0: 'batch'
                      }
                  })
