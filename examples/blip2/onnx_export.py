import os

import requests
import torch
from PIL import Image

# In docker environment
if os.getcwd().startswith('/workspace'):
    os.environ['TORCH_HOME'] = '/workspace/.cache'
    os.environ['TRANSFORMERS_CACHE'] = '/workspace/.cache'

from lavis.models import load_model_and_preprocess

img_url = 'https://storage.googleapis.com/sfr-vision-language-research/LAVIS/assets/merlion.png'
raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')

device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
model, vis_processors, _ = load_model_and_preprocess(
    name="blip2_opt",
    model_type="pretrain_opt2.7b",
    is_eval=True,
    device=device)

image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

torch.save(model.query_tokens, 'query_tokens.pt')

if not os.path.exists('image.pt'):
    torch.save(image, 'image.pt')

txt_caption = model.generate({
    "image": image,
    "prompt": "Question: which city is this? Answer:"
})
print(txt_caption)

visual_wrapper = torch.nn.Sequential(model.visual_encoder, model.ln_vision)
visual_wrapper.float()
image_embeds = visual_wrapper(image)
# torch.save(image_embeds, 'image_embeds.pt')

os.system('mkdir -p ./onnx/visual_encoder')
torch.onnx.export(visual_wrapper.cpu(),
                  image.cpu(),
                  './onnx/visual_encoder/visual_encoder.onnx',
                  opset_version=17,
                  input_names=['input'],
                  output_names=['output'],
                  dynamic_axes={'input': {
                      0: 'batch'
                  }})

image_atts = torch.ones(image_embeds.size()[:-1],
                        dtype=torch.long).to(image.device)
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


q_wrapper = Qformer_wrapper(model.Qformer.bert, model.opt_proj)
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
