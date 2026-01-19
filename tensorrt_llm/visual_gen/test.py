import torch
import visual_gen
from visual_gen.pipelines.flux2_pipeline import ditFlux2Pipeline
model_id = "/workspace/home/.cache/huggingface/hub/models--black-forest-labs--FLUX.2-dev/snapshots/6aab690f8379b70adc89edfa6bb99b3537ba52a3"


# Define visual_gen configs to setup ops, diffusion cache methods, parallelism, etc.
dit_configs = {
   "teacache": {
       "enable_teacache": True,
       "use_ret_steps": False,
       "teacache_thresh": 0.05,
       "ret_steps": 10,
       "cutoff_steps": 50,
   },
   "attn": {
       "type": "default",
   },
   "linear": {
       "type": "flashinfer-nvfp4-cutlass",
       "choices": "default,flashinfer-nvfp4-cutlass",
       "recipe": "dynamic",
   },
   "parallel": {
       "dit_dp_size": 1,
       "dit_ulysses_size": 1,
       "dit_ring_size": 1,
       "dit_cfg_size": 1,
   },
}

visual_gen.setup_configs(**dit_configs)
model_id = "/workspace/home/.cache/huggingface/hub/models--black-forest-labs--FLUX.2-dev/snapshots/6aab690f8379b70adc89edfa6bb99b3537ba52a3"
exclude_pattern = r"^(?!.*(embedder|to_out|norm_out|proj_out|to_add_out|to_added_qkv|stream)).*"
pipe = ditFlux2Pipeline.load_flux_2_dynamic_quantization(
    model_id=model_id,
    torch_dtype=torch.bfloat16,
    exclude_pattern=exclude_pattern, 
    **dit_configs,
    enable_compile=True,
    enable_cuda_graph=True,
)

prompt = "A cat and a dog baking a cake together in a kitchen. The cat is carefully measuring flour, while the dog is stirring the batter with a wooden spoon. The kitchen is cozy, with sunlight streaming through the window."

image = pipe(prompt=prompt).images[0]


image.save("output_flux.png")
