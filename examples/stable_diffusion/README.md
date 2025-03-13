# Stable Diffusion

This document is used to demonstrate how to run [Stable Diffusion v1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5) with TensorRT-LLM on single GPU.
As the stable diffusion is composed of three models(Clip,U-net and Vae), TensorRT-LLM only enables the U-Net part on this pipeline and the rest two models are enabled via Pytorch.

## Prerequisite

Please install the git lfs firstly with below commands.

```bash
sudo apt-get update && sudo apt-get install git-lfs
```

Then, please download the stable diffusion v1.5 checkpoints with your huggingface token.

 ```bash
 cd examples/stable_diffusion && git lfs install
 ```

 ```bash
 git clone https://huggingface.co/runwayml/stable-diffusion-v1-5
 ```


## Run the demo

### 1. Build the engine

```python
python3 build.py
```

`build.py` supports below parameters:
* `--dtype` the precision of the engine, default value is `float16`, and it could be `float32` either.
* `--log_level` the level of log severity. Default value is `info`.
* `--output_dir` the output directory name of the generated engine file.
* `--model_dir`, the pre-downloaded weights of stable diffusion model.


### 2. Run

```python
python3 demo.py
```

`demo.py` has below five parameters.
* `--prompt`, the input text for stable diffusion.
* `--image`, the output path of generated image. Default value is **`image.png`**.
* `--log_level`, the level of log severity. Default value is `info`.
* `--unet_engine`, the pre-built engine file's path.
* `--model_dir`, the pre-downloaded weights of stable diffusion model.

The demo would generate the `image.png` under the `examples/stable_diffusion` folder by default.
