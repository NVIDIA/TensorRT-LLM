# VBench and Similarity Evaluation

This directory contains example scripts demonstrating how to use VBench and similarity metrics to evaluate the generated videos.

## Video Generation

We support two video generation modes, which are controlled by the `multiple_prompts` argument:

- **Generating a Single Video (Default)**: By default, the `multiple_prompts` is set to `False` for single video generation. For detailed instructions on this mode, please refer to the guide in [**`examples/README.md`**](./README.md).

- **Generating Multiple Videos**: To generate multiple videos from a prompt file, set the `multiple_prompts` parameter to `True`. **Note:** The path specified by `--prompt` augument must be a **JSON or TXT** file that contains a list of prompts. You can use the standard prompt files provided by [**VBench**](https://github.com/Vchitect/VBench/tree/master/prompts), or you can specify a path to a custom file. 

## VBench

VBench provides a fine-grained and objective evaluation of video generation quality across multiple dimensions. VBench can be used to evaluate any video. The `videos_path` argument accepts a path to either a single video file or a directory containing multiple videos.


### VBench Installation
```bash
pip install vbench --no-deps
pip install -r ../tools/vbench_requirements.txt

apt-get update
apt-get install -y libgl1
```

**Note:** It is recommended to create a separate environment for `VBench`, as its dependency on the `transformers` library conflicts with us.

### VBench Dimension Explanationm
**Quality Dimensions:**
- **Subject Consistency**: Access subject appearance consistency by calculating DINO feature similarity across fames.
- **Background Consistency**: Access backgroud scenes consistency by calculating CLIP feature similarity across frames.
- **Temporal Flickering**: Access temporal consistency between local and high-frequency by calculating the mean absolute difference across frames.
- **Motion Smoothness**: Evaluate the smoothness of movement and motion utilizing the video frame interpolation model.
- **Dynamic Degree**: Estimate the degree of dynamics using RAFT.
- **Aesthetic Quality**: Evaluate the artistic and beauty value using the LAION aesthetic predictor.
- **Imaging Quality**: Evaluate the distortion (e.g., over-exposure) using the MUSIQ image quality predictor.

**Semantic Dimensions:**
- **Object Class**: Detect the success rate of generating specific objects in text prompt using GRiT.
- **Multiple Objects**: Detect the success rate of generating all objects in text prompts using GRiT.
- **Human Action**: Evaluate whether human subjects in generated videos accurately execute the specific actions mentioned in the text prompts using UMT.
- **Color**: Evaluate whether the colors of generated object align with the prompt using GRiT.
- **Spatial Relationship**: Evaluate whether the spatial relationship of generated objects align with the prompt using rule-based evaluation.
- **Scene**: Evaluate whether the generated scene align with the prompt using Tag2Text.
- **Appearance Style**: Evaluate the appearance style (e.g., oil painting style) consistency between generated videos and prompts using CLIP feature similarity.
- **Temporal Style**: Evaluate the temporal style (e.g., camera motions) consistency between generated videos and prompts using ViCLIP feature similarity.
- **Overall Consistency**: Evaluate the overall consistency between generated videos and prompts using ViCLIP.

### Usage
VBench supports evaluations with three prompt configurations: the standard dimension prompts, the complete suite of standard prompts, or the custom prompts.

#### 1. Use VBench Standard Dimension Prompts

If you want to evaluate the performance of a text-to-video (T2V) model on specific dimensions, you can use the vbench standard dimension prompts to generate videos and then assess them with VBench. The supported dimensions are:

```
'subject_consistency', 'temporal_flickering', 'object_class', 'multiple_objects', 'human_action', 'color', 'spatial_relationship', 'scene', 'temporal_style', 'appearance_style', 'overall_consistency'
```
When using vbench standard dimension prompts to generate videos, set the `multiple_prompts` parameter to `True`. The supported dimension prompt files can be downloaded [here](https://github.com/Vchitect/VBench/tree/master/prompts/prompts_per_dimension).

**Quick Start for Video Generation**
 
Wan
```bash
python examples/wan_t2v.py \
    --model_path /path/to/the/Wan/T2V/model/ \
    --prompt /path/to/the/dimension/prompt/json/or/txt/file/ \
    --output_path /path/to/save/videos/ \
    --multiple_prompts True
```

Cosmos
```bash
python examplescosmos_t2v.py \
    --model_path /path/to/the/Cosmos/T2V/model/ \
    --prompt /path/to/the/dimension/prompt/json/or/txt/file/ \
    --output_path /path/to/save/videos/ \
    --multiple_prompts True
```
More generation arguments can be found [here](./README.md).

**Quick Start for VBench Evaluation**
```bash
vbench evaluate \
    --dimension "subject_consistency temporal_flickering" \
    --videos_path /path/to/video/folder/ \
    --output_path /path/to/save/result/
```
#### 2. Use VBench Standard Prompts
For a comprehensive evaluation of a T2V model across all dimensions, you can utilize the full suite of VBench standard prompts [here](https://github.com/Vchitect/VBench/blob/master/prompts/all_dimension.txt).

The complete dimensions can be categorized into two types: **Quality** and **Semantic** dimensions:
- **Quality dimensions**: 'subject_consistency', 'background_consistency', 'temporal_flickering', 'motion_smoothness', 'dynamic_degree', 'aesthetic_quality', 'imaging_quality'
- **Semantic dimensions**: 'object_class', 'multiple_objects', 'human_action', 'color', 'spatial_relationship', 'scene', 'temporal_style', 'appearance_style', 'overall_consistency'

After computing the scores for all dimensions, you can use [cal_vbench_total_score.py](../tools/cal_vbench_total_score.py) to calculate the Quality Score, Semantic Score, and Total Score.

**Quick Start for Video Generation**

Wan
```bash
python examples/wan_t2v.py \
    --model_path /path/to/the/Wan/T2V/model/ \
    --prompt /path/to/the/standard/prompt/json/or/txt/file/ \
    --output_path /path/to/save/videos/ \
    --multiple_prompts True 
```

Cosmos
```bash
python examples/cosmos_t2v.py \
    --model_path /path/to/the/Cosmos/T2V/model/ \
    --prompt /path/to/the/standard/prompt/json/or/txt/file/ \
    --output_path /path/to/save/videos/ \
    --multiple_prompts True
```

**Quick Start for VBench Evaluation**
```bash
vbench evaluate \
    --dimension "subject_consistency temporal_flickering color human_action" \
    --videos_path /path/to/video/folder/ \
    --output_path /path/to/save/result/
```

**Caculate Quality, Semantic, and Total Score**
```bash
python  ../tools/cal_vbench_total_score.py --eval_results_path /path/to/the/eval_results/json/file
```

#### 3. Use Custom Prompts
The two standard VBench evaluation modes above require generating five videos for the entire prompt suite, which can be inflexible and time-consuming. For a more flexible and faster workflow, we recommend using your own custom prompts. 

A commonly used prompt file PenguinVideoBenchmark from HunyuanVideo can be downloaded [here](https://github.com/Tencent-Hunyuan/HunyuanVideo/blob/main/assets/PenguinVideoBenchmark.csv). Before generating videos, remember to extract the prompts from the downloaded CSV file and save them into a JSON or TXT file. We provide a Python script to perform the file conversion:
```python
# code to extract prompts from PenguinVideoBenchmark.csv and save as PenguinVideoBenchmark.json
import csv
import json
prompt_data = []
with open('PenguinVideoBenchmark.csv', mode='r', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        prompt_data.append(row['prompt'])
json.dump(prompt_data, open('PenguinVideoBenchmark.json', 'w'), indent=4)
```

**Quick Start for Video Generation**

Wan
```bash
python examples/wan_t2v.py \
    --model_path /path/to/the/Wan/T2V/model/ \
    --prompt /path/to/the/custom/prompt/json/or/txt/file/ \
    --output_path /path/to/save/videos/ \
    --multiple_prompts True 
```

Cosmos
```bash
python examples/cosmos_t2v.py \
    --model_path /path/to/the/Cosmos/T2V/model/ \
    --prompt /path/to/the/custom/prompt/json/or/txt/file/ \
    --output_path /path/to/save/videos/ \
    --multiple_prompts True 
```

**Quick Start for VBench Evaluation**
To evaluate videos that are not generated from VBench standard prompts, the `mode` parameter must be set to `'custom_input'` and the supported dimensions are: `'subject_consistency', 'background_consistency', 'motion_smoothness', 'dynamic_degree', 'aesthetic_quality', 'imaging_quality'`.

```bash
vbench evaluate \
    --dimension "subject_consistency background_consistency aesthetic_quality imaging_quality motion_smoothness dynamic_degree" \
    --videos_path /path/to/folder_or_video/ \
    --output_path /path/to/save/result/ \
    --prompt_file /path/to/video/prompts/ \
    --mode=custom_input
```

**NOTE:** You can supply custom prompts using the `--prompt_file` parameter, which points to a JSON file.
The JSON file must contain a single dictionary that maps the path to each video (key) to its corresponding text prompt (value).

## Similarity Metrics(PSNR, SSIM, LPIPS)
The Similarity Metrics are designed to quantify the resemblance between videos generated using sparse attention and the reference videos generated with full attention. This helps assess how well the generation quality is maintained while conserving computational resources.

This benchmark includes the following three core metrics:

* **Peak Signal-to-Noise Ratio (PSNR)**
    * This is a classical metric for image quality assessment based on the pixel-wise Mean Squared Error (MSE). It measures the ratio between the maximum possible power of a signal and the power of corrupting noise that affects its fidelity. 
    * **A higher PSNR value indicates less distortion and higher reconstruction fidelity.**

* **Structural Similarity Index Measure (SSIM)**
    * SSIM is a perceptual metric that aligns more closely with human visual perception than PSNR. It assesses similarity by comparing three key features: luminance, contrast, and structure. 
    * **The SSIM value ranges from -1 to 1, where a value closer to 1 indicates a higher degree of structural similarity.**

* **Learned Perceptual Image Patch Similarity (LPIPS)**
    * This is a modern, deep learning-based metric that measures perceptual similarity. It compares high-level features extracted from deep neural networks (e.g., VGG) to determine how similar two images are, which is considered to be highly correlated with human judgment. 
    * **A lower LPIPS score signifies that the two videos are more perceptually similar.**

The similarity metrics script supports the following two input modes:
* **File-to-File Comparison:** Provide the direct paths to two individual video files to compare them.
* **Directory-to-Directory Comparison:** Provide the paths to two directories for a batch comparison. **Please note:** When comparing directories, you must ensure that both contain the same number of videos. The script will perform a one-to-one comparison, typically by matching sorted filenames.

### Installation
```
pip install lpips
```
### Quick Start
```bash
python tools/similarity_metrics.py \
    --original_path /path/to/reference/video/or/directory \
    --generated_path /path/to/generated/video/or/directory \
    --output_path /path/to/save/result/
```
