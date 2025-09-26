# Out-of-tree Model Development
The file `modeling_opt.py` shows an example of how a custom model can be defined using TRT-LLM APIs without modifying the source code of TRT-LLM.

The file `main.py` shows how to run inference for such custom models using the LLM API.


## Out-of-tree Multimodal Models

For multimodal models, TRT-LLM provides `quickstart_multimodal.py` to quickly run a multimodal model that is defined within TRT-LLM. `trtllm-bench` can be used for benchmarking such models.
However, the following sections describe how to use those tools for out-of-tree models.

### Pre-requisite
To use an out-of-tree model with the quickstart example and trtllm-bench, you need to prepare the model definition files similar to a python module.
Consider the following file structure as an example:
```
modeling_custom_phi
|-- __init__.py
|-- configuration.py
|-- modeling_custom_phi.py
|-- encoder
    |-- __init__.py
    |-- configuration.py
    |-- modeling_encoder.py
````
The files `__init__.py` should be populated with the right imports for the custom model. For example, the `modeling_custom_phi/__init__.py` can contain something like:
```
from .modeling_custom_phi import MyVLMForConditionalGeneration
from . import encoder
```

### Quickstart Example

Once the model definition files are prepared as a python module (as described above), you can use the `--custom_module_dirs` flag in `quickstart_multimodal.py` to load your model and run inference.

```
python3 quickstart_multimodal.py --model_dir ./model_ckpt --modality image --max_tokens 10 --prompt "Describe the image." --media ./demo_lower.png --image_format pil --custom_module_dirs ../modeling_custom_phi
```

### Benchmarking

Similar to the quickstart example, you can use the same CLI argument with `trtllm-bench` to benchmark a custom model.

Prepare the dataset:
```
python ./benchmarks/cpp/prepare_dataset.py --tokenizer ./model_ckpt --stdout dataset --dataset-name lmms-lab/MMMU --dataset-split test --dataset-image-key image --dataset-prompt-key "question" --num-requests 100 --output-len-dist 128,5 > mm_data.jsonl
```


Run the benchmark:
```
trtllm-bench --model ./model_ckpt --model_path ./model_ckpt throughput --dataset mm_data.jsonl --backend pytorch --num_requests 100 --max_batch_size 4 --modality image --streaming --custom_module_dirs ../modeling_custom_phi
```
