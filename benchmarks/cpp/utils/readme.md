# Benchmarking Utils
The logic in this directory supports common benchmarking scenarios within TensorRT-LLM.

## Generating Datasets
One of the problems this logic attempts to solve, is generating input data for benchmarks.

### Extending Dataset Generation
We assume there are two ways one might extend dataset generation:
- Changing the content of the dataset, e.g. allowing dataset generation to support generating inputs for multi-model models.
- Changing the format in which it is exported, e.g. making dataset generation support export for consumption by a new benchmarking tool.

If you wish to change the **content** of a dataset, please adjust `generate.py`.

If you wish to change the **format** of a dataset, please adjust `export.py`.
