!!! WARNING: This is not intended for external users to benchmark the performance numbers of the TRT-LLM product.
!!! This folder contains the benchmark script used internally to assistant TRT-LLM development.

# build_time_benchmark

```bash
# example 1: offline benmark for all the built-in models, see --help for all the options
python ./build_time_benchmark.py --model ALL

# By default, the benmark don't load the weights to save benchmark time, load the weights to test the TRT-LLM load and convert time
# WARNING: this can takes very long time if the model is large, or if you use a online HF model id since it can download the weights
python ./build_time_benchmark.py --model ALL --load

# example 2: benchmark a HF model model w/o downloading the model locally in advance
python ./build_time_benchmark.py --model "TinyLlama/TinyLlama_v1.1" # no weights loading
python ./build_time_benchmark.py --model "openai-community/gpt2" --load # with weights loading

# example 3: benchmark a local download HF model
python  ./build_time_benchmark.py --model /home/scratch.trt_llm_data/llm-models/falcon-rw-1b/

# example 4: benchmark one model with managed weights option, with verbose option
python ./build_time_benchmark.py --model llama2-70b.TP4 --managed_weights -v

# example 5: see help
python ./build_time_benchmark.py --help
```
