# Deepseek-v2

This document shows how to build and run [deepseek-v2](https://arxiv.org/pdf/2405.04434) model in TensorRT-LLM.

- [Deepseek-v2](#deepseek-v2)
    - [Prerequisite](#prerequisite)
    - [Hardware](#hardware)
    - [Overview](#overview)
    - [Support Matrix](#support-matrix)
    - [Usage](#usage)
        - [Build TensorRT engine(s)](#build-tensorrt-engines)

## Prerequisite

First, please download Deepseek-v2 weights from HF https://huggingface.co/deepseek-ai/DeepSeek-V2.

```bash
git lfs install
git clone https://huggingface.co/deepseek-ai/DeepSeek-V2
```

## Hardware

The Deepseek-v2 model requires least 8x80G GPU memory, model contains 236B parameters roughly 472GB memory (with BF16 precision).

***Caution: Current TRT-LLM MLA kernel only support Hopper architecture (SM90). Ampere architecture (SM80 & SM86) will be supported in next release.***

## Overview

The TensorRT LLM Deepseek-v2 implementation can be found in [tensorrt_llm/models/deepseek_v2/model.py](../../tensorrt_llm/models/deepseek_v2/model.py). The TensorRT LLM Deepseek-v2 example code is located in [`examples/models/contrib/deepseek_v2`](./). There is one main file:

* [`convert_checkpoint.py`](./convert_checkpoint.py) to convert the Deepseek-v2 model into TensorRT LLM checkpoint format.

In addition, there are three shared files in the parent folder [`examples`](../../../) can be used for inference and evaluation:

* [`../../../run.py`](../../../run.py) to run the model inference output by given an input text.
* [`../../../summarize.py`](../../../summarize.py) to summarize the article from [cnn_dailmail](https://huggingface.co/datasets/abisee/cnn_dailymail) dataset, it can running the summarize from HF model and TensorRT LLM model.
* [`../../../mmlu.py`](../../../mmlu.py) to running score script from https://github.com/declare-lab/instruct-eval to compare HF model and TensorRT LLM model on the MMLU dataset.

## Support Matrix

- [x] BF16
- [ ] FP8

***Caution: prefer using BF16 over FP16 for Deepseek-v2 since model original training precision is BF16 and we found direct convert BF16 -> FP16 will suffer accuracy drop.***

## Usage

The TensorRT LLM Deepseek-v2 example code locates at [examples/models/contrib/deepseek_v2](./). It takes PyTorch weights as input, and builds corresponding TensorRT engines. The number of TensorRT engines depends on the number of GPUs used to run inference.

### Build TensorRT engine(s)

Below is the step-by-step to run Deepseek-v2 with TensorRT-LLM.

First the checkpoint will be converted to the TensorRT LLM checkpoint format by apply [`convert_checkpoint.py`](./convert_checkpoint.py). After that, the TensorRT engine(s) can be build with TensorRT LLM checkpoint.

```bash
# Convert Deepseek-v2 HF weights to TensorRT LLM checkpoint format.
python convert_checkpoint.py --model_dir ./DeepSeek-V2 \
                            --output_dir ./trtllm_checkpoint_deepseek_v2_8gpu_bf16 \
                            --dtype bfloat16 \
                            --tp_size 8

# With '--load_model_on_cpu' option if total GPU memory is insufficient
python convert_checkpoint.py --model_dir ./DeepSeek-V2 \
                            --output_dir ./trtllm_checkpoint_deepseek_v2_cpu_bf16 \
                            --dtype bfloat16 \
                            --tp_size 8 \
                            --load_model_on_cpu
```


We observe use GPUs(8xH200) the checkpoint conversion time took ~ 34 mints, while use CPUs took ~ 21 mints and CPU memory required >= 770GB.

After the checkpoint conversion, the TensorRT engine(s) can be built with the TensorRT LLM checkpoint.

```bash
# Build engine
trtllm-build --checkpoint_dir ./trtllm_checkpoint_deepseek_v2_8gpu_bf16 \
            --output_dir ./trtllm_engines/deepseek_v2/bf16/tp8-sel4096-isl2048-bs4 \
            --gpt_attention_plugin bfloat16 \
            --gemm_plugin bfloat16 \
            --max_batch_size 4 \
            --max_seq_len 4096 \
            --max_input_len 2048 \
            --use_paged_context_fmha enable
```

***Caution: `--max_batch_size` and `--max_seq_len` are the main factors to determine how many GPU memory will be used during runtime, so later when try to run e.g., `summarize.py` or `mmlu.py` or `gptManagerBenchmark.cpp`may need adjust `--max_batch_size` and `--max_seq_len` accordingly to avoid OOM.(meaning rebuild TensorRT engine with smaller `--max_batch_size` and `--max_seq_len` if needed based on GPU memory size), there is beautiful technical log perf-best-practices.md (https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/performance/perf-best-practices.md) explained the mechanism.***

Test the engine with [run.py](../../../run.py) script:

```bash
mpirun --allow-run-as-root -n 8 python ../../../run.py --engine_dir ./trtllm_engines/deepseek_v2/bf16/tp8-sel4096-isl2048-bs4 \
                --tokenizer_dir ./DeepSeek-V2 \
                --max_output_len 40 \
                --input_text "The president of the United States is person who"
```

and the output will be like:

```
[10/28/2024-15:03:14] [TRT-LLM] [I] Load engine takes: 78.31490468978882 sec
[10/28/2024-15:03:14] [TRT-LLM] [I] Load engine takes: 78.31163835525513 sec
[10/28/2024-15:03:14] [TRT-LLM] [I] Load engine takes: 78.31164216995239 sec
[10/28/2024-15:03:14] [TRT-LLM] [I] Load engine takes: 78.31491041183472 sec
[10/28/2024-15:03:14] [TRT-LLM] [I] Load engine takes: 78.3116364479065 sec
[10/28/2024-15:03:14] [TRT-LLM] [I] Load engine takes: 78.3118085861206 sec
[10/28/2024-15:03:14] [TRT-LLM] [I] Load engine takes: 78.3118691444397 sec
[10/28/2024-15:03:14] [TRT-LLM] [I] Load engine takes: 78.31516337394714 sec
Input [Text 0]: "<｜begin▁of▁sentence｜>The president of the United States is person who"
Output [Text 0 Beam 0]: " is elected by the people of the United States to lead the country. The president is the head of the executive branch of the government. The president is also the commander in chief of the armed forces."
[TensorRT-LLM][INFO] Refreshed the MPI local session
[TensorRT-LLM][INFO] Refreshed the MPI local session
[TensorRT-LLM][INFO] Refreshed the MPI local session
[TensorRT-LLM][INFO] Refreshed the MPI local session
[TensorRT-LLM][INFO] Refreshed the MPI local session
[TensorRT-LLM][INFO] Refreshed the MPI local session
[TensorRT-LLM][INFO] Refreshed the MPI local session
[TensorRT-LLM][INFO] Refreshed the MPI local session
```

If we want to evaluate the model summarization ability, we can use [summarize.py](../../../summarize.py) script:

```bash
mpirun --allow-run-as-root -n 8 python ../../../summarize.py --engine_dir ./trtllm_engines/deepseek_v2/bf16/tp8-sel4096-isl2048-bs4 \
                       --hf_model_dir ./DeepSeek-V2 \
                       --data_type bfloat16 \
                       --batch_size 1 \
                       --test_trt_llm \
                       --test_hf
```

and the output will be like:


```
[10/28/2024-16:46:22] [TRT-LLM] [I] HF Generated :
[10/28/2024-16:46:22] [TRT-LLM] [I]  Input : ['(CNN)James Best, best known for his portrayal of bumbling sheriff Rosco P. Coltrane on TV\'s "The Dukes of Hazzard," died Monday after a brief illness. He was 88. Best died in hospice in Hickory, North Carolina, of complications from pneumonia, said Steve Latshaw, a longtime friend and Hollywood colleague. Although he\'d been a busy actor for decades in theater and in Hollywood, Best didn\'t become famous until 1979, when "The Dukes of Hazzard\'s" cornpone charms began beaming into millions of American homes almost every Friday night. For seven seasons, Best\'s Rosco P. Coltrane chased the moonshine-running Duke boys back and forth across the back roads of fictitious Hazzard County, Georgia, although his "hot pursuit" usually ended with him crashing his patrol car. Although Rosco was slow-witted and corrupt, Best gave him a childlike enthusiasm that got laughs and made him endearing. His character became known for his distinctive "kew-kew-kew" chuckle and for goofy catchphrases such as "cuff \'em and stuff \'em!" upon making an arrest. Among the most popular shows on TV in the early \'80s, "The Dukes of Hazzard" ran until 1985 and spawned TV movies, an animated series and video games. Several of Best\'s "Hazzard" co-stars paid tribute to the late actor on social media. "I laughed and learned more from Jimmie in one hour than from anyone else in a whole year," co-star John Schneider, who played Bo Duke, said on Twitter. "Give Uncle Jesse my love when you see him dear friend." "Jimmy Best was the most constantly creative person I have ever known," said Ben Jones, who played mechanic Cooter on the show, in a Facebook post. "Every minute of his long life was spent acting, writing, producing, painting, teaching, fishing, or involved in another of his life\'s many passions." Born Jewel Guy on July 26, 1926, in Powderly, Kentucky, Best was orphaned at 3 and adopted by Armen and Essa Best, who renamed him James and raised him in rural Indiana. Best served in the Army during World War II before launching his acting career. In the 1950s and 1960s, he accumulated scores of credits, playing a range of colorful supporting characters in such TV shows as "The Twilight Zone," "Bonanza," "The Andy Griffith Show" and "Gunsmoke." He later appeared in a handful of Burt Reynolds\' movies, including "Hooper" and "The End." But Best will always be best known for his "Hazzard" role, which lives on in reruns. "Jimmie was my teacher, mentor, close friend and collaborator for 26 years," Latshaw said. "I directed two of his feature films, including the recent \'Return of the Killer Shrews,\' a sequel he co-wrote and was quite proud of as he had made the first one more than 50 years earlier." People we\'ve lost in 2015 . CNN\'s Stella Chan contributed to this story.']
[10/28/2024-16:46:22] [TRT-LLM] [I]
 Reference : ['James Best, who played the sheriff on "The Dukes of Hazzard," died Monday at 88 .\n"Hazzard" ran from 1979 to 1985 and was among the most popular shows on TV .']
[10/28/2024-16:46:22] [TRT-LLM] [I]
 Output : [[' James Best, best known for his portrayal of bumbling sheriff Rosco P. Coltrane on TV\'s "The Dukes of Hazzard," died Monday after a brief illness. He was 88.']]
[10/28/2024-16:46:22] [TRT-LLM] [I] ---------------------------------------------------------
[10/28/2024-16:49:33] [TRT-LLM] [I] TensorRT LLM (total latency: 32.02327513694763 sec)
[10/28/2024-16:49:33] [TRT-LLM] [I] TensorRT LLM (total output tokens: 1394)
[10/28/2024-16:49:33] [TRT-LLM] [I] TensorRT LLM (tokens per second: 43.53083793080361)
[10/28/2024-16:49:33] [TRT-LLM] [I] TensorRT LLM beam 0 result
[10/28/2024-16:49:33] [TRT-LLM] [I]   rouge1 : 17.85755990133811
[10/28/2024-16:49:33] [TRT-LLM] [I]   rouge2 : 6.273032755727469
[10/28/2024-16:49:33] [TRT-LLM] [I]   rougeL : 14.768323033457317
[10/28/2024-16:49:33] [TRT-LLM] [I]   rougeLsum : 15.700915348496391
[10/28/2024-16:49:33] [TRT-LLM] [I] Hugging Face (total latency: 189.76398921012878 sec)
[10/28/2024-16:49:33] [TRT-LLM] [I] Hugging Face (total output tokens: 1376)
[10/28/2024-16:49:33] [TRT-LLM] [I] Hugging Face (tokens per second: 7.2511123197159)
[10/28/2024-16:49:33] [TRT-LLM] [I] HF beam 0 result
[10/28/2024-16:49:33] [TRT-LLM] [I]   rouge1 : 18.542590123197257
[10/28/2024-16:49:33] [TRT-LLM] [I]   rouge2 : 6.345777100488389
[10/28/2024-16:49:33] [TRT-LLM] [I]   rougeL : 15.235695878419156
[10/28/2024-16:49:33] [TRT-LLM] [I]   rougeLsum : 16.64935135356226
```

At last, we can evaluate the model with [mmlu.py](../../../mmlu.py) script:

```bash
# Download MMLU dataset
mkdir mmlu_data && cd mmlu_data
wget https://people.eecs.berkeley.edu/~hendrycks/data.tar && tar -xf data.tar
# Run MMLU evaluation
mpirun --allow-run-as-root -n 8 python ../../../mmlu.py --engine_dir ./trtllm_engines/deepseek_v2/bf16/tp8-sel4096-isl2048-bs4 \
                  --hf_model_dir ./DeepSeek-V2 \
                  --data_type bfloat16 \
                  --batch_size 1 \
                  --test_trt_llm \
                  --test_hf \
                  --data_dir ./mmlu_data/data/
```

and the output will be like:

```
Average accuracy 0.480 - abstract_algebra
Average accuracy 0.741 - anatomy
Average accuracy 0.888 - astronomy
Average accuracy 0.790 - business_ethics
Average accuracy 0.845 - clinical_knowledge
Average accuracy 0.924 - college_biology
Average accuracy 0.600 - college_chemistry
Average accuracy 0.720 - college_computer_science
Average accuracy 0.510 - college_mathematics
Average accuracy 0.751 - college_medicine
Average accuracy 0.618 - college_physics
Average accuracy 0.860 - computer_security
Average accuracy 0.796 - conceptual_physics
Average accuracy 0.675 - econometrics
Average accuracy 0.800 - electrical_engineering
Average accuracy 0.741 - elementary_mathematics
Average accuracy 0.643 - formal_logic
Average accuracy 0.570 - global_facts
Average accuracy 0.910 - high_school_biology
Average accuracy 0.714 - high_school_chemistry
Average accuracy 0.890 - high_school_computer_science
Average accuracy 0.891 - high_school_european_history
Average accuracy 0.909 - high_school_geography
Average accuracy 0.953 - high_school_government_and_politics
Average accuracy 0.826 - high_school_macroeconomics
Average accuracy 0.522 - high_school_mathematics
Average accuracy 0.916 - high_school_microeconomics
Average accuracy 0.576 - high_school_physics
Average accuracy 0.923 - high_school_psychology
Average accuracy 0.704 - high_school_statistics
Average accuracy 0.907 - high_school_us_history
Average accuracy 0.932 - high_school_world_history
Average accuracy 0.834 - human_aging
Average accuracy 0.893 - human_sexuality
Average accuracy 0.909 - international_law
Average accuracy 0.880 - jurisprudence
Average accuracy 0.853 - logical_fallacies
Average accuracy 0.598 - machine_learning
Average accuracy 0.874 - management
Average accuracy 0.953 - marketing
Average accuracy 0.880 - medical_genetics
Average accuracy 0.920 - miscellaneous
Average accuracy 0.850 - moral_disputes
Average accuracy 0.613 - moral_scenarios
Average accuracy 0.830 - nutrition
Average accuracy 0.859 - philosophy
Average accuracy 0.883 - prehistory
Average accuracy 0.635 - professional_accounting
Average accuracy 0.625 - professional_law
Average accuracy 0.857 - professional_medicine
Average accuracy 0.833 - professional_psychology
Average accuracy 0.745 - public_relations
Average accuracy 0.869 - security_studies
Average accuracy 0.935 - sociology
Average accuracy 0.930 - us_foreign_policy
Average accuracy 0.596 - virology
Average accuracy 0.877 - world_religions
Average accuracy 0.632 - math
Average accuracy 0.801 - health
Average accuracy 0.738 - physics
Average accuracy 0.897 - business
Average accuracy 0.914 - biology
Average accuracy 0.677 - chemistry
Average accuracy 0.762 - computer science
Average accuracy 0.832 - economics
Average accuracy 0.800 - engineering
Average accuracy 0.736 - philosophy
Average accuracy 0.821 - other
Average accuracy 0.902 - history
Average accuracy 0.909 - geography
Average accuracy 0.883 - politics
Average accuracy 0.876 - psychology
Average accuracy 0.919 - culture
Average accuracy 0.660 - law
Average accuracy 0.727 - STEM
Average accuracy 0.740 - humanities
Average accuracy 0.873 - social sciences
Average accuracy 0.821 - other (business, health, misc.)
Average accuracy: 0.785
```
