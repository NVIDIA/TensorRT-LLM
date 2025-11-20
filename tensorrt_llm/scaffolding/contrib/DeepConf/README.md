# DeepConf

This document shows how to speed up reasoning models without training or fine-tuning by using **DeepConf** ([Deep Think with Confidence](https://arxiv.org/abs/2508.15260)) in TensorRT-LLM.

## Overview

Deep Think with Confidence (DeepConf) is a parallel thinking method that enhances both LLM reasoning performance and efficiency at test time. It leverages model-internal confidence signals to dynamically filter low-quality reasoning traces during or after generation. It requires no additional model training or hyperparameter tuning and can be seamlessly integrated into existing serving frameworks. It achieves up to 99.9% accuracy on AIME 2025 while reducing generated tokens by up to 84.7% compared to standard thinking approaches.

## Usage

The core logic for **DeepConf** lives in `deep_conf_controller.py`, which contains four core classes organized in two layers:

**Base Controllers** (building blocks):
1. **DeepConfOfflineController**: Wraps generation with confidence tracking, collecting logprobs for all generated tokens to build a `ConfidenceInfo` object for post-generation analysis. Serves as the foundation for offline voting and warmup phases.
2. **DeepConfOnlineController**: Implements streaming generation with real-time confidence monitoring and dynamic early stopping when confidence thresholds are met. Serves as the foundation for online voting with early termination.

**Voting Controllers** (composed from base controllers):

> Thanks to the high flexibility of the **scaffolding framework**, we can easily leverage base controllers to handle confidence-related operations, while voting controllers remain focused on algorithmic logic without worrying about confidence tracking or online early-exit implementation details.


3. **DeepConfOfflineMajorityVoteController**: Orchestrates parallel generation using multiple `DeepConfOfflineController` instances, then aggregates results via configurable majority voting strategies.
4. **DeepConfOnlineMajorityVoteController**: Two-phase orchestration combining both base controllers: uses `DeepConfOfflineController` for warmup samples to calibrate thresholds, then `DeepConfOnlineController` for final samples with early stopping, aggregating all results through majority voting.


You can adjust the behavior of DeepConf by passing different parameter values:

| Parameter | Description |
|-----------|-------------|
| `warmup_sample_num` | Number of warmup samples for calibrating confidence threshold |
| `sample_num` | Total samples for majority voting (warmup + final) |
| `conf_group_size` | Token chunk size for confidence checking intervals |
| `conf_threshold` | Base confidence threshold for early stopping |
| `confidence_percentile` | Percentile for computing threshold from warmup (lower = earlier stopping) |
| `logprobs_topk` | Number of top logprobs to track per token |

### Quick Start

#### Offline Mode

```bash
python3 examples/scaffolding/contrib/DeepConf/run_generation.py --model_dir deepseek-ai/DeepSeek-R1-0528-Qwen3-8B  --run_type offline_majority_vote
```

#### Online Mode

```bash
python3 examples/scaffolding/contrib/DeepConf/run_generation.py --model_dir deepseek-ai/DeepSeek-R1-0528-Qwen3-8B  --run_type online_majority_vote
```

> **Note**: `run_generation.py` supports various configurable parameters (e.g., `--sample_num`, `--conf_group_size`, `--confidence_percentile`). See the parameter table above or check the code for detailed options.

## Results

Evaluated on the **brumo_2025.jsonl** dataset with the configuration of `warmup_sample_num=16`, `sample_num=256`, `conf_group_size=2048`, `confidence_percentile=10`, and `logprobs_topk=20`, the online mode achieves a 54.5% reduction in output tokens and approximately 1.92x speedup.

| Mode    | Mean Gen Time   | Mean Tokens   |
|---------|-----------------|---------------|
| Online  | 1506.4s         | ~2.0M         |
| Offline | 2891.4s         | ~4.4M         |

Under the same configuration, confidence-based voting methods significantly improve accuracy, with `top10_bottom_window_filtered` boosting the accuracy from 88.14% (`basic_majority_vote`) to 94.92%.

| Vote Policy                  |   Accuracy |
|------------------------------|------------|
| top10_bottom_window_filtered |     0.9492 |
| top10_tail_filtered          |     0.9153 |
| mean_confidence_weighted     |     0.8983 |
| tail_confidence_weighted     |     0.8983 |
| bottom_window_weighted       |     0.8983 |
| min_window_weighted          |     0.8983 |
| basic_majority_vote          |     0.8814 |
| single_vote                  |     0.7966 |

## References

- Blog post: [Deep Think with Confidence](https://jiaweizzhao.github.io/deepconf/)
- Paper: [https://arxiv.org/abs/2508.15260](https://arxiv.org/abs/2508.15260)
- Codebase: [https://github.com/facebookresearch/deepconf](https://github.com/facebookresearch/deepconf)

If you use DeepConf for your research, please cite the [paper](https://arxiv.org/abs/2508.15260):
 ```
@article{fu2025deep,
  title={Deep think with confidence},
  author={Fu, Yichao and Wang, Xuewei and Tian, Yuandong and Zhao, Jiawei},
  journal={arXiv preprint arXiv:2508.15260},
  year={2025}
}
 ```
