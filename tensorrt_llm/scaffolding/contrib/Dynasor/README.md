# Dynasor

This document shows how to speed up reasoning models without training or fine‑tuning by using **Dynasor** ([Efficiently Scaling LLM Reasoning with Certaindex](https://arxiv.org/abs/2412.20993)) in TensorRT‑LLM.

## Overview

Reasoning models often exhibit poor token efficiency, wasting tokens by second‑guessing themselves. **Dynasor** is a certainty‑based approach that dynamically allocates inference compute for reasoning models and stops inference as soon as the LLM has enough information to make a decision.

Currently, this folder provides only **Dynasor‑CoT**, which applies Chain‑of‑Thought (CoT) reasoning. It optimizes models such as `Deepseek‑R1` and its distilled variants. Support for additional reasoning algorithms (Self‑Consistency, Monte Carlo Tree Search, and Rebase) will be added later.

## Usage

The core logic for **Dynasor‑CoT** lives in the `DynasorGenerationController` class in `dynasor_controller.py`. It extends the base `Controller` and implements certainty‑based stopping.

You can adjust the compute‑saving level by initializing `DynasorGenerationController` with different values for:

- `certainty_threshold`: Number of consecutive identical and confident probe answers required to consider the generation as certain.
- `chunk_size`: Number of tokens to generate per proposal round.

Lowering either value saves more tokens but may risk accuracy.

### Quick Start

1. **Basic usage**
  `DynasorGenerationController` is a compute‑saving alternative to `NativeGenerationController`. To try it, run:
   ```bash
   python examples/scaffolding/contrib/Dynasor/scaffolding_dynasor_run.py
   ```

2. **Add aggregation method**
  You can wrap `DynasorGenerationController` with other controllers—for example, `MajorityVoteController` to perform majority voting:
    ```bash
    python examples/scaffolding/contrib/Dynasor/scaffolding_dynasor_run.py --majority_vote
    ```

 ## References

 - Blog post - Dynasor: More Efficient Chain-of-Thought Through Certainty Probing: https://hao-ai-lab.github.io/blogs/dynasor-cot/
 - Paper - https://arxiv.org/abs/2412.20993
 - Codebase - https://github.com/hao-ai-lab/Dynasor

 If you use Dynasor for your research, please cite our [paper](https://arxiv.org/abs/2412.20993):
 ```
 @article{fu2024efficiently,
   title={Efficiently Scaling LLM Reasoning with Certaindex},
   author={Fu, Yichao and Chen, Junda and Zhu, Siqi and Fu, Zheyu and Dai, Zhongdongming and Zhuang, Yonghao and Ma, Yian and Qiao, Aurick and Rosing, Tajana and Stoica, Ion and Zhang, Hao},
   journal={arXiv preprint arXiv:2412.20993},
   year={2024}
 }
 ```

## Acknowledgments

**Dynasor** in TensorRT‑LLM is built upon the `tensorrt_llm/scaffolding` framework, which supports a variety of inference‑time compute methods—such as chain‑of‑thought, majority voting, best‑of‑N sampling, MCTS, and more. We’re grateful to the original `scaffolding` contributors for their excellent work.

If you’re researching in this area and interested in extending it, you’re warmly invited to contribute your own inference‑time compute methods to `scaffolding`.
