# Accuracy Test Suite

This folder maintains an accuracy test suite of TensorRT-LLM. Our CI/CD workflow and QA cycles run these tests to protect the model implementations from accuracy regressions.

This test suite employs a *hypothesis testing* methodology, which decides the evaluation sample volume and accuracy thresholds based on objective statistics. This prevents the thresholds from being neither
* too close to the reference (so the tests *intermittently fail* for reasonable accuracy variance) nor
* too far away from the reference (so the tests *always pass* even accuracy regresses).

In addition, most tests are based on the official offline API -- [LLM API](https://nvidia.github.io/TensorRT-LLM/llm-api/index.html). Hence, the tests can easily leverage inflight fused batching and other performance optimizations, and thus run efficiently. Compared with the online API [trtllm-serve](https://nvidia.github.io/TensorRT-LLM/commands/trtllm-serve.html), offline API provides clearer error messages and easier for debugging.

This test suite is organized as following:
* [accuracy_core.py](./accuracy_core.py) provides the test harness, including hypothesis testing logics, evaluation task configurations, and common utilities.
* [test_cli_flow.py](./test_cli_flow.py) contains the tests with CLI workflow, i.e., checkpoint conversion, engine building and evaluation.
* [test_llm_api.py](./test_llm_api.py) contains the tests with LLM API and TensorRT backend.
* [test_llm_api_pytorch.py](./test_llm_api_pytorch.py) contains the tests with LLM API and PyTorch backend.
* [references](./references) registers the reference accuracies for each task, each model and each specification (e.g., data type, quantization).
* [scripts](./scripts) provides some utility scripts that may help setup accuracy tests.

Currently, the below tasks are supported.

| Dataset | Task | Metric | LLM API | CLI flow |
|:-------:|:----:|:------:|:-------:|:--------:|
| CNN Dailymail     | summarization       | rouge      | Y | Y |
| MMLU              | QA; multiple choice | accuracy   | Y | Y |
| GSM8K             | QA; regex matching  | accuracy   | Y | N |
| GPQA              | QA; multiple choice | accuracy   | Y | N |
| Humaneval         | code completion     | rouge*     | N | Y |
| ZeroScrolls       | summarization       | rouge      | N | Y |
| Passkey retrieval | retrieval           | accuracy   | N | Y |
| SlimPajama-6B     | perplexity          | perplexity | N | Y |

\* Rouge is an informal evaluation metric for code completion.

New accuracy tests are strongly recommended to be added to this test suite, in particular, in the LLM API style (i.e., [test_llm_api.py](./test_llm_api.py) and [test_llm_api_pytorch.py](./test_llm_api_pytorch.py)). There are some legacy accuracy tests outside this test suite (e.g., the tests in [examples](../examples) folder), but they are not recommended anymore.


## Background: Why This Test Suite?

It probably seems simple to setup an accuracy test:
* Decide the dataset and task; if the dataset is large, optionally decide the sample volume of a subset.
* Evaluate the model on the subset and obtain a reference accuracy.
* Setup a threshold slightly lower than the reference accuracy.
* Implement the testing code that automatically runs the same evaluation and compares the resulted accuracy to the threshold.
    * If the evaluated accuracy is higher than the threshold, the test passes.
    * If the evaluated accuracy is lower than the threshold, the test fails.

Once implemented, the test will be run in the CI/CD workflow or QA cycles, to protect the model from accuracy regression due to future code changes.

The above steps are quite intuitive except for a seemingly trivial question: How to decide the sample volume and threshold?

According to our engineering experience, a model's accuracy can slightly vary because it *reasonably* executes on different kernels (e.g., different batch sizes, fusion patterns, kernel implementations, hardwares). That means, a model's accuracy can slightly drop but it doesn't mean accuracy regression. Another engineering insight is that increasing the sample volume can reduce the evaluated accuracy variance. This is also intuitive because the evaluated accuracy is typically averaged over sample scores, and sampled average score’s variance is inversely proportional to the sample volume ([central limit theorem](https://en.wikipedia.org/wiki/Central_limit_theorem)).

Thus, it becomes tricky when deciding the sample volume and threshold:
* Higher sample volume reduces the evaluated accuracy variance.
* Lower sample volume reduces the test cost.
* Higher threshold makes the test more stict, so that accuracy regression can be easily detected.
* Lower threshold makes the test more robust, so that reasonable accuracy variance can be ignored.

From the statistics view, we are balancing several conflicting objectives:
* Minimize sample volume $n$.
* Minimize minimum detectable effect $\theta$ (the minimum accuracy difference regarded as a regression).
* Minimize false positive rate $\alpha$ (the probability that the test fails but accuracy does not regress).
* Minimize false negative rate $\beta$ (the probability that the test passes but accuracy regresses).

Increasing $n$ allows lower $\theta$, $\alpha$ and $\beta$. Given $n$ and $\theta$, threshold setting is trading off $\alpha$ and $\beta$. Hypothesis testing provides a rigorous solution to the balance.

Within this solution, we first decide $\theta$, $\alpha$ and $\beta$, and then compute the least required sample volume and threshold. To clarify, the three parameters $\theta$, $\alpha$ and $\beta$ don't increase the complexity to add a new test. Existing evaluation tasks have provide these parameters, and new tasks can use default values: $\theta = 2$ (scoring from 0 to 100), $\alpha = 0.05$ and $\beta = 0.2$.

In addition to the hypothesis testing framework, this accuracy test suite provides:
* Test harness, which encapsulates common test logics.
    * The new test functions are simplified in relative to the legacy style.
* Centralized accuracy reference registration.
    * The accuracy references are registered in YAML files in [references](./references), in stead of being hard-coded in testing code.
    * The accuracy references are categorized by tasks, models and accuracy specifications, which allows fine-grained management.
* Performant evaluation.
    * The test harness utilizes our own inference optimizations, which accelerates accuracy evaluation.


## How to Add Accuracy Tests



### Understanding Existing Tasks and Test Cases

Given an existing task, $\theta$, $\alpha$, $\beta$ and $n$ are all configured. For example, in [accuracy_core.py](./accuracy_core.py) the MMLU task is defined as following:

```python
class MMLU(AccuracyTask):
    ...
    ALPHA = 0.01
    BETA = 0.2
    SIGMA = 50
    NUM_SAMPLES = 4096
    ...
```

The parameters $\alpha$, $\beta$ and $n$ are clearly present. With an additional score variance $\sigma$, the parameter $\theta$ can be computed (See formulas in [Hypothesis Testing Methodology](#hypothesis-testing-methodology) or function `compute_theta` in [accuracy_core.py](./accuracy_core.py)).

Each test case aims to evaluate the accuracy of a model with some accuracy specifications (e.g., data type, quantization). To achieve this, it runs one or more evaluation tasks.

For each task, the test case looks for the reference accuracy from the YAML files in [references](./references). Together with other parameters, the threshold can be computed (See formulas in [Hypothesis Testing Methodology](#hypothesis-testing-methodology) or function `compute_threshold` in [accuracy_core.py](./accuracy_core.py)). Then, the test case runs the model on the task (a subset of volume $n$) and get the evaluated accuracy.

If all the evaluated accuracies are equal to or higher than the corresponding thresholds, the test passes.

### Add New Test Cases with Existing Tasks

We suggest supporting your model with LLM API, and then add tests to [test_llm_api.py](./test_llm_api.py) or [test_llm_api_pytorch.py](./test_llm_api_pytorch.py). Typically, a test class is responsible for a model (corresponding to a unique Hugging Face model ID); it contains several test methods for different features (e.g., quantizations, parallelisms). For example, in [test_llm_api_pytorch.py](./test_llm_api_pytorch.py) the model [`meta-llama/Llama-3.1-8B-Instruct`](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) has the test class defined as:

```python
class TestLlama3_1_8BInstruct(LlmapiAccuracyTestHarness):
    MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
    MODEL_PATH = f"{llm_models_root()}/llama-3.1-model/Llama-3.1-8B-Instruct"

    def test_bfloat16(self, ...):
        # create an LLM instance with tested features enabled, optionally with pytest parameters
        llm = LLM(self.MODEL_PATH, ...)
        # use a context manager to explicitly deconstruct the LLM instance upon exiting
        with llm:
            # create and run the MMLU task
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm)
            # create and run the GSM8K task
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)
```

Please inherit `LlmapiAccuracyTestHarness` when defining a new test class. At the class level, `MODEL_NAME` is the unique Hugging Face model ID; it is also the key to find the accuracy references in the registration files. `MODEL_PATH` is the Hugging Face model checkpoint path accessible by the test machine. Normally, `MODEL_NAME` and `MODEL_PATH` should be the non-quantized version.

At the method level, the test code should create an LLM instance with the tested features enabled, and then create and run task instances one by one. Existing tasks are imported from [accuracy_core.py](./accuracy_core.py).

TODO: accuracy specs and yaml files

NOT support LLM Api?

### Add New Tasks

* Estimate $\sigma$ from the full dataset.
* Decide a target minimum detectable effect $\theta$ based on the nature of dataset and corresponding accuracy metric.
* Decide $\alpha$ and $\beta$ based on the importance of model.
* Iterate sample volume $n$ from small to large, and compute $\theta$ until it satisfies (is equal to or lower than) the target $\theta$.
* Evaluate the model on the subset of sample volume $n$, resulting in the reference accuracy.
* The threshold $\gamma$ is automatically setup based on $\alpha$, $\sigma$, $n$ and the reference accuracy.



## Hypothesis Testing Methodology

### Null Hypothesis and Alternative Hypothesis

For a given dataset and model, the evaluated scores can be viewed as a population with mean $\mu$ and variance $\sigma$. Note that the distribution is not necessarily to be a normal distribution.

When we finish implementing a model, we need to setup an accuracy *reference*. By evaluating the model on a subset of $n$ samples, we practically draw $n$ scores $`x_1, x_2, \dots, x_n`$ from the population, and thus we can compute and record the sample average $`\bar{x} = \frac{1}{n} \sum_{i} x_i`$.

When testing if there is an accuracy *regression*, we once again evaluate the model on $n$ samples, resulting in $`x'_1, x'_2, \dots, x'_n`$, and also sample average $`\bar{x'} = \frac{1}{n} \sum_{i} x'_i`$. The question is that, are these $n$ samples drawn from the same distribution to the referenced one? This can be formulated as a hypothesis testing problem:

* Null Hypothesis ($H_0$): $`x'_1, x'_2, \dots, x'_n`$ are drawn from the same distribution to the reference.
* Alternative Hypothesis ($H_1$): $`x'_1, x'_2, \dots, x'_n`$ are from a different distribution from the reference.

Since we care about accuracy regression only, so it should be a one-tailed hypothesis testing problem:

* Null Hypothesis ($H_0$): $`x'_1, x'_2, \dots, x'_n`$ are drawn from a distribution with a mean equal to or higher than the reference.
* Alternative Hypothesis ($H_1$): $`x'_1, x'_2, \dots, x'_n`$ are drawn from a distribution with a mean lower than the reference.

![Hypothesis Testing](./media/hypothesis-testing.svg)

### Two-Sample T-Test

According to the two-sample t-test method, we can compute the t-statistic $`t = \frac{\bar{x'} - \bar{x}}{\sqrt{2 \sigma^2 / n}}`$. According to the Central Limit Theorem (CLT), the t-statistic is from a distribution that converges to the standard normal distribution $\mathcal{N} (0, 1)$.

Given the threshold $\gamma$, the false positive (type I error) rate $\alpha$ can be formulated as:

$$
\begin{equation*}
\begin{aligned}
\alpha &= P \left(\bar{x'} \leq \gamma \mid t \sim \mathcal{N} (0, 1) \right) \\
&= P \left(t \leq \frac{\gamma - \bar{x}}{\sqrt{2 \sigma^2 / n}} \mid t \sim \mathcal{N} (0, 1) \right).
\end{aligned}
\end{equation*}
$$

In practive, we setup a $\alpha$ (e.g., 0.05) and then compute the threshold $\gamma$:

$$
\begin{equation*}
\gamma = \Phi^{-1} (\alpha) \cdot \sqrt{2 \sigma^2 / n} + \bar{x}.
\end{equation*}
$$

Note that $\alpha$ is typically smaller than 0.5, so $\gamma < \bar{x}$.

Given the minimum detectable effect $\theta$, the false negative (type II error) rate $\beta$ can be formulated as:

$$
\begin{equation*}
\begin{aligned}
\beta &= P \left(\bar{x'} > \gamma \mid t \sim \mathcal{N} (-\frac{\theta}{\sqrt{2 \sigma^2 / n}}, 1) \right) \\
&= P \left(t > \frac{\gamma - \bar{x}}{\sqrt{2 \sigma^2 / n}} \mid t \sim \mathcal{N} (-\frac{\theta}{\sqrt{2 \sigma^2 / n}}, 1) \right) \\
&= P \left(t + \frac{\theta}{\sqrt{2 \sigma^2 / n}} > \frac{\gamma - \bar{x} + \theta}{\sqrt{2 \sigma^2 / n}} \mid t + \frac{\theta}{\sqrt{2 \sigma^2 / n}} \sim \mathcal{N} (0, 1) \right) \\
&= P \left(t + \frac{\theta}{\sqrt{2 \sigma^2 / n}} > \Phi^{-1} (\alpha) + \frac{\theta}{\sqrt{2 \sigma^2 / n}} \mid t + \frac{\theta}{\sqrt{2 \sigma^2 / n}} \sim \mathcal{N} (0, 1) \right)
\end{aligned}
\end{equation*}
$$

In practice, we setup a $\beta$ (e.g., 0.2) and then compute $\theta$:

$$
\begin{equation*}
\begin{aligned}
\theta &= (\Phi^{-1} (1-\beta) - \Phi^{-1} (\alpha)) \cdot \sqrt{2 \sigma^2 / n} \\
&= - (\Phi^{-1} (\alpha) + \Phi^{-1} (\beta)) \cdot \sqrt{2 \sigma^2 / n}
\end{aligned}
\end{equation*}
$$

Note that $\alpha$ and $\beta$ are typical smaller than 0.5, so $\theta > 0$.

References: [student's t-test](https://en.wikipedia.org/wiki/Student%27s_t-test), [statistics test power](https://en.wikipedia.org/wiki/Power_(statistics)).
