# Accuracy Evaluation

## Hypothesis testing methodology

### Null hypothesis and alternative hypothesis

For a given dataset and model, the evaluated scores can be viewed as a population with mean $\mu$ and variance $\sigma$. Note that the distribution is not necessarily to be a normal distribution.

When we finish implementing a model, we need to setup an accuracy *reference*. By evaluating the model on a subset of $n$ samples, we practically draw $n$ scores $x_1, x_2, \dots, x_n$ from the population, and thus we can compute and record the sample average $\bar{x} = \frac{1}{n} \sum_{i} x_i$.

When testing if there is an accuracy *regression*, we once again evaluate the model on $n$ samples, resulting in $x'_1, x'_2, \dots, x'_n$, and also sample average $\bar{x'} = \frac{1}{n} \sum_{i} x'_i$. The question is that, are these $n$ samples drawn from the same distribution to the referenced one? This can be formulated as a hypothesis testing problem:

* Null Hypothesis ($H_0$): $x'_1, x'_2, \dots, x'_n$ are drawn from the same distribution to the reference.
* Alternative Hypothesis ($H_1$): $x'_1, x'_2, \dots, x'_n$ are from a different distribution from the reference.

Since we care about accuracy regression only, so it should be a one-tailed hypothesis testing problem:

* Null Hypothesis ($H_0$): $x'_1, x'_2, \dots, x'_n$ are drawn from a distribution with a mean equal to or higher than the reference.
* Alternative Hypothesis ($H_1$): $x'_1, x'_2, \dots, x'_n$ are drawn from a distribution with a mean lower than the reference.

![Hypothesis Testing](./media/hypothesis-testing.svg)

### Two-sample t-test

According to the two-sample t-test method, we can compute the t-statistic $t = \frac{\bar{x'} - \bar{x}}{\sqrt{2 \sigma^2 / n}}$. According to the Central Limit Theorem (CLT), the t-statistic is from a distribution that converges to the standard normal distribution $\mathcal{N} (0, 1)$.

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

References:
* https://en.wikipedia.org/wiki/Student%27s_t-test
* https://en.wikipedia.org/wiki/Power_(statistics)

## Steps to add accuracy tests

* Estimate $\sigma$ from the full dataset.
* Decide a target minimum detectable effect $\theta$ based on the nature of dataset and corresponding accuracy metric.
* Decide $\alpha$ and $\beta$ based on the importance of model.
* Iterate sample volume $n$ from small to large, and compute $\theta$ until it satisfies (is equal to or lower than) the target $\theta$.
* Evaluate the model on the subset of sample volume $n$, resulting in the reference accuracy.
* The threshold $\gamma$ is automatically setup based on $\alpha$, $\sigma$, $n$ and the reference accuracy.
