# 时序相关性赋能稀疏注意力：面向 Blackwell 的启发式 Top-K 内核

NVIDIA TensorRT LLM 团队

## 目录

- [引言](#引言)
- [背景：DeepSeek 稀疏注意力中的 Indexer Top-K](#背景deepseek-稀疏注意力中的-indexer-top-k)
  - [Lightning Indexer 与 Top-K 选择](#lightning-indexer-与-top-k-选择)
  - [现有基于基数选择的 Top-K 实现](#现有基于基数选择的-top-k-实现)
  - [经典 Top-K 方法的复杂度对比](#经典-top-k-方法的复杂度对比)
- [长序列挑战](#长序列挑战)
  - [Agentic AI 与持续增长的上下文长度](#agentic-ai-与持续增长的上下文长度)
  - [长序列下 Indexer Top-K 为何成为瓶颈](#长序列下-indexer-top-k-为何成为瓶颈)
- [LLM 稀疏注意力中的时序相关性](#llm-稀疏注意力中的时序相关性)
  - [实验观测](#实验观测)
  - [理论基础：RoPE 频率结构](#理论基础rope-频率结构)
  - [预计算候选索引](#预计算候选索引)
- [启发式引导 Top-K 算法](#启发式引导-top-k-算法)
  - [核心思想](#核心思想)
  - [算法阶段](#算法阶段)
    - [阶段 1：预索引统计](#阶段-1预索引统计)
    - [阶段 2：割线法阈值搜索](#阶段-2割线法阈值搜索)
    - [阶段 3：无 Ballot 候选收集](#阶段-3无-ballot-候选收集)
    - [阶段 4：基于直方图的精确选择](#阶段-4基于直方图的精确选择)
  - [复杂度分析](#复杂度分析)
- [Blackwell 架构上的 GPU Kernel 实现](#blackwell-架构上的-gpu-kernel-实现)
  - [单 CTA 设计](#单-cta-设计)
  - [关键优化](#关键优化)
  - [共享内存布局](#共享内存布局)
  - [内存占用对比](#内存占用对比)
- [实验评估](#实验评估)
  - [实验设置](#实验设置)
  - [正确性验证](#正确性验证)
  - [单算子性能](#单算子性能)
    - [合成数据](#合成数据)
    - [真实解码数据](#真实解码数据)
  - [集成与启用方式](#集成与启用方式)
  - [端到端精度](#端到端精度)
  - [端到端吞吐量](#端到端吞吐量)
- [未来工作](#未来工作)
- [致谢](#致谢)

## 引言

Agentic AI 工作负载的兴起——LLM 自主浏览网页、规划任务、编写代码以及编排多步工具调用——推动上下文长度从数千 token 增长到数十万 token。在这些场景中，推理流水线中与序列长度线性相关的每一个组件都可能成为潜在瓶颈。[DeepSeek-V3.2](https://api-docs.deepseek.com/news/news251201) 引入的 DeepSeek 稀疏注意力 (DSA) 通过轻量级 **Lightning Indexer** 和 **Top-K 选择器** 仅保留最重要的键值条目，从而缓解注意力的二次复杂度。尽管稀疏 MLA 内核和 Indexer MQA 内核已经过深度优化（参见 [Tech Blog 15](blog15_Optimizing_DeepSeek_V32_on_NVIDIA_Blackwell_GPUs.md)），Top-K 选择本身——从数万乃至数十万个 indexer 分数中选取 top-2048 条目——随着序列增长在 DSA 模块延迟中占比越来越大。

本文介绍一种 **时序相关启发式引导 Top-K 算法**，该算法利用了 LLM 推理的一个基本特性：重要键值 token 的集合在连续解码步之间变化缓慢。通过将上一步的 Top-K 结果作为预测信号，算法仅需 1–2 次全局数据遍历即可估计出紧凑的阈值，随后利用无 ballot 共享内存技术收集和精炼候选集。在 NVIDIA Blackwell GPU 上运行真实 DeepSeek-V3.2 解码工作负载时，该数据感知方法相比生产环境的基数选择 (radix-select) 内核平均获得 **1.81×** 的单算子加速（单层单步最高达 **2.36×**），且不损失输出精度。

我们将详细阐述基于 RoPE 频率结构的理论基础、包含逐阶段复杂度分析的四阶段算法、与 `torch.topk` 对比的正确性验证、单算子和端到端基准测试以及在 TensorRT-LLM 中的集成路径。

## 背景：DeepSeek 稀疏注意力中的 Indexer Top-K

### Lightning Indexer 与 Top-K 选择

如 [Tech Blog 15](blog15_Optimizing_DeepSeek_V32_on_NVIDIA_Blackwell_GPUs.md) 所述，DSA 通过轻量级 MQA 机制计算索引分数：

$$I_{t} = \sum_{j=1}^{h}W_j^I \cdot \text{ReLU}(Q_{t, j}^I (K_t^I)^T)$$

索引分数张量 $I_t \in \mathbb{R}^N$（$N$ 为当前序列长度）量化了每个历史键值 token 对当前查询 token 的重要性。Top-K 操作随后选择分数最高的 $K = 2048$ 个位置，仅这些位置用于后续的稀疏 MLA 计算。

<div align="center">
<figure>
  <img src="../media/tech_blog19_indexer_topk.png" alt="Indexer Top-K 选择" width="900" height="auto">
</figure>
</div>
<p align="center"><sub><em>图 1. Indexer Top-K 选择示意图（以 K=5, N=10 为例）。Indexer 为每个历史 token 生成一个分数，Top-K 操作选取分数最高的 K 个位置并输出其索引。在 DeepSeek 稀疏注意力中，K=2048，N 的范围可从 8K 到 128K+。</em></sub></p>

Top-K 步骤对 DSA 流水线至关重要：它必须足够快（避免成为延迟瓶颈）且足够正确（保持模型精度）。随着 $N$ 从 8K 增长到 64K 或更长，挑战愈加严峻——Top-K 内核需要处理成比例增长的数据，而输出大小（$K = 2048$）保持不变。

### 现有基于基数选择的 Top-K 实现

TensorRT-LLM 生产环境的实现（`indexerTopK.cu` 中的 `invokeIndexerTopKDecode`）根据序列长度分发到不同的内核变体：

<div align="center">
<figure>
  <img src="../media/tech_blog19_topk_dispatch_flowchart.png" alt="Top-K 分发流程图" width="600" height="auto">
</figure>
</div>
<p align="center"><sub><em>图 1b. `invokeIndexerTopKDecode` 中原始的解码阶段 Top-K 分发逻辑（加入启发式 Top-K 之前）。内核针对短序列选择插入排序，中等序列选择基数排序，超长序列选择多 CTA 拆分方案。</em></sub></p>

核心算法（`topKPerRowJob`）使用与数据分布无关的 **基数选择 (radix-select)** 方法。该算法将 32 位浮点数表示拆分为数位组，迭代缩小候选集：

1. **直方图**：使用共享内存原子操作统计每个数位桶中的元素数量
2. **前缀和**：通过 `cub::BlockScan` 计算累积计数
3. **查找阈值**：确定第 $K$ 大元素所在的桶
4. **过滤**：保留阈值桶中的候选元素；直接输出更大桶中的元素

实现遵循 **half → 11 → 11 → 10** 位分解（4 次迭代），并在候选集降至 2048 以下时提前退出，切换为 CUB 基数排序或基于比较的排名。对于极长序列（N > 200K），拆分 CTA 路径将工作分配到 10 个 CTA 并合并结果。

### 经典 Top-K 方法的复杂度对比

<div align="center">

| 算法 | 时间复杂度 | 全局内存遍历次数 | GPU 适用性 |
|:------:|:-----------:|:-----------------:|:-----------:|
| `torch.topk`（基于排序） | $O(N \log N)$ | 多次 | 通用，常数较大 |
| 基数选择（TRT-LLM 生产版本） | $O(R \cdot N / P)$ | $R$ 次（$R \leq 4$） | 良好，分布无关 |
| 堆/优先队列 | $O(N \log K)$ | 1 次 | $K$ 较大时 GPU 并行性差 |
| **启发式引导（本工作）** | $O((I+1) \cdot N/P)$ | $I + 1$ 次（I ≈ 1–2） | 预测良好时最优 |

</div>

其中 $N$ 为序列长度，$K = 2048$，$P$ 为线程数，$R$ 或 $I$ 为迭代遍历次数。启发式方法的关键区别在于有效全局内存遍历次数 $I$ 取决于初始阈值估计的质量——如后文所示，在 LLM 解码工作负载中，估计质量始终很高。

## 长序列挑战

### Agentic AI 与持续增长的上下文长度

现代 Agentic AI 系统通常处理 32K–128K token 或更长的上下文：

- **多轮工具调用**：Agent 在数十轮交互中累积对话历史、工具输出和中间推理
- **长文档推理**：对大型代码库、法律文件或科学论文进行摘要、问答和分析
- **代码生成**：具有跨文件依赖关系的大型代码仓库上下文

这些工作负载对推理流水线中与 $N$ 成正比的每一个组件都构成压力。虽然稀疏 MLA 内核受益于 token 稀疏性（无论 $N$ 多大，只计算 $K = 2048$ 个 token 的注意力），indexer MQA 内核也已通过 FP8 算术和 Blackwell 专用指令进行了优化，但 Top-K 选择器仍需在每个解码步中扫描所有 $N$ 个 indexer 分数。

### 长序列下 Indexer Top-K 为何成为瓶颈

DSA 解码步的三大组件具有根本不同的扩展特性。由于三者在解码阶段均为访存受限，其 roofline 代价与总访存量成正比：

<div align="center">

| 组件 | 扩展特性 | 总访存量 | 随 $N$ 增长的趋势 |
|:------:|:---------:|:---------:|:-----------------:|
| **Indexer MQA** | O(N) | N × d_i × 2B | 线性增长 |
| **Top-K（radix-select）** | O(R·N) | R × N × 4B | 线性增长（R 次遍历） |
| **稀疏 MLA** | O(K) | K × d × 2B | **恒定**（K 固定） |

</div>

<sub><em>N：序列长度。K = 2048：固定 Top-K 数量。R ≈ 3：radix-select 对完整分数数组的遍历次数。d_i = 128：indexer 头维度。d = 192：MLA 头维度（128 non-PE + 64 PE）。</em></sub>

Top-K 的访存量 $R \cdot N \cdot 4\text{B}$ 随 $N$ 线性增长，而稀疏 MLA 保持恒定在 $K \cdot d \cdot 2\text{B}$。这意味着 **Top-K 在 DSA 延迟中的占比随序列长度单调递增**——从短序列时的次要组件逐步成为长序列下的主要瓶颈。生产环境的 radix-select 内核已比 `torch.topk` 快 7.4 倍（参见 [Tech Blog 15](blog15_Optimizing_DeepSeek_V32_on_NVIDIA_Blackwell_GPUs.md)），但除了 $O(R \cdot N)$ 的原始访存量之外，还有两个因素限制了其效率：

- **多遍历数据重复读取**：$R \approx 3$ 个 radix-select 步骤中，每步都对全部 $N$ 个元素执行两轮完整扫描（直方图构建 + 过滤/收集），每次内核调用总计约 6 次 $N$ 元素扫描。
- **共享内存原子操作串行化**：直方图阶段对 shared memory 桶计数器使用 `atomicAdd`（2048 个桶，热门桶存在竞争），收集阶段将所有合格元素通过单个 `atomicAdd(&counter, 1)` 写入——竞争同一地址的线程被串行化，使实际 SIMT 利用率远低于访存带宽理论上限。

这推动了对一种能够利用 LLM 解码特有属性来减少全局内存遍历次数的 Top-K 算法的探索。

## LLM 稀疏注意力中的时序相关性

### 实验观测

一个关键的实验观测是：Top-K 索引集在 LLM 解码过程中表现出**强时序相关性**——第 $t$ 步的重要 token 集合与第 $t-1$ 步的集合存在显著重叠。我们在 SWE-Bench-64K 评测的真实 DeepSeek-V3.2 解码阶段 indexer logits 上测量了该重叠率（命中率）：

<div align="center">
<figure>
  <img src="../media/tech_blog19_hit_ratio.png" alt="Top-K 命中率" width="900" height="auto">
</figure>
</div>
<p align="center"><sub><em>图 2. DeepSeek-V3.2 不同层连续解码步之间的 Top-K 重叠率（命中率），在 SWE-Bench-64K 上测量（2025 个解码步）。Layer 20–60 展现 35–50% 的平均重叠率（最高约 60%），而 Layer 0 和 1 的重叠率接近零（约 1–2%），反映了浅层与深层 indexer 分数的不同动态特性。</em></sub></p>

下图直观展示了这一现象：在连续解码步之间，大多数被选中的 Top-K token 保持不变（深蓝色），仅少部分发生变化（橙色）。其底层机制是注意力分数矩阵的 Toeplitz 结构——分数仅取决于相对位置 $\Delta = n - m$，因此前进一步仅导致分数景观的平滑偏移。

<div align="center">
<figure>
  <img src="../media/tech_blog19_temporal_correlation_diagram.png" alt="时序相关性与 Toeplitz 结构" width="800" height="auto">
</figure>
</div>
<p align="center"><sub><em>图 3. 上：具有 offset+1 偏移的时序相关性——当查询位置前进 1 时，所有相对位置 $\Delta$ 偏移 +1，约 60% 的 Top-K token 保持（深蓝色，右移一个单元格），约 40% 发生变化（橙色）。不同步对之间进入/离开的具体 token 不同。左下：注意力分数矩阵的 Toeplitz 结构。右下：归一化 $f(\Delta)$ 对比——标准 RoPE（蓝色虚线）峰值快速衰减，而 YaRN（深色实线）在大 $\Delta$ 处仍保持显著峰值，使 Top-K 能同时选择近距离和远距离位置的 token。</em></sub></p>

这种时序相关性并非偶然——它有基于 indexer 注意力机制结构的严格理论依据。

### 理论基础：Toeplitz 结构与 RoPE 频率分析

DSA 的 lightning indexer 通过 [RoPE](https://arxiv.org/abs/2104.09864) 编码的查询和键张量之间的 MQA 点积计算 token 重要性分数。在 RoPE 中，位置 $(n, m)$ 处的每对查询-键乘以一个分块对角旋转矩阵 $R_{n-m}$，其元素为位置相关角度的余弦和正弦。对于 indexer 的 RoPE 维度（DeepSeek-V3.2 中 $d_{\text{rope}} = 64$），位置编码对注意力分数的贡献简化为：

$$f(\Delta) = 2 \sum_{i=0}^{d_{\text{rope}}/2 - 1} \cos(\Delta \cdot \theta_i), \quad \Delta = n - m, \quad \theta_i = \beta^{-2i/d_{\text{rope}}}$$

其中 $\beta = 10000$ 为 RoPE 基频率。该表达式是全 1 向量经 $R_\Delta$ 变换后的内积，代表了独立于数据内容的纯位置编码贡献。

**Toeplitz 性质。** 由于 $f$ 仅依赖相对位置 $\Delta = n - m$，位置分数矩阵 $P \in \mathbb{R}^{S \times S}$ 是一个 [Toeplitz 矩阵](https://zh.wikipedia.org/wiki/%E6%89%98%E6%99%AE%E5%88%A9%E8%8C%A8%E7%9F%A9%E9%98%B5)——沿每条对角线恒定。这将理解哪些 token 对在位置上被优先选择的问题从二维矩阵分析简化为对 $f(\Delta)$ 的**一维函数分析**。

**多尺度余弦叠加。** $f(\Delta)$ 是 $d_{\text{rope}}/2 = 32$ 个余弦的叠加，周期跨越 $2\pi/\theta_0 \approx 6.3$ 到 $2\pi/\theta_{31} \approx 58{,}600$——频率比达 10,000:1。全局最大值在 $\Delta = 0$（自身位置）；次级峰出现在余弦波建设性干涉处。由于 $f$ 是平滑的，查询前进一步（$\Delta \to \Delta + 1$）仅对分数景观产生微小扰动——这直接解释了为什么 Top-K 索引在连续解码步之间变化缓慢。

**YaRN 扩展。** DeepSeek-V3.2 通过 [YaRN](https://arxiv.org/abs/2309.00071)（缩放因子 $s = 40$）扩展 RoPE，它对低频分量进行插值而保留高频分量。对 $f(\Delta)$ 的效果是**大相对位置处的峰值仍保持显著**而非单调衰减（参见图 3 右下）。这产生了更加空间均匀的重要 token 分布——有利于启发式 Top-K 方法，因为 Top-K 集合同时涵盖近距离和远距离位置，为下一步提供更丰富的预测信号。

### 预计算候选索引

RoPE 的频率结构支持静态预计算。给定模型的 RoPE 参数（维度 $d = 64$，YaRN 缩放因子 `scaling_factor = 40`），我们可以对所有可能的相对位置计算理想化的分数函数 $f(\Delta)$，并找到分数最大的 $K$ 个位置：

$$\mathcal{P}_{\text{static}} = \text{argtopk}_{\Delta \geq 0} \; f(\Delta)$$

早期方案曾考虑使用 $f(\Delta)$ 的**峰值索引**（$f'(\Delta) = 0$ 且 $f''(\Delta) < 0$ 的位置）作为候选。然而分析表明，峰值索引仅实现约 17–35% 的预测成功率——重要峰值占总 token 的比例很低（约 13%），且许多真实 Top-K 索引落在峰值之间。**基于 TopK 的预测**（直接使用 $f(\Delta)$ 的 Top-K）实现了 45–100% 的成功率，提升数倍。

该预计算索引集捕获了**结构先验**——RoPE 频率结构天然偏好的位置。在推理过程中，位置 $n$ 处查询的实际 Top-K 索引是满足 $n - m \in \mathcal{P}_{\text{static}}$ 的位置 $m$，并受实际 Q/K 张量内容和 indexer 权重 $W^I$ 的数据相关性调制。

实践中，我们使用**上一步的 Top-K 结果**作为预测信号 (`preIdx`)，它同时捕获了结构化的 RoPE 先验和数据相关的内容关联。对于大多数层（L20–L60），这实现了预测准确率 α ≈ 0.35–0.50（上一步 Top-K 索引中 35–50% 仍在当前 Top-K 集合中），足以使启发式算法在 1–2 次插值迭代内收敛。值得注意的是，第 0 和第 1 层几乎无时序相关性（α ≈ 0.01），导致启发式内核需要更多插值迭代——这与基准测试中这两层较低的加速比一致。

## 启发式引导 Top-K 算法

### 核心思想

给定输入向量 $\mathbf{x} = (x_0, x_1, \ldots, x_{N-1}) \in \mathbb{R}^N$ 和预测索引集 $\mathcal{P} = \{p_0, p_1, \ldots, p_{M-1}\} \subset \{0, \ldots, N-1\}$（$M = 2048$），寻找包含 $\mathbf{x}$ 中 $K$ 个最大值索引的集合 $\mathcal{S}^*$，$|\mathcal{S}^*| = K$。

**核心定理**：若阈值 $T$ 满足 $K \leq |\{i : x_i \geq T\}| \leq C$（其中 $C$ = MAX\_CANDIDATES = 6144），则候选集 $\mathcal{C} = \{i : x_i \geq T\}$ 包含所有 Top-K 元素，即 $\mathcal{S}^* \subseteq \mathcal{C}$。

该算法利用预测索引 $\mathcal{P}$ 估计一个接近真实第 $K$ 大值 $x_{(K)}$ 的阈值 $T$。在估计良好的情况下，只需 1–2 次全局数据遍历（而非基数选择的 3–4 次），随后在共享内存中对小候选集进行精炼。

### 算法阶段

<div align="center">
<figure>
  <img src="../media/tech_blog19_algorithm_flow.png" alt="启发式引导 Top-K 算法流程" width="800" height="auto">
</figure>
</div>
<p align="center"><sub><em>图 4. 启发式引导 Top-K 算法流程。四个阶段在单个 CTA 内顺序执行。复杂度标注显示各阶段代价；I 和 S 分别为阈值搜索和 snap 迭代次数。</em></sub></p>

#### 阶段 1：预索引统计

利用预测索引集 $\mathcal{P}$ 计算对应输入值的最小值、最大值和均值：

$$T_0 = \bar{x}_{\mathcal{P}} = \frac{1}{|\mathcal{P}|} \sum_{i \in \mathcal{P}} x_i$$

当预测准确率 $\alpha \approx 0.5$ 时，$T_0$ 近似为：

$$T_0 \approx \alpha \cdot \mathbb{E}[x_i \mid i \in \mathcal{S}^*] + (1 - \alpha) \cdot \mathbb{E}[x_i]$$

这比无条件均值更接近 $x_{(K)}$，从而加速阶段 2 的收敛。该阶段使用散列 `__ldg` 读取和 `redux.sync` warp 归约（sm\_80+ 上单指令替代 5 次 shuffle 操作）。

**代价**：$O(M/P)$ — 用 $P = 512$ 个线程读取 $M = 2048$ 个散列全局内存值。

#### 阶段 2：割线法阈值搜索

<div align="center">
<figure>
  <img src="../media/tech_blog19_secant_method.png" alt="割线法阈值搜索" width="750" height="auto">
</figure>
</div>
<p align="center"><sub><em>图 5. 阶段 2 插值阈值搜索。从 T₀ = pmean、搜索区间 [pmin, pmax] 开始，算法计算 f(T₀)：因 f(T₀) > C，令 val_lo = T₀。割线 ① 连接 (val_lo, cnt_lo) 与 (val_hi, cnt_hi)，与 f_target 的交点确定 T₁。因 f(T₁) < K，令 val_hi = T₁，区间收窄。割线 ② 连接更新后的锚点，与 f_target 的交点确定 T₂，T₂ 落入目标区域 [K, C]——收敛完成。首次迭代施加阻尼（f ≤ 0.50）以防过冲。注：实际 f(T) = count(input ≥ T) 为单调不增的阶梯函数，图中光滑曲线仅为示意。</em></sub></p>

定义计数函数 $f(T) = |\{i : x_i \geq T\}|$，为关于 $T$ 的单调非递增阶梯函数。目标是找到 $T^*$ 使得 $K \leq f(T^*) \leq C$。

每次迭代通过 `blockCountGE` 计算精确的全局计数——这是一个带宽受限的循环，使用 `float4` 向量化 `__ldg` 加载、纯寄存器比较与累加以及 warp 级归约。关键的是，`blockCountGE` 将每个线程的局部计数缓存到 `smem->per_thread_counts[tid]`（在 warp 归约之前），该缓存被阶段 3 复用以消除冗余的 $N$ 次扫描（详见下文）。

阈值更新采用割线法插值：

$$T_{\text{new}} = T_{\text{lo}} + \frac{f(T_{\text{lo}}) - f_{\text{target}}}{f(T_{\text{lo}}) - f(T_{\text{hi}})} \cdot (T_{\text{hi}} - T_{\text{lo}})$$

首次迭代施加阻尼（$\leq 0.5$）以防过冲，并在浮点精度极限时回退到二分搜索。对于平滑 CDF，割线法实现超线性收敛（阶 $\approx 1.618$）。

当阶段 2 正常收敛（`done=1`）时，**安全网守卫**会跳过后续原本冗余的验证 `blockCountGE` 调用，每次内核调用节省 $\sim$4 µs。

**代价**：$O(I \cdot N/P)$，$I$ 为迭代次数。真实解码数据（预测良好）：I ≈ 1–2；合成数据（预测较差）：I ≈ 4–6。

#### 阶段 3：无 Ballot 候选收集

找到有效阈值后，所有 $\geq T$ 的元素被收集到共享内存。该阶段采用**无 ballot** 设计，并有一个关键优化：**阶段 3 子遍历 1（计数）被消除**，转而复用阶段 2 最后一次 `blockCountGE` 调用缓存的逐线程计数。由于阈值在阶段 2 的最终计数和阶段 3 之间未改变，缓存的 `smem->per_thread_counts[tid]` 值直接有效——节省了一次完整的 $N$ 元素重新扫描（$\sim$4 µs）。

- **写入偏移计算**：从缓存读取逐线程计数 → warp 前缀和 → 跨 warp 前缀和 → 预计算写入偏移。
- **子遍历 2（写入）**：每个线程写入其预分配的共享内存范围。无 `__ballot_sync`，无 `__shfl_sync`，无 `atomicAdd`。

无 ballot 设计至关重要：`__ballot_sync` 充当编译器屏障，使 L2 加载流水线串行化。原始的逐元素 ballot 方案耗时 ~30,000 周期（`blockCountGE` 的 5 倍）；无 ballot 方案耗时 ~16,000 周期。

**代价**：$O(N/P)$ — 一次扫描完整输入的子遍历（计数子遍历已通过缓存消除）。

#### 阶段 4：基于直方图的精确选择

<div align="center">
<figure>
  <img src="../media/tech_blog19_phase4_detail.png" alt="阶段 4 直方图详情" width="800" height="auto">
</figure>
</div>
<p align="center"><sub><em>图 6. 阶段 4 详情：4a 扫描候选集获取 min/max 范围；4b 构建 2048 桶直方图并进行 warp 并行第 K 桶搜索（16 个 warp × 128 桶，160 步串行扫描）；4d 通过 snap 迭代精炼，收敛条件为 n>(T) < K ≤ n≥(T)；4e 分区输出最终结果。</em></sub></p>

若候选数不恰好等于 $K$，则在共享内存中精炼出恰好 $K$ 个元素：

1. **最小/最大值扫描**：在候选集上扫描以获取精确的直方图区间
2. **2048 桶直方图**：通过 `atomicAdd` 对候选集进行单遍直方图统计，随后采用 **warp 并行第 K 桶搜索**：每个 warp 累加其 `NUM_BINS / NUM_WARPS` 个桶（从高到低），然后 `tid=0` 扫描 `NUM_WARPS` 个 warp 总计以定位目标 warp，最后目标 warp 的 `lane=0` 定位精确的桶。这将串行扫描步数从 2048 减少到 2 × NUM_WARPS + NUM_BINS / NUM_WARPS = 160（减少 12.8 倍）。
3. **Snap 迭代**：通过逐步跨越不同数据值将阈值精炼至精确的第 $K$ 大值。每次融合 snap 迭代在一次共享内存扫描中计算 `(count_ge, count_gt, snap_up, snap_down)`。收敛条件：$n_{>}(T) < K \leq n_{\geq}(T)$。使用 2048 桶时，每次内核调用仅需 **1–3** 次 snap 迭代。
4. **分区**：无条件输出 $> T^*$ 的元素，用 $= T^*$ 的元素填充剩余槽位。

**代价**：$O(S \cdot C/P)$，S ≈ 1–3 次 snap 迭代，$C \leq 6144$ 个候选。这是纯共享内存操作，无需全局内存访问。

### 复杂度分析

<div align="center">

| 阶段 | 时间复杂度 | 内存访问 | 空间 |
|:------:|:-----------:|:---------:|:------:|
| 阶段 1（预索引统计） | O(M/P) | M 次散列全局读取 | O(1) 寄存器 |
| 阶段 2（阈值搜索） | O(I·N/P) | I×N 次顺序读取（L2） | O(1) 寄存器 |
| 阶段 3（候选收集） | O(N/P) | N 次顺序读取（L2） | O(C) 共享内存 |
| 阶段 4（精确选择） | O(S·C/P) | 仅共享内存 | O(B) 共享内存 |
| **总计** | **O((I+1)·N/P + S·C/P)** | (I+1)×N + M 全局 | ~60 KB 共享 |

</div>

对于真实解码数据（I ≈ 2, S ≈ 2, P = 512, B = 2048, C ≤ 6144）：约 3N/P + 2C/P 次内存访问。阶段 3 的计数缓存优化消除了一次完整的 N 次扫描（计数子遍历），将总全局内存遍历次数从 (I+2) 降低为 (I+1)。相比 radix-select 基线需要 R·N/P 次访问（R ≈ 3–4 次遍历，每次包含直方图 + 前缀和 + 过滤），启发式方法的全局内存访问次数更少，且共享内存同步开销显著减少。

## Blackwell 架构上的 GPU Kernel 实现

### 单 CTA 设计

启发式 Top-K 内核实现为 **单 CTA（协作线程阵列）** 内核，每个 block 512 个线程。该设计带来以下优势：

- 所有阶段间通信通过共享内存完成（无需全局同步）
- 高效复用跨阶段的 L2 缓存（输入数据保持热数据状态）
- 作为现有逐行 Top-K 内核的直接替换，集成简便
- 兼容 CUDA Graph（每个 batch 固定网格维度）

每个 CTA 处理 batch 中的一行（一个查询 token 的 indexer 分数）。多行内核 `heuristicTopKMultiRowKernel` 是一个轻量封装，计算每行参数（序列长度、指针偏移）并委托给 `heuristicTopKJob` 设备函数。

### 关键优化

内核集成了多项针对系统化 profiling 发现的特定瓶颈的优化：

<div align="center">

| 优化 | 描述 |
|:------:|:------:|
| `__ldg` + `redux.sync` | 只读纹理缓存加载 + 单指令 warp 归约（sm\_80+） |
| 无 Ballot 阶段 3 | 消除 `__ballot_sync` 导致的 L2 加载流水线串行化编译器屏障 |
| 安全网守卫 | 阶段 2 正常收敛时跳过冗余的 `blockCountGE` |
| 2048 桶直方图 + warp 并行扫描 | warp 并行第 K 桶搜索减少串行扫描步数；更少的 snap 迭代 |
| 阶段 3 计数缓存 | 缓存阶段 2 `blockCountGE` 的逐线程计数；消除阶段 3 冗余的 $N$ 次扫描 |

</div>

### 共享内存布局

内核使用约 60 KB 共享内存：

```text
KernelSmem {
    float  keys[MAX_CANDIDATES];       // 6144 × 4B = 24,576 B
    int    vals[MAX_CANDIDATES];       // 6144 × 4B = 24,576 B
    int    warp_counts[NUM_WARPS];     //   16 × 4B =     64 B
    int    histogram[NUM_BINS];        // 2048 × 4B =  8,192 B
    int    per_thread_counts[BLOCK_SIZE]; // 512 × 4B = 2,048 B  ← P3 计数缓存
    // + 标量临时变量                     //            ~    40 B
}                                      // 总计: ~60 KB
```

在 Blackwell（sm\_100）上，内核通过 `cudaFuncSetAttribute` 启用扩展共享内存（>48 KB）。`MAX_CANDIDATES = 6144` 的缓冲区大小提供了 $K = 2048$ 的 3 倍裕量，确保安全网阈值始终能找到有效范围而不溢出。`per_thread_counts` 数组支持阶段 3 计数缓存优化（消除冗余的 $N$ 次扫描），2048 桶直方图以额外共享内存为代价减少了 snap 迭代次数。

### 内存占用对比

<div align="center">

| 指标 | 基数排序 | 启发式引导 |
|:------:|:--------:|:---------:|
| **每 CTA SMEM** | ~28 KB（基于 union 的分时复用） | ~60 KB（平展结构体，候选持久驻留） |
| **扩展 SMEM opt-in** | 否 | 是（>48 KB） |
| **额外 HBM** | 0 | scratch + prev\_topk（见下文） |

</div>

启发式内核使用约 2 倍于 radix-select 的 SMEM，因为阶段 3 收集的候选必须在阶段 4 中持续可访问（无法通过 union 复用）。在 Blackwell（sm\_100，每 SM 228 KB 共享内存）上仍可支持每 SM 3 个 CTA——对典型 batch 大小已有足够的占用率。

启发式路径需要两块额外的持久预分配 HBM 缓冲区（兼容 CUDA Graph）：

- **`heuristic_scratch_values`**（$B \times K \times 4\text{B}$）：一个只写的 dummy 缓冲区，用于内核的 output-values 路径。DSA 流水线仅消费输出 indices，但内核无条件地同时写出 values 以保持最优 SASS 代码生成质量。该缓冲区写入后不被下游读取。
- **`heuristic_prev_topk`**（$L \times B \times K \times 4\text{B}$，$L$ 为本地 DSA 层数）：存储每层上一个解码步的 Top-K indices 作为下一步的 preIdx。需要独立缓冲区是因为（1）`topk_indices_buffer` 每步原地覆盖（读写冲突），（2）每层需要独立的时序状态，（3）CUDA Graph replay 要求稳定地址的反馈缓冲区。

## 实验评估

### 实验设置

**平台**：NVIDIA B200 GPU（Blackwell，sm\_100）。所有微内核基准测试均为单 batch、单行（单 CTA），启发式内核和生产 radix-select 基线均使用 **512 线程/block**，确保在相同线程级资源下公平对比。

**输入数据**分为两类：

<div align="center">

| 类别 | 来源 | preIdx | 时序相关性 |
|:------:|:------:|:--------:|:-----------:|
| **合成随机** | 随机 Q/K（1 + Aₘ · 𝒩(0,1)）+ YaRN-RoPE → 单头 qᵀR_Δk | 静态 RoPE 先验（全 1 Q/K → f(Δ) Top-K） | 中等（~60%+ 重叠，取决于 Aₘ） |
| **真实解码** | DeepSeek-V3.2 SWE-Bench-64K indexer logits（多头 MQA，含 Wᴵ、ReLU） | 上一步的实际 Top-K 输出 | 高（~35–50% 重叠） |

</div>

合成流水线计算仅在 RoPE 维度（$d_{\text{rope}} = 64$）上的简化单头点积，而真实 indexer 使用跨 64 个头的多头加权求和 $I_t = \sum_j W_j^I \cdot \text{ReLU}(Q_{t,j}^I (K_t^I)^T)$。合成数据捕获位置结构但缺乏真实解码中的内容相关性，导致 preIdx 质量较低、Phase 2 需要更多迭代。

<details>
<summary><b>合成分数生成代码</b></summary>

```python
import math, torch, numpy as np

def yarn_inv_freq(dim=64, base=10000.0, sf=40.0, orig_max=4096, bf=32, bs=1):
    """DeepSeek-V3.2 YaRN 频率计算。"""
    pos_f = base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
    freq_extra, freq_inter = 1.0 / pos_f, 1.0 / (sf * pos_f)
    lo = max(int(dim * math.log(orig_max/(bf*2*math.pi)) / (2*math.log(base))), 0)
    hi = min(int(math.ceil(dim*math.log(orig_max/(bs*2*math.pi)) / (2*math.log(base)))), dim-1)
    ramp = torch.clamp((torch.arange(dim//2).float() - lo) / max(hi-lo, 1e-3), 0, 1)
    return freq_inter * ramp + freq_extra * (1 - ramp)

def compute_static_pre_idx(N, K=2048, d_rope=64):
    """从全 1 Q/K 的 RoPE 结构先验计算 preIdx：f(Δ) = 2·Σcos(Δ·θᵢ)。"""
    theta = yarn_inv_freq(d_rope).numpy()
    f = 2 * np.cos(np.outer(np.arange(N), theta)).sum(axis=1)
    return torch.from_numpy(f).topk(K).indices.int()

def generate_indexer_scores(N, K=2048, Am=0.1, d_rope=64, device="cuda"):
    """生成合成分数（随机 Q/K + YaRN-RoPE）和静态 preIdx。"""
    inv_freq = yarn_inv_freq(d_rope).to(device)
    pos = torch.arange(N, device=device).float()
    cos_t = torch.cos(pos[:, None] * inv_freq[None, :])
    sin_t = torch.sin(pos[:, None] * inv_freq[None, :])
    def rope(x, c, s):
        x1, x2 = x[..., ::2], x[..., 1::2]
        return torch.cat([x1*c - x2*s, x2*c + x1*s], dim=-1)
    q = 1.0 + Am * torch.randn(1, d_rope, device=device)
    k = 1.0 + Am * torch.randn(N, d_rope, device=device)
    scores = (rope(q, cos_t[:1], sin_t[:1]) @ rope(k, cos_t, sin_t).T).squeeze(0)
    pre_idx = compute_static_pre_idx(N, K, d_rope).to(device)
    return scores, pre_idx
```

</details>

**基准测试方法**：所有内核耗时通过 `nsys` profiling 采集，输入数据为冷启动状态（每次内核调用前刷新 L2 缓存以消除缓存预热干扰）。

### 正确性验证

启发式 Top-K 内核在所有测试序列长度（$N$ = 8K–131K）下产生与 `torch.topk` **逐比特一致**的 Top-K 索引集。对于相等值的非确定性排序（与生产内核行为一致），如 [Tech Blog 15](blog15_Optimizing_DeepSeek_V32_on_NVIDIA_Blackwell_GPUs.md) 所验证，对精度影响可忽略。

### 单算子性能

#### 合成数据

<div align="center">
<figure>
  <img src="../media/tech_blog19_synthetic_scaling.png" alt="合成数据扩展性" width="750" height="auto">
</figure>
</div>
<p align="center"><sub><em>图 7. 合成数据下的内核延迟随序列长度变化（B200）。两种内核均为 O(N) 扩展；启发式内核的拟合斜率约为基线的 42%，反映了更少的全局内存遍历次数。加速比标注在代表性 N 值处。</em></sub></p>

<div align="center">

| $N$ | 启发式 (ns) | 生产基线 (ns) | 加速比 |
|:-----:|:------------:|:-------------:|:--------:|
| 8,192 | 16,512 | 11,200 | 0.68× |
| 16,384 | 21,856 | 21,984 | **1.01×** |
| 32,768 | 26,112 | 32,928 | **1.26×** |
| 65,536 | 31,904 | 47,936 | **1.50×** |
| 70,690 | 36,864 | 51,200 | **1.39×** |
| 131,072 | 43,392 | 76,128 | **1.75×** |

</div>

<sub><em>表 1. B200，norm/gamma/beta 分布合成输入。"生产基线"为 `topKPerRowDecode` 内核（$N < 12{,}288$ 时使用插入排序，$N \geq 12{,}288$ 时使用基数排序）。"加速比" = 基线耗时 / 启发式耗时。</em></sub>

短序列（N = 8,192）时，阶段 1（散列 preIdx 读取）和阶段 2（插值迭代）的开销超过节省的时间，启发式内核约慢 32%。启发式内核在 N = 16,384 左右达到平衡，随着序列长度增长逐步超越基线——在 N = 131,072 时达到 **1.75×**。这一扩展优势源于启发式内核的全局内存遍历次数（$I + 1 \approx 3\text{-}4$）增长缓慢，而基数选择方法的多遍历 直方图 + 前缀和 + 过滤 流水线在大 $N$ 时单遍开销更高。

**关键洞察**：合成基准测试中使用的静态 RoPE 结构先验作为 preIdx 与真实 Top-K 之间具有较高的重叠率，使 Phase 2 能以较少的迭代次数收敛。结合高效的单 CTA 设计和无 ballot 收集方式，在较长序列长度下提供了一致的扩展优势。

#### 真实解码数据

在 SWE-Bench-64K 推理中捕获的真实 DeepSeek-V3.2 解码阶段 indexer logits 上评估（prompt 长度 68,665，decode 长度 2,025）。从 2,024 个解码步（跳过第 0 步，因无 preIdx 可用）中每隔 128 步采样（从第 1 步开始，共 16 个等间距采样点），再加上最后一个解码步（第 2,024 步，$N = 70{,}690$），共 17 个采样点在 9 个代表性层上进行 profiling。每个采样点的 preIdx 为前一解码步的 Top-K 输出，反映真实的时序相关性。

<div align="center">
<figure>
  <img src="../media/tech_blog19_real_data_bars.png" alt="真实数据逐层延迟" width="800" height="auto">
</figure>
</div>
<p align="center"><sub><em>图 8. N = 70,690（最后一个解码步）时的逐层内核延迟（B200）。启发式内核在所有 9 层上实现 1.32×–2.11× 的加速比。L21 受益最大（2.11×），因其高度一致的 beta 分布；L0 受益最小（1.32×），因其异质性的 lognormal 分布。</em></sub></p>

**所有 17 个采样解码步（$N = 68{,}667$–$70{,}690$）的平均加速比：**

<div align="center">

| 层 | 平均加速比 | 最小加速比 | 最大加速比 |
|:----:|:-----------:|:-----------:|:----------:|
| 0 | **1.48×** | 1.17× | 1.83× |
| 1 | **1.72×** | 1.57× | 1.89× |
| 20 | **1.76×** | 1.32× | 2.21× |
| 21 | **1.99×** | 1.49× | 2.28× |
| 22 | **1.92×** | 1.67× | 2.30× |
| 40 | **1.80×** | 1.26× | 2.20× |
| 41 | **1.79×** | 1.15× | 2.24× |
| 42 | **2.00×** | 1.33× | 2.36× |
| 60 | **1.86×** | 1.33× | 2.14× |
| **总平均** | **1.81×** | — | — |

</div>

<sub><em>表 3. 17 个采样解码步（步长 128，共 2,024 步）的逐层平均、最小和最大加速比。启发式内核相对生产 radix-select 基线实现总平均 1.81× 加速。</em></sub>

结果表明启发式内核在所有层和所有解码步上**一致优于**生产 radix-select 基线。加速比非常稳定：即使最差层（Layer 0，平均 1.48×）仍有显著改善，最佳层（21、42）平均加速比达到 2.0×。

在真实解码数据上，`pmean` 由于 preIdx 的高时序相关性（连续步之间约 50% 重叠）能够准确逼近真实阈值，使阶段 2 在 1–2 次迭代内收敛。不同层的差异（如 Layer 0 加速比低于 Layer 21）反映了不同注意力层分数分布特性的差异——某些层具有更尖锐的分数分布，收敛更快，而其他层则具有更平坦的分布，需要额外的插值步骤。


#### 数据分布分析

理解每层的分数分布对于解释性能差异至关重要。我们对 SWE-Bench-64K 解码 logits（9 层 × 2,024 解码步 × ~70,690 元素）进行了统计拟合，以表征每层的分布特性：

<div align="center">

| 层 | 最佳拟合分布 | 占比 | 次优分布 | 占比 | 平均 Logit | 峰度 |
|:----:|:-------------:|:------:|:---------:|:------:|:-----------:|:------:|
| L0 | **lognorm** | 42.9% | beta | 26.0% | −4.12 | −0.128 |
| L1 | **logistic** | 59.4% | t | 40.6% | −0.47 | +0.931 |
| L20 | **beta** | 90.1% | weibull\_min | 9.9% | −0.65 | −0.282 |
| L21 | **beta** | 99.7% | weibull\_min | 0.2% | −0.87 | −0.337 |
| L22 | **weibull\_min** | 63.9% | beta | 36.1% | −3.04 | −0.111 |
| L40 | **beta** | 86.1% | lognorm | 5.0% | −3.16 | −0.162 |
| L41 | **beta** | 79.5% | lognorm | 10.3% | −2.76 | −0.138 |
| L42 | **beta** | 63.7% | weibull\_min | 36.4% | −4.51 | −0.621 |
| L60 | **weibull\_min** | 93.6% | beta | 6.3% | −2.26 | −0.396 |

</div>

对阈值搜索性能的关键观察：
- **Beta 分布主导** L20/21/40/41/42 — 有界形状分布，`pmean` 能准确逼近 Top-K 阈值，使阶段 2 快速收敛（1–2 次迭代）。
- **L1 具有尖峰特征**（峰度 +0.931），厚尾分布（logistic/t 分裂）——阈值估计更困难，需要更多阶段 2 迭代。
- **L0 最为异质**（三方分裂：lognorm/beta/weibull），值域最宽（17.28）——解释了其在各基准测试中一致较低的加速比。
- 所有层的平均 logit 值均为负值（−0.47 至 −4.51），与注意力后 indexer logit 空间一致。

#### 启发式算法的分布敏感性

启发式 Top-K 算法的加速比本质上对输入分数分布敏感，因为分布决定了两个关键因素：（1）`pmean` 与真实第 K 大值的接近程度（阶段 2 初始猜测质量），以及（2）阈值附近的候选密度（阶段 4 snap 迭代次数）。

综合合成数据和真实数据基准测试，呈现出清晰的模式：

<div align="center">

| 分布类型 | 代表层 | preIdx 重叠率 | 阶段 2 迭代 | 观测加速比 |
|:---------:|:--------:|:-------------:|:-----------:|:----------:|
| **Beta**（有界，尖峰） | L21, L40, L41 | 高 | 1–2 | **1.80–2.11×** |
| **Weibull**（右偏） | L22, L60 | 中–高 | 2–3 | **1.72–1.92×** |
| **Logistic/t**（厚尾） | L1 | 中 | 2–3 | **1.74×** |
| **Lognorm**（异质） | L0 | 较低 | 3–4 | **1.32×** |
| **合成数据**（gamma/beta 类，静态 preIdx） | N=70K | 中 | 2–4 | **1.39×** |

</div>

模式一致：**有界尖峰分布**（beta）产生最佳加速比，因为 `pmean` 接近 Top-K 阈值且插值过程中候选数量快速下降。**厚尾分布**（logistic、lognorm）将分数质量分散到更宽范围，导致 `pmean` 的初始估计精度降低——需要更多插值迭代，阈值附近的候选集也可能更密集（更多 snap 迭代）。

值得注意的是，即使最差的真实数据层（L0，lognorm，1.32×）和使用静态 preIdx 的合成基准测试（1.39×）也始终优于基线。这展示了该算法**对多样化分数分布的强鲁棒性**——无论底层分布是有界的（beta）、厚尾的（logistic）、偏斜的（weibull）还是异质的（lognorm），均能提供稳定的性能加速。

### 集成与启用方式

启发式 Top-K 通过两级分发集成到 TensorRT-LLM DSA 流水线中：

<div align="center">
<figure>
  <img src="../media/tech_blog19_dispatch_logic.png" alt="分发逻辑" width="800" height="auto">
</figure>
</div>
<p align="center"><sub><em>图 9. 完整的解码阶段 Top-K 分发。Python 层在 CuTE DSL Top-K（小 batch）和 C++ 内核之间选择。在 C++ 内核内部，当 preIdx 和 scratch 缓冲区可用时启发式路径具有最高优先级；否则通过 radix-select 回退链处理请求。</em></sub></p>

**Python 层**（`dsa.py`）：若 CuTE DSL Top-K 后端已启用且 batch 大小较小（$\leq 256$ token），则直接使用。否则分发到 C++ `indexer_topk_decode` 算子：

```python
if self.use_cute_dsl_topk and num_gen_tokens <= 256:
    torch.ops.trtllm.cute_dsl_indexer_topk_decode(logits, kv_lens, indices, topk, next_n)
else:
    torch.ops.trtllm.indexer_topk_decode(logits, kv_lens, indices, next_n, topk,
                                          pre_idx=pre_idx,               # 上一步的 Top-K
                                          heuristic_scratch=heuristic_scratch)  # scratch 缓冲区
```

**C++ 层**（`indexerTopK.cu`）：`canUseHeuristic` 门控在选择启发式快速路径前检查所有条件：

```cpp
bool const canUseHeuristic = preIdx != nullptr       // 上一步 Top-K 可用
    && stride1 == 1                                  // 内存连续布局
    && topK == kHeuristicTopK                        // K = 2048
    && preIdxCount == kHeuristicSize                 // M = 2048
    && preIdxStride >= preIdxCount                   // stride 有效
    && numColumns < effectiveSplitWorkThreshold      // N < 200K（单 CTA）
    && heuristicScratch != nullptr;                  // scratch 缓冲区已分配
```

当任一条件不满足时（如预填充阶段、N ≥ 200K、首 token 尚无 preIdx），分发逻辑回退到原始 radix-select 流水线，确保不产生性能退化。

**启用方式。** 启发式 Top-K 由 `DeepSeekSparseAttentionConfig` 中的 `enable_heuristic_topk` 字段控制（默认：`false`）。可通过传递给 `trtllm-serve`、`trtllm-bench` 或 `trtllm-eval` 的 YAML 配置文件启用：

```yaml
# config.yml（或 extra_llm_api_options.yaml）
sparse_attention_config:
    algorithm: dsa
    enable_heuristic_topk: true    # 设为 false 使用默认 radix-select
```

该 YAML 可通过 `--config` 或 `--extra_llm_api_options` 标志传递给 `trtllm-serve`、`trtllm-bench` 或 `trtllm-eval`。启发式路径仅在 sm\_100+（Blackwell）GPU 上激活；在更早的架构上该标志被静默忽略。

### 端到端精度

使用 `trtllm-eval` 在五个基准测试上验证 DeepSeek-V3.2 NVFP4 模型的端到端精度（B200 ×8，EP8+DP8，MTP-1）。每个基准测试独立运行多次以评估运行间方差。

<div align="center">

| 基准测试 | 样本数 | 基线（平均） | 启发式（平均） | 差值 | 实验次数（B / H） |
|:---------:|:--------:|:------------:|:-------------:|:------:|:-----------------:|
| **MMLU** | 14,042 | 87.51 | 87.50 | **−0.01** | 1 / 4 |
| **GSM8K** | 1,319 | 95.11 | 95.23 | **+0.12** | 1 / 4 |
| **GPQA-Diamond** | 198 | 77.27 | 77.15 | **−0.12** | 1 / 4 |
| **LongBench V1** | ~5,000 | 44.61 | 44.28 | **−0.33** | 8 / 8 |
| **LongBench V2** | 215 | 49.58 | 49.12 | **−0.46** | 5 / 5 |

</div>

<sub><em>表 4. DeepSeek-V3.2 NVFP4 端到端精度总结。"实验次数（B / H）"= 基线/启发式的独立运行次数。所有差值均在运行间方差范围内，统计上不显著。</em></sub>

**各基准测试详情：**

- **MMLU**（14,042 题，单 token 输出）：启发式结果在 87.52/87.51/87.49/87.49 间确定性波动——与基线 87.51 几乎一致。短输出基准测试中 DSA indexer Top-K 几乎不被触发，因为解码序列极短。
- **GSM8K**（1,319 道数学题）：启发式平均 95.23（范围 95.03–95.49）vs 基线 95.11——与 Blog 15 中 NVFP4 检查点的参考值 95.26 一致。
- **GPQA-Diamond**（198 题，思考模式长输出）：启发式平均 77.15（范围 75.25–78.79）vs 基线 77.27。每次实验的高方差（±3 分）源于小样本量；两者均在噪声范围内。
- **LongBench V1**（~5,000 任务，平均 ISL ~10K）：各 8 次独立实验。基线范围 44.25–45.01，启发式范围 43.73–44.67。差值（−0.33）远小于 ±0.76 的运行间波动。
- **LongBench V2**（215 任务，平均 ISL ~130K）：各 5 次独立实验。基线范围 47.44–52.09，启发式范围 47.91–50.23。差值（−0.46）相比基线 ±4.65 的自身波动可忽略——这是该小规模长上下文基准测试的固有特性。

**复现方法。** 所有精度实验使用相同的 YAML 配置（关键字段如下）传递给 `trtllm-eval`。切换启发式和基线只需设置 `enable_heuristic_topk` 为 `true` 或 `false`：


### 端到端吞吐量

在 B200 ×8 上使用 DeepSeek-V3.2 NVFP4（EP8，MTP-1，NVFP4 量化，FP8 KV cache）进行端到端最小延迟推理基准测试，使用 `trtllm-bench` 和**长上下文合成数据集**（ISL=16K，OSL=131K，batch=1，concurrency=1）。该工作负载压力测试解码阶段——生成 131K 个输出 token，序列长度从 16K 增长到约 147K——正是启发式 Top-K 最活跃的阶段。

<div align="center">

| 指标 | 基线 | 启发式 | 差值 |
|:------:|:------:|:--------:|:------:|
| **TPOT** (ms) | 9.456 | 9.410 | **−0.49%** |
| **输出吞吐量** (tps) | 105.66 | 106.18 | **+0.49%** |
| **每 GPU 输出** (tps/gpu) | 13.21 | 13.27 | **+0.49%** |
| **TTFT** (ms) | 1051.19 | 1051.01 | −0.02% |
| **总延迟** (s) | 1240.5 | 1234.5 | **−0.48%** |

</div>

<sub><em>表 5. B200 ×8 最小延迟场景，Batch=1，ISL=16K，OSL=131K，EP8+DP8，MTP-1。合成随机数据集（通过 `prepare_dataset.py` 以固定 ISL/OSL 生成）。</em></sub>

端到端改进适度（~0.5%），原因有二：（1）Top-K 内核仅是每个解码步中 MoE、注意力、MLP 和通信等众多组件之一；（2）合成数据集的 indexer 分数时序相关性弱于真实推理数据（如 SWE-Bench），限制了启发式内核的优势。在具有更强时序相关性和更长有效上下文的真实 Agentic 工作负载上，逐步的 Top-K 节省在 100K+ 解码步上累积——每步约 6 µs 的节省 × 65K 解码迭代 ≈ 0.4 s 的总墙时间缩减，与观测到的 6 s 延迟下降（1240.5 → 1234.5 s）一致。

## 未来工作

- **多 CTA 启发式支持超长序列**：当前单 CTA 设计在 $N > 200$K 时回退到 radix-select。将启发式方法扩展到多 CTA 协作拆分方案，可解锁超长序列的加速，通过 CTA 级并行提升 GPU 占用率和吞吐量，并消除分发逻辑中的最后一条回退路径。
- **预填充阶段的解析预测方法**：预填充阶段不存在上一步的 Top-K 作为 `preIdx`。研究解析推导的预测信号——如静态 RoPE/YaRN 频率先验 $f(\Delta)$ 或同一预填充 batch 内的查询间相关性——以在无时序历史的情况下启动启发式路径，从而有望加速预填充阶段的 indexer Top-K。
- **跨模型泛化验证**：在其他关键路径中使用 Top-K 选择的稀疏注意力架构上验证时序相关引导的 Top-K，如 RocketKV 的生成阶段动态 Top-K 和 NSA 风格的块稀疏选择，将该方法确立为通用的稀疏注意力加速原语。
- **多 batch 与变长序列统一调优**：当前内核针对 batch=1 最小延迟场景优化。扩展到多 batch 解码工作负载和 MTP > 1 投机解码产生的变长序列场景，其中各请求序列长度分化，统一的内核参数配置（线程数、迭代预算、候选容量）需要自适应调优。
- **下一代 GPU 架构适配**：面向 Blackwell 之后的架构（如 sm\_120+）进行移植和优化，利用预期的共享内存容量增长、warp 级归约原语改进和 L2 缓存带宽提升，进一步降低 `blockCountGE` 和直方图操作的单次迭代开销。

## 致谢

本工作展示了**数据感知算法设计**——根据目标工作负载的统计特性定制 GPU 内核算法——能够带来超越分布无关方法的显著性能提升。我们欢迎社区贡献者共同拓展这一方向，无论是新的预测策略、跨模型验证还是下一代架构适配。TensorRT-LLM 是开源项目——诚邀您加入我们，共建 Agent 时代更快的 GPU 推理生态。
