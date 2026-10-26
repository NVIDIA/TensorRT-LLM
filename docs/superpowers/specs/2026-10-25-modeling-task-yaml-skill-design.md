# 设计：交互式生成 agent-flow modeling task.yaml 的 skill

## 目的与动机

在 agent-flow 中接入一个新模型时，用户需要手写 `task.yaml`。schema 本身很简单：三个路径、
可选的 `slurm-environment`、两个自由文本列表 `completion_criteria` 和 `implements_tips`。
真正决定一次运行成败的内容，比如精度标准、要验证哪些功能、环境信息，都只能靠用户用自然语言
写进这两个列表。这些内容容易漏写，也容易写得模糊。

这个 skill 通过对话引导用户填写：
- 已知字段逐项确认；
- 用户没有主动提到、但很重要的字段，主动询问；
- 能从 checkpoint 推断出来的信息，先推断再请用户确认；

最后产出一份能直接交给工作流运行的 `task.yaml`。

## 目标

- 覆盖 `modeling-bringup` 和 `modeling-v2-onboard` 共用的 task.yaml 部分。两者共用
  `agent-flow/agent_flow/workflows/modeling_bringup/task_schema.py`，该文件已在 main 上。
- 分四个部分对话：基础配置 → 运行方式 → 精度验证 → 功能验证。
- 把用户的选择展开成清晰、可验证的英文句子，写进 `completion_criteria` 和 `implements_tips`。
- 写出文件后，用 `load_and_validate_task_yaml` 校验，并告诉用户怎样运行。

## 非目标

- `modeling-v2-onboard` 特有的内容：`gpu_pool`（节点池）、并行拓扑选择、性能 Pareto、性能模型。
  这些等 v2-onboard 合入 main 后再扩展。
- 修改 agent-flow 的 schema 或工作流代码。
- 不问、也不写入环境细节：PATH/PYTHONPATH/HOME、HF 离线变量、MPI/UCX 变量、编译产物软链接。
  用户主动提供时，可以原样写进 `implements_tips`。

## 反目标（技术上能用，但仍算失败）

- 悄悄填入用户没听说过的默认值。所有默认值都必须在对话里说出来。
- 一次抛出一长串问题，让用户像填表一样回答。
- 生成的 task.yaml 通不过 schema 校验，或者里面的路径不存在。

## Skill 位置与形式

- 位置：`.claude/skills/modeling-task-yaml/SKILL.md`。名字按 `.claude/README.md` 的
  `<prefix>-<descriptive-name>` 规则取。（我的决定）
- `SKILL.md` 描述对话流程。可以附带一份带注释的模板，或者一份标准句子库，用来存放下文
  要写入的英文句子。
- schema 校验直接调用 agent-flow 里现有的 `load_and_validate_task_yaml`，不要另写一份
  schema。导入方式：直接按文件路径加载 `task_schema.py`，例如用 `importlib` 从文件加载。
  原因是通过 `agent_flow` 包导入会连带导入 `claude_agent_sdk`，而这个包在部分机器上
  不可用。（我的决定）
- 验证方式：拿一个真实 checkpoint 手动走一遍对话流程，不加自动化测试。（我的决定）

## 对话流程

总的原则：
- 第 1、2 部分各用一张表单，一条消息问完这一部分的所有字段；第 3、4 部分一次只问一件事。
- 有固定候选的问题，用选择题，即 AskUserQuestion。
- 每个部分结束时，用表格逐字段列出具体值、来源和检查结果，请用户确认；不能只说"已核实"。

### 第 1 部分：基础配置

| 项 | 写入位置 | 检查 |
|---|---|---|
| `reference_code_path`（HF modeling 文件） | 顶层 | 必须存在 |
| `checkpoint_path`（HF checkpoint 目录） | 顶层 | 必须存在，并且有 `config.json` |
| `trtllm_repo_path` | 顶层 | 必须存在，并且是 git checkout |
| 目标 GPU（例如 GB300 / sm_103） | `implements_tips` 的一句话 | 能从 partition 或本机推断时，先推断再确认 |
| `LLM_MODELS_ROOT` | `implements_tips` 的一句话 | 目录必须存在；检查下面有没有 `datasets/openai/gsm8k`，选了多模态时还要检查 MMMU 数据，没有就提醒用户 |

拿到 checkpoint 之后，读取 `config.json` 等文件，推断以下信息，供后面几部分使用：
- 架构名；
- 是否多模态，依据包括 `vision_config`、`preprocessor_config.json` 等；
- 是否带 MTP 层，例如 `num_nextn_predict_layers`；
- 是否有 `chat_template.jinja`。

推断出来的结论要说给用户听。

### 第 2 部分：运行方式

先问"是否使用 slurm 模式"。

- 不用：不写 `slurm-environment`。
- 用：问以下 5 项，全部写在 `slurm-environment` 下面，不写 `gpu_pool`。
  - `slurm_partition`
  - `docker_image`：`.sqsh` 文件必须存在
  - `account`
  - `qos`
  - `container_mounts`

`modeling-bringup` 的 prompt 会让 agent 读取整段 `slurm-environment`，所以 account、qos、
`container_mounts` 写在这里，agent 能看到。

### 第 3 部分：精度验证

**e2e 精度**
- 默认用 GSM8K，通过 `trtllm-eval` 跑。标准是和 reference 分数相差小于 1%（绝对分差）。
- 先问用户有没有 reference 分数：
  - 有：把具体数值写进标准。
  - 没有：先判断 HF 实现能不能跑这个 checkpoint，依据是 `config.json` 里的
    `quantization_config`（例如 NVFP4 这类 TRT-LLM 专用格式 HF 跑不了）和权重大小
    （能否在目标节点上用 HF 单进程跑）。
    - 能跑：写成"agent 用 HF 实现（`reference_code_path`），在同样的评测设置下自己跑出
      reference 分数，再和它比较"。
    - 跑不了：把原因告诉用户，并要求用户提供以下任意一种：一个具体分数；或者另一个基准，
      比如同一模型的 BF16 checkpoint 路径，或者公开发布的分数。不能写一条达不到的标准。
- 用户可以换数据集，也可以改容忍度。
- 如果第 4 部分勾选了多模态，再加 MMMU，规则和 GSM8K 完全相同。
- 如果存在 `chat_template.jinja`，在 `implements_tips` 里提醒：评测时要确认是否需要套用
  chat template。

**单 op 精度**：默认写入 `completion_criteria`，原文就用这句：

> "Every op matches its reference within 1 ulp of the output dtype (relative to each row's
> max magnitude). A wider band needs a stated derivation. A faster replacement need not match
> the previous op bit for bit."

### 第 4 部分：功能验证

用多选题，候选如下：

| 功能 | 默认 |
|---|---|
| CUDA graph | 勾选 |
| overlap scheduler | 勾选 |
| chunked prefill | 勾选 |
| KV cache reuse | 不勾选 |
| speculative decoding | 不勾选 |
| disaggregated serving | 不勾选 |
| 多模态（视觉部分） | 不勾选；只有第 1 部分识别为多模态时才出现 |

- **验证方式**：所有勾选的功能同时打开，在这个配置下跑第 3 部分的 e2e 精度验证
  （GSM8K，勾选多模态时还有 MMMU），并达到标准。写成一条 `completion_criteria`，
  其中列出全部勾选的功能。
- **disagg 例外**：`trtllm-eval` 在进程内跑模型，不支持 disagg。所以 disagg 不放进上面那次
  合并运行，单独写一条标准："GSM8K（勾选多模态时还有 MMMU）通过 `trtllm-serve` 的 disagg
  服务来跑，标准和 e2e 精度相同"。服务怎么搭建交给 planner 推断。
- **spec decoding / disagg**：勾选后，问一句"有没有要补充的说明"，用户可以不填。
  - 填了：把说明写进 `implements_tips`。
  - 没填：写一句 "left to the planner to infer from the checkpoint and the codebase"。
  - checkpoint 带 MTP 层时，可以把这个事实告诉用户，作为参考。
- **多模态**：
  - 勾选：精度标准加上 MMMU。
  - 识别为多模态但没有勾选：在 `implements_tips` 里写明
    "text model only; vision tower out of scope"。

### 收尾

1. 展示完整的 task.yaml 草稿，请用户确认或修改。
2. 问 workspace 路径，把 `task.yaml` 写进去。如果目标文件已经存在，先问用户，不能直接覆盖。
3. 用 agent-flow 的 `load_and_validate_task_yaml` 校验。校验失败时，把错误告诉用户，
   改完再校验。
4. 告诉用户运行命令，例如 `modeling-bringup --task <path> --workspace <path>`。

## 交给实现者决定的事

- skill 的具体名字和 frontmatter 里的 description 怎么写。description 要让 Claude 在
  "写 / 生成 modeling task.yaml、接入新模型到 agent-flow" 这类请求下触发。
- 标准句子的最终英文措辞。除单 op 精度那句以外，其余句子保持精简、可以验证即可。
- 用模板文件还是在 SKILL.md 里写示例。
- 怎样找到 agent-flow 里的 `task_schema.py`：优先用 `trtllm_repo_path` 下的
  `agent-flow/`，找不到时再问用户。
