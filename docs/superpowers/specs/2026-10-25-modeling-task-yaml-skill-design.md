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
- 写出文件后，用工作流自己的检查来校验（`modeling-v2-onboard --check`，没有 v2-onboard 时用
  `load_and_validate_task_yaml`），并告诉用户怎样运行。对话中不再重复实现这些检查。

## 非目标

- `modeling-v2-onboard` 特有的内容中，除 `gpu_pool`（节点池）以外的部分：并行拓扑选择、
  性能 Pareto、性能模型。这些等 v2-onboard 合入 main 后再扩展。
- 修改 agent-flow 的 schema 或工作流代码（`--check` 由 v2-onboard 分支提供）。
- 不问、也不写入环境细节：PATH/PYTHONPATH/HOME、HF 离线变量、MPI/UCX 变量、编译产物软链接。
  用户主动提供时，可以原样写进 `implements_tips`。

## 反目标（技术上能用，但仍算失败）

- 悄悄填入用户没听说过的默认值。所有默认值都必须在对话里说出来。
- 一次抛出一长串问题，让用户像填表一样回答。
- 生成的 task.yaml 通不过 schema 校验，或者里面的路径不存在。

## Skill 位置与形式

- 位置：`.agents/skills/modeling-task-yaml/SKILL.md`，Codex 从这里读取；
  `.claude/skills/modeling-task-yaml` 是指向它的软链接，供 Claude Code 读取。名字按
  `.claude/README.md` 的 `<prefix>-<descriptive-name>` 规则取。（我的决定）
- `SKILL.md` 描述对话流程。可以附带一份带注释的模板，或者一份标准句子库，用来存放下文
  要写入的英文句子。
- schema 校验直接调用 agent-flow 里现有的 `load_and_validate_task_yaml`，不要另写一份
  schema。导入方式：直接按文件路径加载 `task_schema.py`，例如用 `importlib` 从文件加载。
  原因是通过 `agent_flow` 包导入会连带导入 `claude_agent_sdk`，而这个包在部分机器上
  不可用。（我的决定）
- 验证方式：拿一个真实 checkpoint 手动走一遍对话流程，不加自动化测试。（我的决定）

## 对话流程

总的原则：
- 对话语言按用户在这次对话里的第一条消息决定：中文就用中文，否则用英语，之后所有消息（表单、
  确认表、提问、选项）都用这种语言，用户要求才切换。字段名、路径、命令保持原样；写进 YAML 的
  句子始终是英文。
- 第 1、2 部分各用一张表单，一条消息问完这一部分的所有字段；第 3、4 部分一次只问一件事。
- 有固定候选的问题，用当前工具自带的选择题工具（Claude Code 的 AskUserQuestion、Codex 的
  `request_user_input`）。没有这类工具，或者多选题而工具只支持单选时，用统一的文本格式：
  选项编号、默认项标 `(default)`、说明回复方式（单个编号、逗号分隔的多个编号，或 "default"）。
- 每个部分结束时，用表格逐字段列出具体值和来源，请用户确认；不能只说"已核实"。

### 第 1 部分：基础配置

| 项 | 写入位置 | 说明 |
|---|---|---|
| `reference_code_path`（HF modeling 文件） | 顶层 | — |
| `checkpoint_path`（HF checkpoint 目录） | 顶层 | — |
| `trtllm_repo_path` | 顶层 | 干净的 checkout |
| 目标 GPU（例如 GB300 / sm_103） | `implements_tips` 的一句话 | 能从本机推断时，先推断再确认 |

路径、镜像、mounts 等检查统一放在收尾时的 `--check` 里做。不再问 `LLM_MODELS_ROOT`：
运行只从 `checkpoint_path` 读权重，`LLM_MODELS_ROOT` 只用来找评测数据集，数据集改在第 3 部分直接问。

拿到 checkpoint 之后，读取 `config.json` 等文件，推断以下信息，供后面几部分使用：
- 架构名；
- 是否多模态，依据包括 `vision_config`、`preprocessor_config.json` 等；
- 是否带 MTP 层，例如 `num_nextn_predict_layers`；
- 是否有 `chat_template.jinja`。

推断出来的结论要说给用户听。

### 第 2 部分：运行方式

先问"是否使用 slurm 模式"。

- 不用：不写 `slurm-environment`。
- 用：`slurm_partition`、`docker_image`、`account`、`qos` 都先由 skill 推断再请用户确认，不让用户从空白开始填：
  有 `sinfo` 就建议用 Slurm；分区取 `sinfo` 里的默认 GPU 分区；镜像取仓库
  `jenkins/current_image_tags.properties` 里的 CI 镜像（Grace 架构 GPU 用 SBSA 版）；
  `account`、`qos` 取 `sacctmgr` 的 association。用户给的旧 task.yaml 或 memory 优先。
  agent-flow 有 v2-onboard 时，再问是否使用 GPU pool。
  - 不用 pool：`account`、`qos`、`container_mounts` 直接写在 `slurm-environment` 下面，agent 会读到。
  - 用 pool：这些字段写在 `slurm-environment.gpu_pool` 下面，不在两处重复。用户只填 `nodes` 和
    `max_nodes`；`account`、`qos`、`gpus_per_node`、`hours`、`container_mounts` 由 skill 从当前环境
    （`sacctmgr`、`sinfo`）、旧的 task.yaml 或 memory 推断，列出来源请用户确认。

### 第 3 部分：精度验证

**e2e 精度**
- 默认用 GSM8K，通过 `trtllm-eval` 跑。标准是和 reference 分数相差小于 1%（绝对分差）。
- 问数据集来源：用户自己给本地路径（缺失只提醒；Slurm 下要在某个 mount 下面），或者在线下载
  （让 agent 自己搜索并下载）。多模态的 MMMU 同样处理。
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
2. 直接给出一个新的 workspace 完整路径请用户确认或替换：`<agent-flow>/workspace/<name>`，`<agent-flow>`
   是用户的工作流命令实际安装自的 agent-flow 目录（`workspace/` 在它的 `.gitignore` 里）。这个目录在
   `trtllm_repo_path` 里面、或者还没装 agent-flow 时，改用 checkout 旁边的 `<trtllm_repo_path 的上一级>/<name>`，
   并说明原因：modeling-v2-onboard 拒绝放在 checkout 里的 workspace，因为最后一个任务会提交那里所有未跟踪的文件。
   `<name>` 是 checkpoint 目录名的小写加 `-v2-onboard` 或 `-bringup`，目录已存在就加 `-2`、`-3`。
   确认后创建目录并写入 `task.yaml`；用户换成的目录里已有 task.yaml 时，先问再覆盖。
3. 用 `modeling-v2-onboard --check` 校验（schema、checkout、workspace、checkpoint、镜像、mounts、
   GPU pool）；没有 v2-onboard 时用 `load_and_validate_task_yaml`。失败时把问题告诉用户，改完再校验。
4. Slurm + v2-onboard 时，问用户要不要现在跑运行时检查（`--check --smoke`：提交一个短 Slurm 作业，
   检查容器、GPU、mounts、CUDA PyTorch；要等节点）。不跑的话，planner 启动后由环境任务做同样的检查，
   并产出 worker 用的 wrapper。
5. 给出运行命令。modeling-v2-onboard 先问 run 里的 planner 和 worker 用 Claude Code（默认）还是 Codex，
   Codex 时命令加 `--backend codex`。只问后端：模型和推理强度用工作流代码里该后端的默认值，不问也不加
   `--model`；modeling-bringup 没有统一的后端开关，直接给命令。然后问要不要由当前会话
   代为启动：要的话用 `setsid nohup <命令> > <workspace>/launch.log 2>&1 &` 后台启动，报告 PID 和日志路径，
   并说明重跑同一条命令会接着跑；不要就由用户自己运行。

## 交给实现者决定的事

- skill 的具体名字和 frontmatter 里的 description 怎么写。description 要让 Claude 在
  "写 / 生成 modeling task.yaml、接入新模型到 agent-flow" 这类请求下触发。
- 标准句子的最终英文措辞。除单 op 精度那句以外，其余句子保持精简、可以验证即可。
- 用模板文件还是在 SKILL.md 里写示例。
- 怎样找到 agent-flow 里的 `task_schema.py`：优先用 `trtllm_repo_path` 下的
  `agent-flow/`，找不到时再问用户。
