# Design: SJF (Shortest Job First) Waiting Queue Policy

## 1. Problem Statement

TRT-LLM 当前的 WaitingQueue 只支持 FCFS（先来先服务）策略。在 prompt 长度差异大的混合负载（如 agent 场景、长 system prompt + 短 user input）中，FCFS 会导致：

- **队头阻塞**：一个 32K+ token 的长请求占住队头，阻塞后面大量短请求
- **P99 延迟飙升**：短请求被长请求阻塞，TTFT P99 显著增大
- **吞吐浪费**：短请求本可快速完成释放 KV cache，但被迫等待

vLLM 的实验数据显示：变长场景吞吐提升 6.2%，Agent 场景 E2E 延迟优化 7.3%。

## 2. Proposed Solution

在 WaitingQueue 层实现 SJF 策略，让短请求优先从等待队列进入 active_requests。同时用等待时间老化（aging）机制防止长请求饿死。

### 核心设计

**评分公式**（参考 vLLM，但修正了已知问题）：

```
score = length_weight × length_score + time_weight × time_score
```

其中：
- `length_score = 1 / (1 + len(prompt_tokens) / length_median)` — prompt 越短，score 越高（范围 0~1）
- `time_score = wait_time / time_median` — 等待越久，score 越高（无上界，自然老化）
- `wait_time = current_time - arrival_time`

**与 vLLM 的关键差异**：

| 方面 | vLLM | TRT-LLM (本设计) |
|------|------|------------------|
| score 计算时机 | `__lt__` 中动态调用 `time.time()` | 每轮 pop 前批量计算一次，避免堆比较中的开销 |
| 数据结构 | heap + WeightedScoreSorter wrapper | 简单 list + 每轮 sort（请求数通常 <1000） |
| length_score | `reverse_len` flag 控制 | `1/(1+len/median)` 单调递减，无需 flag |
| remove 操作 | 创建新 wrapper 可能 hash 不匹配 | 基于 request_id set 过滤，天然无此问题 |
| 配置暴露 | 代码内常量 | SchedulerConfig.sjf_config 子字段，YAML 可配 |

**为什么用 list+sort 而不是 heap**：
- WaitingQueue 在每次调度循环中会被 `get_from_waiting_queue()` 遍历（peek → pop 循环）
- 队列长度通常 <1000（受限于 max_num_sequences），sort 开销可忽略
- list 支持高效的 `remove_by_ids`（O(n) filter），heap 的 remove 更复杂
- `prepend_requests` 在 list 上简单直接，heap 上需要 re-heapify
- 简单实现更容易调试和维护

## 3. Constraints & Invariants

1. **WaitingQueue ABC 接口不变** — SJFWaitingQueue 必须实现所有 abstract 方法
2. **`prepend_requests` 语义** — 被 prepend 回来的请求是已经 pop 出去但无法处理的（attention_dp 不满足），它们应该保持原来的优先级顺序（不重新排序）
3. **线程安全** — WaitingQueue 只在 executor loop 的单线程中访问（由 `_util.py` 和 `request_utils.py` 调用），无需额外同步
4. **arrival_time 可用性** — `item.request.py_arrival_time` 通过 `base_worker.py:592` 设置，来自 `request.arrival_time`。如果为 None，fallback 到 `time.time()` at add_request time
5. **不影响 CapacityScheduler** — SJF 只控制 WaitingQueue 出队顺序，不影响后续的 capacity/microbatch 调度
6. **兼容 MTP/spec decoding** — child_req_ids 的请求作为整体参与排序（使用 parent 的 score）
7. **Pydantic 配置类** — SJF 参数通过 `SjfConfig` Pydantic model 暴露，嵌入 `SchedulerConfig`

## 4. API Design

### 4.1 枚举扩展

```python
class WaitingQueuePolicy(StrEnum):
    FCFS = "fcfs"
    SJF = "sjf"
```

### 4.2 SJF 配置

```python
class SjfConfig(StrictBaseModel):
    length_median: int = Field(default=32768, description="Median prompt length for normalization")
    time_median: float = Field(default=5.0, description="Median wait time in seconds for normalization")
    length_weight: float = Field(default=0.5, description="Weight for length-based score")
    time_weight: float = Field(default=0.5, description="Weight for wait-time-based score")
```

放在 `llm_args.py` 中，作为 `SchedulerConfig` 的 optional 字段：

```python
class SchedulerConfig(StrictBaseModel, PybindMirror):
    ...
    sjf_config: Optional[SjfConfig] = Field(default=None, description="SJF scheduling parameters")
```

### 4.3 YAML 配置示例

```yaml
scheduler_config:
  capacity_scheduler_policy: MAX_UTILIZATION
  waiting_queue_policy: sjf
  sjf_config:
    length_median: 32768
    time_median: 5.0
    length_weight: 0.5
    time_weight: 0.5
```

### 4.4 SJFWaitingQueue 类

```python
class SJFWaitingQueue(WaitingQueue):
    def __init__(self, sjf_config: Optional[SjfConfig] = None):
        self._requests: list[RequestQueueItem] = []
        self._prepended: list[RequestQueueItem] = []  # prepend 回来的高优请求
        self._sorted = False  # dirty flag, sort 只在 peek/pop 时惰性触发
        self._config = sjf_config or SjfConfig()
        self._arrival_times: dict[int, float] = {}  # request_id → fallback arrival time

    def add_request(self, request: RequestQueueItem) -> None:
        # 记录 fallback arrival time（如果 py_arrival_time 不可用）
        if not hasattr(request.request, 'py_arrival_time') or request.request.py_arrival_time is None:
            self._arrival_times[request.id] = time.time()
        self._requests.append(request)
        self._sorted = False  # 标记需要重新排序

    def _get_arrival_time(self, item: RequestQueueItem) -> float:
        """获取请求到达时间，优先用 py_arrival_time，fallback 到 side dict。"""
        arrival = getattr(item.request, 'py_arrival_time', None)
        if arrival is not None:
            return arrival
        return self._arrival_times.get(item.id, time.time())

    def _compute_score(self, item: RequestQueueItem, now: float) -> float:
        """计算请求的 SJF score，越高越优先。"""
        prompt_len = len(item.request.input_token_ids) if item.request.input_token_ids else 0
        arrival_time = self._get_arrival_time(item)
        wait_time = max(0.0, now - arrival_time)

        length_score = 1.0 / (1.0 + prompt_len / self._config.length_median)
        time_score = wait_time / self._config.time_median

        return (self._config.length_weight * length_score
                + self._config.time_weight * time_score)

    def _ensure_sorted(self) -> None:
        """惰性排序：只在需要 peek/pop 且 dirty 时排序。"""
        if not self._sorted and self._requests:
            now = time.time()
            self._requests.sort(key=lambda item: self._compute_score(item, now), reverse=True)
            self._sorted = True

    def peek_request(self) -> RequestQueueItem:
        if self._prepended:
            return self._prepended[0]
        self._ensure_sorted()
        if not self._requests:
            raise IndexError("peek from an empty queue")
        return self._requests[0]

    def pop_request(self) -> RequestQueueItem:
        # prepended 请求优先（它们是被退回的高优请求）
        if self._prepended:
            item = self._prepended.pop(0)
            self._arrival_times.pop(item.id, None)
            return item
        self._ensure_sorted()
        if not self._requests:
            raise IndexError("pop from an empty queue")
        item = self._requests.pop(0)
        self._arrival_times.pop(item.id, None)
        return item
```

**peek/pop 一致性保证**：使用 `_sorted` dirty flag 实现惰性排序。`add_request` 设置 dirty=True，`_ensure_sorted()` 在首次 peek/pop 时排序一次。后续连续的 peek→pop 调用（如 `get_from_waiting_queue` 中的循环）不会重新排序，保证返回同一个 item。

**arrival_time fallback**：`RequestQueueItem` 没有 `_added_time` 字段，使用 `self._arrival_times` side dict 在 `add_request` 时记录 fallback 时间戳。`pop_request` 和 `remove_by_ids` 时清理。

**input_token_ids 含义**：`item.request.input_token_ids` 是原始 prompt token IDs（在 WaitingQueue 层尚未经过 chunking），反映的是完整 prompt 长度。这正是 SJF 需要的：按原始 prompt 长度排序，因为 prefill 成本与原始长度成正比。

**关键设计决策：`_prepended` 分离**

`prepend_requests` 是 `get_from_waiting_queue()` 将无法处理的请求退回队列时调用的。这些请求已经被 pop 出来过（即它们曾经是最高优先级的），退回后应该在下一轮优先重试。如果把它们混入普通列表重新排序，可能因为 score 变化导致顺序改变，产生不必要的反复 pop/prepend。

**`prepend_requests` 顺序语义**：调用方 `request_utils.py:155` 传入 `reversed(pending_requests)`，FCFS 的 `extendleft` 会再次反转回原序。SJFWaitingQueue 的 `prepend_requests` 接收 iterable 后直接 `list(requests) + self._prepended` 拼接到前面，保持传入顺序。

**`__iter__`/`__bool__`/`__len__`** 需要同时考虑 `_prepended` 和 `_requests` 两个列表。

## 5. Failure Modes

| 失败场景 | 影响 | 恢复 |
|---------|------|------|
| `py_arrival_time` 为 None | wait_time 计算错误 | `add_request` 时记录 fallback 时间 |
| `input_token_ids` 为空或 None | 除零/异常 | 长度为 0 时 length_score = 1.0（最高优先） |
| 所有请求长度相同 | SJF 退化为 FCFS（按等待时间） | 正常行为，无需处理 |
| `time_median` = 0 | 除零 | Pydantic validation: `gt=0` |
| `length_median` = 0 | 除零 | Pydantic validation: `gt=0` |
| 极长等待时间 | time_score 远大于 length_score | 正常行为：老化机制确保不饿死 |

## 6. Quantitative Analysis

**排序开销估算**：
- 典型队列长度：10~500 请求（受 max_num_sequences 限制）
- Python list.sort() 对 500 元素：~0.05ms（Timsort, O(n log n)）
- 每次调度循环一次 sort，相对于一次 prefill（~10-100ms）可忽略
- 对比 heap：heap 的 `remove_by_ids` 需要 O(n) scan + O(n log n) heapify，总开销类似

**SJF 收益场景**：
- 混合长度负载（如 MLPerf DeepSeek-R1 eval dataset，自然长度分布）
- Agent 场景（长 system prompt + 短 user turn 交替）
- 长 system prompt cache hit 后实际 prefill 变短的场景

**SJF 无效场景**：
- 所有请求长度一致（ISL=8192 固定）→ 退化为 FCFS
- 队列几乎为空（低并发）→ 无排序必要

## 7. Files to Modify

| 文件 | 修改 |
|------|------|
| `tensorrt_llm/llmapi/llm_args.py` | 添加 `SJF` 枚举值, `SjfConfig` 类, `SchedulerConfig.sjf_config` 字段 |
| `tensorrt_llm/_torch/pyexecutor/scheduler/waiting_queue.py` | 添加 `SJFWaitingQueue` 类, 更新 `create_waiting_queue()` |
| `tensorrt_llm/_torch/pyexecutor/scheduler/__init__.py` | 导出 `SJFWaitingQueue` |
| `tensorrt_llm/_torch/pyexecutor/_util.py` | 传递 `sjf_config` 给 `create_waiting_queue()` |
| `tensorrt_llm/_torch/pyexecutor/py_executor.py` | 接收 `sjf_config` 参数 |
| `tests/unittest/_torch/executor/test_waiting_queue.py` | 添加 SJF 测试 |
