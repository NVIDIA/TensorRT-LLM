# TimeoutManager 使用指南

## 概述

`TimeoutManager` 是一个用于简化测试用例中超时处理的工具类。它可以帮助减少重复的超时检查代码，使测试代码更加简洁和易于维护。

## 问题背景

在现有的测试用例中，经常可以看到这样的重复模式：

```python
remaining_timeout = timeout_from_marker
convert_start = time.time()
# ... 执行操作 ...
convert_time = time.time() - convert_start
remaining_timeout -= convert_time
if remaining_timeout <= 0:
    raise TimeoutError("Timeout exceeded after convert phase!")
```

这种模式在每个阶段都会重复，导致代码冗长且容易出错。

## 解决方案

`TimeoutManager` 提供了三种使用方式来简化超时处理：

### 1. 直接使用 TimeoutManager

```python
from ..utils.timeout_manager import TimeoutManager

def test_example(timeout_from_marker):
    timeout_manager = TimeoutManager(timeout_from_marker)

    # 使用 execute_with_timeout 方法
    result = timeout_manager.execute_with_timeout(
        lambda: some_operation(),
        phase_name="convert"
    )
```

### 2. 使用上下文管理器

```python
def test_example(timeout_from_marker):
    timeout_manager = TimeoutManager(timeout_from_marker)

    # 使用 timed_operation 上下文管理器
    with timeout_manager.timed_operation("convert"):
        result = some_operation()
```

### 3. 使用装饰器

```python
from ..utils.timeout_manager import with_timeout_management

@with_timeout_management
def test_example(timeout_from_marker, timeout_manager):
    # timeout_manager 会自动注入
    with timeout_manager.timed_operation("convert"):
        result = some_operation()
```

## 完整示例对比

### 原始代码（冗长）

```python
def test_llm_commandr_plus_4gpus_summary(commandr_example_root,
                                         llm_commandr_plus_model_root,
                                         llm_datasets_root, llm_rouge_root,
                                         llm_venv, cmodel_dir, engine_dir,
                                         use_weight_only, timeout_from_marker):
    dtype = 'float16'
    tp_size = 4
    model_name = os.path.basename(llm_commandr_plus_model_root)

    # Convert phase
    print("Converting checkpoint...")
    remaining_timeout = timeout_from_marker
    convert_start = time.time()
    ckpt_dir = convert_weights(
        llm_venv=llm_venv,
        example_root=commandr_example_root,
        cmodel_dir=cmodel_dir,
        model=model_name,
        model_path=llm_commandr_plus_model_root,
        data_type=dtype,
        tp_size=tp_size,
        gpus=tp_size,
        use_weight_only=use_weight_only,
        timeout=remaining_timeout
    )
    convert_time = time.time() - convert_start
    remaining_timeout -= convert_time
    if remaining_timeout <= 0:
        raise TimeoutError("Timeout exceeded after convert phase!")

    # Build phase
    print("Building engines...")
    build_start = time.time()
    build_cmd = ["trtllm-build", ...]
    check_call(" ".join(build_cmd),
               shell=True,
               env=llm_venv._new_env,
               timeout=remaining_timeout)
    build_time = time.time() - build_start
    remaining_timeout -= build_time
    if remaining_timeout <= 0:
        raise TimeoutError("Timeout exceeded after build phase!")

    # Run phase
    print("Running engines...")
    run_start = time.time()
    run_cmd = [...]
    venv_mpi_check_call(llm_venv, ["mpirun", "-n", str(tp_size)], run_cmd,
                        timeout=remaining_timeout)
    run_time = time.time() - run_start
    remaining_timeout -= run_time
    if remaining_timeout <= 0:
        raise TimeoutError("Timeout exceeded after run phase!")

    # Summary phase
    print("Running summary...")
    summary_start = time.time()
    summary_cmd = generate_summary_cmd(...)
    venv_mpi_check_call(llm_venv, ["mpirun", "-n", str(tp_size)], summary_cmd,
                        timeout=remaining_timeout)
    summary_time = time.time() - summary_start
    remaining_timeout -= summary_time
    if remaining_timeout <= 0:
        raise TimeoutError("Timeout exceeded after summary phase!")
```

### 重构后的代码（简洁）

```python
@with_timeout_management
def test_llm_commandr_plus_4gpus_summary(commandr_example_root,
                                         llm_commandr_plus_model_root,
                                         llm_datasets_root, llm_rouge_root,
                                         llm_venv, cmodel_dir, engine_dir,
                                         use_weight_only, timeout_manager):
    dtype = 'float16'
    tp_size = 4
    model_name = os.path.basename(llm_commandr_plus_model_root)

    # Convert phase
    print("Converting checkpoint...")
    with timeout_manager.timed_operation("convert"):
        ckpt_dir = convert_weights(
            llm_venv=llm_venv,
            example_root=commandr_example_root,
            cmodel_dir=cmodel_dir,
            model=model_name,
            model_path=llm_commandr_plus_model_root,
            data_type=dtype,
            tp_size=tp_size,
            gpus=tp_size,
            use_weight_only=use_weight_only,
            timeout=timeout_manager.remaining_timeout
        )

    # Build phase
    print("Building engines...")
    with timeout_manager.timed_operation("build"):
        build_cmd = ["trtllm-build", ...]
        check_call(" ".join(build_cmd),
                  shell=True,
                  env=llm_venv._new_env,
                  timeout=timeout_manager.remaining_timeout)

    # Run phase
    print("Running engines...")
    with timeout_manager.timed_operation("run"):
        run_cmd = [...]
        venv_mpi_check_call(llm_venv, ["mpirun", "-n", str(tp_size)], run_cmd,
                           timeout=timeout_manager.remaining_timeout)

    # Summary phase
    print("Running summary...")
    with timeout_manager.timed_operation("summary"):
        summary_cmd = generate_summary_cmd(...)
        venv_mpi_check_call(llm_venv, ["mpirun", "-n", str(tp_size)], summary_cmd,
                           timeout=timeout_manager.remaining_timeout)
```

## 主要优势

1. **代码简洁性**: 减少了约 60% 的样板代码
2. **可维护性**: 超时逻辑集中管理，易于修改
3. **可读性**: 测试逻辑更清晰，重点突出
4. **一致性**: 所有测试用例使用相同的超时处理模式
5. **错误处理**: 自动化的超时检查和错误报告

## API 参考

### TimeoutManager 类

#### 构造函数
```python
TimeoutManager(initial_timeout: Optional[float] = None)
```

#### 主要方法

- `remaining_timeout`: 属性，获取剩余超时时间
- `reset(timeout: Optional[float] = None)`: 重置超时管理器
- `check_timeout(phase_name: str = "operation")`: 检查是否超时
- `timed_operation(phase_name: str = "operation")`: 上下文管理器
- `execute_with_timeout(operation, phase_name, **kwargs)`: 执行带超时的操作
- `call_with_timeout(func, *args, phase_name, **kwargs)`: 调用带超时的函数

### 装饰器

- `@with_timeout_management`: 自动注入 timeout_manager 参数

## 迁移指南

要将现有测试用例迁移到使用 `TimeoutManager`：

1. 导入 `TimeoutManager` 或 `with_timeout_management`
2. 创建 `TimeoutManager` 实例或使用装饰器
3. 将每个阶段的超时处理替换为 `timed_operation` 上下文管理器
4. 移除手动的 `time.time()` 和超时检查代码

## 注意事项

1. `TimeoutManager` 会自动处理超时检查，无需手动调用 `check_timeout()`
2. 如果 `timeout_from_marker` 为 `None`，则不会进行超时检查
3. 上下文管理器会自动计算操作时间并更新剩余超时时间
4. 超时错误会包含阶段名称，便于调试
