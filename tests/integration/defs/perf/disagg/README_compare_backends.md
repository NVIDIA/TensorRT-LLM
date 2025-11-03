# Backend性能对比工具

## 功能说明

比较DEFAULT backend（当前为NIXL）和UCX的性能测试结果，**只关注DEFAULT比UCX慢的情况**。

### 核心特性

- 🎯 **单向比较**: 只在DEFAULT比UCX慢且超过阈值时才标记为Fail
- ✅ **性能提升视为Pass**: DEFAULT比UCX快时总是Pass（这是好事！）
- 📊 **可配置DEFAULT**: 支持切换DEFAULT backend（将来可能从NIXL切换到其他backend）
- 🔢 **智能去重**: 自动去除测试用例名称中的序号（如_001, _015）

## 使用方法

### 基本用法（使用默认NIXL作为DEFAULT）

```bash
python compare_backends.py --csv-path perf_script_test_results.csv --threshold 5.0
```

### 生成CSV和HTML报告

```bash
python compare_backends.py \
    --csv-path perf_script_test_results.csv \
    --threshold 5.0 \
    --output backend_comparison.csv \
    --html backend_comparison.html
```

### 指定其他DEFAULT backend

```bash
# 如果将来切换到其他backend作为DEFAULT
python compare_backends.py \
    --csv-path perf_script_test_results.csv \
    --threshold 5.0 \
    --default-backend OTHER_BACKEND \
    --html report.html
```

## 参数说明

| 参数 | 必填 | 默认值 | 说明 |
|------|------|--------|------|
| `--csv-path` | ✅ | - | 性能测试结果CSV文件路径 |
| `--threshold` | ❌ | 5.0 | 性能退化阈值（百分比）。只在DEFAULT比UCX慢超过此值时Fail |
| `--default-backend` | ❌ | NIXL | DEFAULT backend名称（当前为NIXL，将来可能切换） |
| `--output` | ❌ | - | 输出CSV文件路径 |
| `--html` | ❌ | - | 输出HTML报告文件路径 |

## 输出说明

### CSV输出列

- **test_case_name_default**: DEFAULT backend的测试用例名称
- **test_case_name_ucx**: UCX backend的测试用例名称
- **metric_type**: 性能指标类型（DISAGG_SERVER_TTFT / DISAGG_SERVER_E2EL）
- **default_value**: DEFAULT backend的性能值（ms）
- **ucx_value**: UCX backend的性能值（ms）
- **diff_pct**: 绝对差异百分比
- **regression_pct**: 退化/提升百分比（正值=退化，负值=提升）
- **status**: Pass/Fail

### HTML报告特性

- 📊 **可视化统计**: 总计、通过、性能退化数量一目了然
- 🎨 **智能颜色标识**: 
  - ✅ Pass显示绿色
  - ❌ Fail显示红色
  - 正值退化用红色，负值提升用绿色
- 📈 **性能趋势**: 清晰展示DEFAULT相对UCX的性能变化
- 📱 **响应式设计**: 适配不同屏幕尺寸
- 🔍 **悬停效果**: 鼠标悬停时行高亮

## 判定规则

### ✅ Pass条件
1. DEFAULT和UCX都有有效数据
2. **DEFAULT比UCX慢不超过阈值**（允许略慢）
3. **DEFAULT比UCX快**（性能提升，无论多少都是Pass！😊）

### ❌ Fail条件
1. DEFAULT或UCX缺少数据
2. **DEFAULT比UCX慢超过阈值**（性能退化）

### 示例说明

假设阈值为5%：

| 场景 | DEFAULT值 | UCX值 | 退化% | 状态 | 说明 |
|------|-----------|-------|-------|------|------|
| 场景1 | 200ms | 200ms | 0% | ✅ Pass | 性能一致 |
| 场景2 | 210ms | 200ms | +5% | ✅ Pass | 慢5%，刚好在阈值内 |
| 场景3 | 212ms | 200ms | +6% | ❌ Fail | 慢6%，超过阈值 |
| 场景4 | 190ms | 200ms | -5% | ✅ Pass | 快5%，性能提升！ |
| 场景5 | 100ms | 200ms | -50% | ✅ Pass | 快50%，大幅提升！ |

**关键点**: 只要DEFAULT不比UCX慢太多（不超过阈值），或者更快，就是Pass！

## 示例

```bash
# 使用5%阈值进行对比
python compare_backends.py \
    --csv-path perf_script_test_results.csv \
    --threshold 5.0 \
    --output results.csv \
    --html results.html

# 查看HTML报告
# Windows: start results.html
# Linux: xdg-open results.html
# macOS: open results.html
```

## 注意事项

1. ✅ 脚本会自动过滤出`disagg_perf`相关的测试用例
2. ❌ `wideep_perf`和`wideep_accuracy`类型的测试会被忽略
3. 🔢 测试用例名称中的序号（如_001, _015）会被自动规范化
4. 🚨 如果有失败用例（性能退化），脚本会返回退出码1
5. 📊 对于TTFT和E2EL指标，数值越小越好

## 输出示例

### 终端输出
```
CSV结果已保存到: results.csv
HTML报告已保存到: results.html

============= 统计信息 =============
DEFAULT Backend: NIXL
对比Backend: UCX
阈值: 5.0%
-----------------------------------
总计: 24
通过: 20 (DEFAULT性能正常)
失败: 4 (DEFAULT比UCX慢超过5.0%)
===================================
```

### HTML报告包含

- 🔍 标题显示当前DEFAULT backend
- ⚠️ 清晰的判定规则说明
- 📊 汇总卡片（总计/通过/性能退化）
- 📋 详细对比表格（含退化/提升百分比）
- 🎨 颜色编码的状态和趋势标识

