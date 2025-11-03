#!/bin/bash
# Backend性能对比工具使用示例
# 比较DEFAULT backend (当前为NIXL) 和 UCX 的性能

# 设置CSV文件路径
CSV_PATH="perf_script_test_results.csv"

# 示例1: 基本使用，默认NIXL作为DEFAULT，打印到终端
echo "=== 示例1: 基本使用 (DEFAULT=NIXL, 阈值=5%) ==="
python compare_backends.py \
    --csv-path "$CSV_PATH" \
    --threshold 5.0

# 示例2: 生成CSV输出
echo -e "\n=== 示例2: 生成CSV (DEFAULT=NIXL, 阈值=10%) ==="
python compare_backends.py \
    --csv-path "$CSV_PATH" \
    --threshold 10.0 \
    --default-backend NIXL \
    --output backend_comparison.csv

# 示例3: 同时生成CSV和HTML报告
echo -e "\n=== 示例3: 生成CSV和HTML (DEFAULT=NIXL, 阈值=5%) ==="
python compare_backends.py \
    --csv-path "$CSV_PATH" \
    --threshold 5.0 \
    --default-backend NIXL \
    --output backend_comparison.csv \
    --html backend_comparison.html

# 示例4: 如果将来切换到其他backend作为DEFAULT
echo -e "\n=== 示例4: 使用其他DEFAULT backend ==="
# python compare_backends.py \
#     --csv-path "$CSV_PATH" \
#     --threshold 5.0 \
#     --default-backend OTHER_BACKEND \
#     --html report_other.html

echo -e "\n✅ 完成！"
echo "可以通过浏览器打开 backend_comparison.html 查看可视化报告"
echo ""
echo "📊 报告说明:"
echo "  - 只关注 DEFAULT 比 UCX 慢的情况"
echo "  - DEFAULT 比 UCX 快时总是 Pass (性能提升!)"
echo "  - DEFAULT 比 UCX 慢超过阈值时才 Fail (性能退化)"

