#!/usr/bin/env python3
"""
比较不同backend(UCX vs NIXL)的性能测试结果
"""

import pandas as pd
import argparse
import re
import sys


def normalize_test_name(test_name):
    """
    去掉test_name中的序号数字（如_001, _015等）
    例如: deepseek-r1-fp4_015_8k1k -> deepseek-r1-fp4_8k1k
    """
    # 匹配 model_XXX_ 格式，去掉XXX数字
    pattern = r'(_\d{3}_)'
    normalized = re.sub(pattern, '_', test_name)
    return normalized


def extract_backend(test_name):
    """从test_name中提取backend类型"""
    match = re.search(r'ccbackend:(\w+)', test_name)
    return match.group(1) if match else None


def extract_base_case_name(test_name):
    """
    提取标准化的case名称（去除backend信息和序号）
    """
    # 先标准化去掉序号
    normalized = normalize_test_name(test_name)
    
    # 去掉ccbackend部分，保留其他参数
    # 替换 ccbackend:XXX 为 ccbackend:BACKEND
    pattern = r'ccbackend:\w+'
    base_case = re.sub(pattern, 'ccbackend:BACKEND', normalized)
    
    return base_case


def compare_backends(csv_path, threshold=5.0, default_backend='NIXL'):
    """
    比较DEFAULT backend和UCX的性能指标
    只关注DEFAULT比UCX慢的情况
    
    Args:
        csv_path: CSV文件路径
        threshold: 性能差异阈值（百分比）
        default_backend: DEFAULT backend名称（当前为NIXL，将来可能切换）
    
    Returns:
        DataFrame: 比较结果
    """
    # 读取CSV
    df = pd.read_csv(csv_path)

    if len(df) == 0:
        print(f"No data found in CSV file: {csv_path}")
        sys.exit(0)

    # 过滤只保留disagg_perf相关的测试
    # 从test_name字段判断
    df = df[df['test_name'].str.contains('disagg_perf_file:', na=False)]
    if len(df) == 0:
        print(f"No disagg_perf tests found in CSV file: {csv_path}")
        sys.exit(0)

    # 提取backend和标准化的case名称
    df['backend'] = df['test_name'].apply(extract_backend)
    df['base_case_name'] = df['test_name'].apply(extract_base_case_name)
    
    # 按base_case_name和metric_type分组
    grouped = df.groupby(['base_case_name', 'metric_type'])
    
    results = []
    
    for (base_case, metric_type), group in grouped:
        # 获取DEFAULT backend和UCX的数据
        default_data = group[group['backend'] == default_backend]
        ucx_data = group[group['backend'] == 'UCX']
        
        # 如果两者都没有数据，跳过（这个case可能不存在）
        if len(default_data) == 0 and len(ucx_data) == 0:
            continue
        
        # 提取数值
        default_value = default_data['perf_metric'].values[0] if len(default_data) > 0 else None
        ucx_value = ucx_data['perf_metric'].values[0] if len(ucx_data) > 0 else None
        
        # 判断状态
        status = 'Pass'
        diff_pct = None
        regression_pct = None
        
        # 如果一方有值另一方没有，标记为Fail（测试运行失败）
        if default_value is None or ucx_value is None:
            status = 'Fail'
        elif ucx_value != 0:
            # 计算性能差异百分比
            # 对于TTFT和E2EL这种指标，数值越小越好
            # regression_pct > 0 表示DEFAULT比UCX慢（性能退化）
            # regression_pct < 0 表示DEFAULT比UCX快（性能提升）
            regression_pct = ((default_value - ucx_value) / ucx_value) * 100
            diff_pct = abs(regression_pct)
            
            # 只在DEFAULT比UCX慢且超过阈值时才Fail
            if regression_pct > threshold:
                status = 'Fail'
            else:
                status = 'Pass'
        else:
            # UCX值为0是异常情况
            if default_value != 0:
                status = 'Fail'
        
        # 构建输出行
        test_case_name_default = base_case.replace('ccbackend:BACKEND', f'ccbackend:{default_backend}')
        test_case_name_ucx = base_case.replace('ccbackend:BACKEND', f'ccbackend:UCX')
        
        results.append({
            'test_case_name_default': test_case_name_default,
            'test_case_name_ucx': test_case_name_ucx,
            'metric_type': metric_type,
            'default_value': default_value,
            'ucx_value': ucx_value,
            'diff_pct': diff_pct,
            'regression_pct': regression_pct,
            'status': status
        })
    
    # 转换为DataFrame
    result_df = pd.DataFrame(results)
    
    return result_df


def generate_html_report(result_df, threshold, default_backend, output_path):
    """生成HTML格式的比较报告"""
    
    # 统计信息
    total = len(result_df)
    failed = len(result_df[result_df['status'] == 'Fail'])
    passed = total - failed
    
    # HTML模板
    html_template = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Backend性能对比报告 - DEFAULT vs UCX</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1600px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            border-bottom: 3px solid #4CAF50;
            padding-bottom: 10px;
        }}
        .summary {{
            display: flex;
            justify-content: space-around;
            margin: 30px 0;
        }}
        .summary-box {{
            flex: 1;
            margin: 0 10px;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            color: white;
        }}
        .summary-box.total {{
            background-color: #2196F3;
        }}
        .summary-box.pass {{
            background-color: #4CAF50;
        }}
        .summary-box.fail {{
            background-color: #f44336;
        }}
        .summary-box h2 {{
            margin: 0;
            font-size: 36px;
        }}
        .summary-box p {{
            margin: 5px 0 0 0;
            font-size: 14px;
        }}
        .info {{
            background-color: #e3f2fd;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
        }}
        .warning-box {{
            background-color: #fff3e0;
            border-left: 4px solid #ff9800;
            padding: 15px;
            margin-bottom: 20px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
            font-size: 14px;
        }}
        th {{
            background-color: #4CAF50;
            color: white;
            padding: 12px;
            text-align: left;
            position: sticky;
            top: 0;
        }}
        td {{
            padding: 10px;
            border-bottom: 1px solid #ddd;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .status-pass {{
            background-color: #4CAF50;
            color: white;
            padding: 5px 10px;
            border-radius: 4px;
            font-weight: bold;
        }}
        .status-fail {{
            background-color: #f44336;
            color: white;
            padding: 5px 10px;
            border-radius: 4px;
            font-weight: bold;
        }}
        .metric-type {{
            background-color: #2196F3;
            color: white;
            padding: 3px 8px;
            border-radius: 3px;
            font-size: 12px;
        }}
        .regression {{
            color: #f44336;
            font-weight: bold;
        }}
        .improvement {{
            color: #4CAF50;
            font-weight: bold;
        }}
        .neutral {{
            color: #666;
        }}
        .test-name {{
            font-family: monospace;
            font-size: 12px;
            word-break: break-all;
        }}
        .footer {{
            margin-top: 30px;
            text-align: center;
            color: #666;
            font-size: 12px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🔍 Backend性能对比报告: DEFAULT ({default_backend}) vs UCX</h1>
        
        <div class="info">
            <strong>DEFAULT Backend:</strong> {default_backend}
            <br>
            <strong>对比Backend:</strong> UCX
            <br>
            <strong>阈值设置:</strong> {threshold}%
            <br>
            <strong>说明:</strong> 只关注DEFAULT比UCX慢的情况。性能退化超过阈值时标记为Fail
        </div>
        
        <div class="warning-box">
            <strong>⚠️ 注意:</strong> 
            <ul style="margin: 5px 0;">
                <li>✅ <strong>Pass</strong>: DEFAULT性能与UCX接近，或比UCX更好</li>
                <li>❌ <strong>Fail</strong>: DEFAULT比UCX慢超过{threshold}%（性能退化）</li>
                <li>📊 正值表示DEFAULT比UCX慢，负值表示DEFAULT比UCX快</li>
            </ul>
        </div>
        
        <div class="summary">
            <div class="summary-box total">
                <h2>{total}</h2>
                <p>总测试数</p>
            </div>
            <div class="summary-box pass">
                <h2>{passed}</h2>
                <p>通过</p>
            </div>
            <div class="summary-box fail">
                <h2>{failed}</h2>
                <p>性能退化</p>
            </div>
        </div>
        
        <table>
            <thead>
                <tr>
                    <th style="width: 22%;">DEFAULT ({default_backend})</th>
                    <th style="width: 22%;">UCX</th>
                    <th style="width: 10%;">指标类型</th>
                    <th style="width: 10%;">DEFAULT值</th>
                    <th style="width: 10%;">UCX值</th>
                    <th style="width: 8%;">差异(%)</th>
                    <th style="width: 10%;">退化/提升(%)</th>
                    <th style="width: 8%;">状态</th>
                </tr>
            </thead>
            <tbody>
                {table_rows}
            </tbody>
        </table>
        
        <div class="footer">
            <p>生成时间: {timestamp}</p>
        </div>
    </div>
</body>
</html>
"""
    
    # 生成表格行
    table_rows = []
    for _, row in result_df.iterrows():
        status_class = 'status-pass' if row['status'] == 'Pass' else 'status-fail'
        
        # 格式化差异百分比
        if pd.notna(row['diff_pct']):
            diff_str = f"{row['diff_pct']:.2f}%"
        else:
            diff_str = 'N/A'
        
        # 格式化退化/提升百分比
        if pd.notna(row['regression_pct']):
            if row['regression_pct'] > 0:
                # 正值：DEFAULT比UCX慢（退化）
                regression_str = f"+{row['regression_pct']:.2f}%"
                regression_class = 'regression'
            else:
                # 负值：DEFAULT比UCX快（提升）
                regression_str = f"{row['regression_pct']:.2f}%"
                regression_class = 'improvement'
        else:
            regression_str = 'N/A'
            regression_class = 'neutral'
        
        # 格式化数值
        default_val = f"{row['default_value']:.2f}" if pd.notna(row['default_value']) else 'N/A'
        ucx_val = f"{row['ucx_value']:.2f}" if pd.notna(row['ucx_value']) else 'N/A'
        
        row_html = f"""
                <tr>
                    <td class="test-name">{row['test_case_name_default']}</td>
                    <td class="test-name">{row['test_case_name_ucx']}</td>
                    <td><span class="metric-type">{row['metric_type']}</span></td>
                    <td>{default_val}</td>
                    <td>{ucx_val}</td>
                    <td>{diff_str}</td>
                    <td class="{regression_class}">{regression_str}</td>
                    <td><span class="{status_class}">{row['status']}</span></td>
                </tr>
        """
        table_rows.append(row_html)
    
    # 填充模板
    from datetime import datetime
    html_content = html_template.format(
        default_backend=default_backend,
        threshold=threshold,
        total=total,
        passed=passed,
        failed=failed,
        table_rows=''.join(table_rows),
        timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    )
    
    # 写入文件
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)


def main():
    parser = argparse.ArgumentParser(
        description='比较DEFAULT backend和UCX的性能测试结果，只关注DEFAULT比UCX慢的情况'
    )
    parser.add_argument(
        '--csv-path',
        type=str,
        required=True,
        help='性能测试结果CSV文件路径'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=5.0,
        help='性能差异阈值（百分比），默认5.0%%. 只在DEFAULT比UCX慢超过此阈值时标记为Fail'
    )
    parser.add_argument(
        '--default-backend',
        type=str,
        default='NIXL',
        help='DEFAULT backend名称（默认NIXL，将来可能切换为其他backend）'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='输出CSV文件路径（可选，默认打印到stdout）'
    )
    parser.add_argument(
        '--html',
        type=str,
        help='输出HTML报告文件路径（可选）'
    )
    
    args = parser.parse_args()
    
    # 执行比较
    result_df = compare_backends(args.csv_path, args.threshold, args.default_backend)
    
    # 输出CSV结果
    if args.output:
        result_df.to_csv(args.output, index=False)
        print(f"CSV结果已保存到: {args.output}")
    else:
        print(result_df.to_string(index=False))
    
    # 输出HTML报告
    if args.html:
        generate_html_report(result_df, args.threshold, args.default_backend, args.html)
        print(f"HTML报告已保存到: {args.html}")
    
    # 统计信息
    total = len(result_df)
    failed = len(result_df[result_df['status'] == 'Fail'])
    passed = total - failed
    
    print(f"\n============= 统计信息 =============")
    print(f"DEFAULT Backend: {args.default_backend}")
    print(f"对比Backend: UCX")
    print(f"阈值: {args.threshold}%")
    print(f"-----------------------------------")
    print(f"总计: {total}")
    print(f"通过: {passed} (DEFAULT性能正常)")
    print(f"失败: {failed} (DEFAULT比UCX慢超过{args.threshold}%)")
    print(f"===================================\n")    
    sys.exit(1 if failed > 0 else 0)

if __name__ == '__main__':
    main()

