#!/bin/bash
# python3 /home/scratch.fredw_sw/trt-llm-github-3/TensorRT-LLM/examples/scaffolding/contrib/DeepResearch/run_deep_research.py

# 默认配置
ENABLE_DETERMINISTIC=1
ENABLE_DEBUG=0
ENABLE_AGENT_HIERARCHY=0
SET_UNIQUE_ID_ZERO=0

# Benchmark enable flags
ENABLE_NORMAL_AGENT=0
ENABLE_CHATBOT=0
ENABLE_MULTIROUND_CHATBOT=0
ENABLE_BURST_AGENT=0

MODEL=gpt-oss-20b
MODEL_DIR=/home/scratch.trt_llm_data/llm-models/gpt_oss/gpt-oss-20b
ENABLE_STATISTICS=0
ENABLE_QUERY_COLLECTOR=0

# Normal agent parameters
AGENT_PROMPT_NUM=128
NORMAL_AGENT_CONCURRENCY=40

# Chatbot parameters
CHAT_PROMPT_NUM=20
CHAT_CONCURRENCY=40
MAX_TOKENS_CHAT=16384

# Burst agent parameters
BURST_DELAY=30.0
BURST_PROMPT_NUM=50
BURST_AGENT_CONCURRENCY=100

# Multi-round Chatbot parameters
MULTI_ROUND_ROUNDS=3

# 帮助信息
show_help() {
    cat << EOF
用法: $0 [选项]

所有开关类参数都支持 0/1 格式，方便修改：

Benchmark 启用选项:
    --enable_normal_agent [0|1]        启用正常 agent benchmark (默认: 0)
    --enable_chatbot [0|1]             启用 chatbot benchmark (默认: 0)
    --enable_multiround_chatbot [0|1]  启用 multi-round chatbot benchmark (默认: 0)
    --enable_burst_agent [0|1]         启用突发 agent benchmark (默认: 0)

通用选项:
    --model MODEL                      模型名称 (默认: gpt-oss-20b)
    --enable_statistics [0|1]          启用统计信息 (默认: 0)
    --enable_query_collector [0|1]     启用 query collector (默认: 0)
    --enable_deterministic [0|1]       启用确定性模式 (默认: 1)
    --enable_debug [0|1]               启用 debug 模式 (默认: 0)
    --enable_agent_hierarchy [0|1]     启用 agent hierarchy (默认: 0)
    --set_unique_id_zero [0|1]         设置 unique_id 为 0 (默认: 0)

Normal Agent 参数:
    --agent_prompt_num "NUM1 NUM2..."  agent prompt 数量，支持多个值 (默认: 128)
    --normal_agent_concurrency "NUM1 NUM2..."  正常 agent 并发数，支持多个值 (默认: 40)

Chatbot 参数:
    --chat_prompt_num NUM              chatbot prompt 数量 (默认: 20)
    --chat_concurrency NUM             chatbot 并发数 (默认: 40)
    --max_tokens_chat NUM              chatbot 最大生成 token 数 (默认: 1024)

Burst Agent 参数:
    --burst_delay SECONDS              突发流量延迟秒数 (默认: 30.0)
    --burst_prompt_num NUM             突发流量 prompt 数量 (默认: 50)
    --burst_agent_concurrency NUM      突发流量并发数 (默认: 100)

    -h, --help                         显示此帮助信息

Multi-round Chatbot 参数:
    --multi_round_rounds NUM           多轮对话轮数 (默认: 3)

日志目录:
    启用 agent hierarchy + set_unique_id_zero: /home/scratch.docao_gpu/work/log/enable_agent_tree_id_0
    启用 agent hierarchy: /home/scratch.docao_gpu/work/log/enable_agent_tree
    禁用 agent hierarchy: /home/scratch.docao_gpu/work/log/disable_agent_tree

示例:
    # 只运行正常 agent
    $0 --enable_normal_agent 1

    # 运行正常 agent + chatbot
    $0 --enable_normal_agent 1 --enable_chatbot 1

    # 运行正常 agent + 突发 agent（模拟压力测试）
    $0 --enable_normal_agent 1 --enable_burst_agent 1 --burst_delay 30 --burst_prompt_num 50

    # 三种都运行
    $0 --enable_normal_agent 1 --enable_chatbot 1 --enable_burst_agent 1

    # 遍历多个 agent_prompt_num 和 concurrency
    $0 --enable_normal_agent 1 --agent_prompt_num "20 40 60 80" --normal_agent_concurrency "20 40"

    # 带 agent hierarchy
    $0 --enable_normal_agent 1 --enable_agent_hierarchy 1

EOF
    exit 0
}

# 辅助函数：验证 0/1 参数
validate_bool_param() {
    local param_name="$1"
    local value="$2"
    if [[ ! "$value" =~ ^[01]$ ]]; then
        echo "错误: $param_name 需要参数 0 或 1，收到: '$value'"
        exit 1
    fi
}

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        # Benchmark enable flags
        --enable_normal_agent)
            validate_bool_param "--enable_normal_agent" "$2"
            ENABLE_NORMAL_AGENT="$2"
            shift 2
            ;;
        --enable_chatbot)
            validate_bool_param "--enable_chatbot" "$2"
            ENABLE_CHATBOT="$2"
            shift 2
            ;;
        --enable_burst_agent)
            validate_bool_param "--enable_burst_agent" "$2"
            ENABLE_BURST_AGENT="$2"
            shift 2
            ;;
        --enable_multiround_chatbot)
            validate_bool_param "--enable_multiround_chatbot" "$2"
            ENABLE_MULTIROUND_CHATBOT="$2"
            shift 2
            ;;
        # Common options
        --model)
            MODEL="$2"
            shift 2
            ;;
        --model_dir)
            MODEL_DIR="$2"
            shift 2
            ;;
        --enable_statistics)
            validate_bool_param "--enable_statistics" "$2"
            ENABLE_STATISTICS="$2"
            shift 2
            ;;
        --enable_query_collector)
            validate_bool_param "--enable_query_collector" "$2"
            ENABLE_QUERY_COLLECTOR="$2"
            shift 2
            ;;
        --enable_deterministic)
            validate_bool_param "--enable_deterministic" "$2"
            ENABLE_DETERMINISTIC="$2"
            shift 2
            ;;
        --enable_debug)
            validate_bool_param "--enable_debug" "$2"
            ENABLE_DEBUG="$2"
            shift 2
            ;;
        --enable_agent_hierarchy)
            validate_bool_param "--enable_agent_hierarchy" "$2"
            ENABLE_AGENT_HIERARCHY="$2"
            shift 2
            ;;
        --set_unique_id_zero)
            validate_bool_param "--set_unique_id_zero" "$2"
            SET_UNIQUE_ID_ZERO="$2"
            shift 2
            ;;
        # Normal agent parameters
        --agent_prompt_num)
            AGENT_PROMPT_NUM="$2"
            shift 2
            ;;
        --normal_agent_concurrency)
            NORMAL_AGENT_CONCURRENCY="$2"
            shift 2
            ;;
        # Chatbot parameters
        --chat_prompt_num)
            CHAT_PROMPT_NUM="$2"
            shift 2
            ;;
        --chat_concurrency)
            CHAT_CONCURRENCY="$2"
            shift 2
            ;;
        --max_tokens_chat)
            MAX_TOKENS_CHAT="$2"
            shift 2
            ;;
        # Burst agent parameters
        --burst_delay)
            BURST_DELAY="$2"
            shift 2
            ;;
        --burst_prompt_num)
            BURST_PROMPT_NUM="$2"
            shift 2
            ;;
        --burst_agent_concurrency)
            BURST_AGENT_CONCURRENCY="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            ;;
        # Multi-round Chatbot parameters
        --multi_round_rounds)
            MULTI_ROUND_ROUNDS="$2"
            shift 2
            ;;
        *)
            echo "错误: 未知参数 '$1'"
            echo "使用 --help 查看帮助信息"
            exit 1
            ;;
    esac
done

# 检查是否至少启用了一个 benchmark
if [ "$ENABLE_NORMAL_AGENT" = "0" ] && [ "$ENABLE_CHATBOT" = "0" ] && [ "$ENABLE_BURST_AGENT" = "0" ] && [ "$ENABLE_MULTIROUND_CHATBOT" = "0" ]; then
    echo "错误: 至少需要启用一个 benchmark"
    echo "使用 --enable_normal_agent, --enable_chatbot, --enable_multiround_chatbot, 或 --enable_burst_agent"
    exit 1
fi

# 根据参数设置环境变量
if [ "$ENABLE_DETERMINISTIC" = "1" ]; then
    export SCAFFOLDING_DETERMINISTIC=1
fi

if [ "$ENABLE_DEBUG" = "1" ]; then
    export DEBUG=1
fi

export DEBUG_AGENT_HIERARCHY=1
if [ "$ENABLE_AGENT_HIERARCHY" = "1" ]; then
    export ENABLE_SUB_REQUEST_MARKER=1
fi

if [ "$SET_UNIQUE_ID_ZERO" = "1" ]; then
    export SET_UNIQUE_ID_ZERO=1
fi

# 根据 agent_hierarchy 和 set_unique_id_zero 设置确定日志目录
if [ "$ENABLE_AGENT_HIERARCHY" = "1" ] && [ "$SET_UNIQUE_ID_ZERO" = "1" ]; then
    LOG_BASE_DIR="/home/scratch.fredw_sw/work/log/enable_agent_tree_id_0"
elif [ "$ENABLE_AGENT_HIERARCHY" = "1" ]; then
    LOG_BASE_DIR="/home/scratch.fredw_sw/work/log/enable_agent_tree"
else
    LOG_BASE_DIR="/home/scratch.fredw_sw/work/log/disable_agent_tree"
fi

echo "========================================"
echo "配置信息:"
echo "  模型: $MODEL"
echo ""
echo "  Benchmark 启用:"
echo "    Normal Agent: $([ "$ENABLE_NORMAL_AGENT" = "1" ] && echo "启用" || echo "禁用")"
echo "    Chatbot: $([ "$ENABLE_CHATBOT" = "1" ] && echo "启用" || echo "禁用")"
echo "    Multi-round Chatbot: $([ "$ENABLE_MULTIROUND_CHATBOT" = "1" ] && echo "启用" || echo "禁用")"
echo "    Burst Agent: $([ "$ENABLE_BURST_AGENT" = "1" ] && echo "启用" || echo "禁用")"
echo ""
if [ "$ENABLE_NORMAL_AGENT" = "1" ]; then
    echo "  Normal Agent 参数:"
    echo "    Prompt 数量: $AGENT_PROMPT_NUM ($(echo $AGENT_PROMPT_NUM | wc -w) 个值)"
    echo "    并发数量: $NORMAL_AGENT_CONCURRENCY ($(echo $NORMAL_AGENT_CONCURRENCY | wc -w) 个值)"
fi
if [ "$ENABLE_CHATBOT" = "1" ]; then
    echo "  Chatbot 参数:"
    echo "    Prompt 数量: $CHAT_PROMPT_NUM"
    echo "    并发数量: $CHAT_CONCURRENCY"
    echo "    最大生成 token: $MAX_TOKENS_CHAT"
fi
if [ "$ENABLE_MULTIROUND_CHATBOT" = "1" ]; then
    echo "  Multi-round Chatbot 参数:"
    echo "    Prompt 数量: $CHAT_PROMPT_NUM"
    echo "    并发数量: $CHAT_CONCURRENCY"
fi
if [ "$ENABLE_BURST_AGENT" = "1" ]; then
    echo "  Burst Agent 参数:"
    echo "    延迟: ${BURST_DELAY}s"
    echo "    Prompt 数量: $BURST_PROMPT_NUM"
    echo "    并发数量: $BURST_AGENT_CONCURRENCY"
fi
if [ "$ENABLE_MULTIROUND_CHATBOT" = "1" ]; then
    echo "  Multi-round Chatbot 参数:"
    echo "    轮数: $MULTI_ROUND_ROUNDS"
fi
echo ""
echo "  通用设置:"
echo "    确定性模式: $([ "$ENABLE_DETERMINISTIC" = "1" ] && echo "启用" || echo "禁用")"
echo "    Debug 模式: $([ "$ENABLE_DEBUG" = "1" ] && echo "启用" || echo "禁用")"
echo "    Agent Hierarchy: $([ "$ENABLE_AGENT_HIERARCHY" = "1" ] && echo "启用" || echo "禁用")"
echo "    Set Unique ID Zero: $([ "$SET_UNIQUE_ID_ZERO" = "1" ] && echo "启用" || echo "禁用")"
echo "    Statistics: $([ "$ENABLE_STATISTICS" = "1" ] && echo "启用" || echo "禁用")"
echo "    Query Collector: $([ "$ENABLE_QUERY_COLLECTOR" = "1" ] && echo "启用" || echo "禁用")"
echo "  日志目录: ${LOG_BASE_DIR}"
echo "========================================"
echo ""

# 创建 log 目录（如果不存在）
mkdir -p "${LOG_BASE_DIR}"

# 构建 benchmark 模式字符串（用于日志文件名）
BENCHMARK_MODE=""
[ "$ENABLE_NORMAL_AGENT" = "1" ] && BENCHMARK_MODE="${BENCHMARK_MODE}normal_"
[ "$ENABLE_BURST_AGENT" = "1" ] && BENCHMARK_MODE="${BENCHMARK_MODE}burst_"
[ "$ENABLE_CHATBOT" = "1" ] && BENCHMARK_MODE="${BENCHMARK_MODE}chat_"
[ "$ENABLE_MULTIROUND_CHATBOT" = "1" ] && BENCHMARK_MODE="${BENCHMARK_MODE}multiround_chat_"
BENCHMARK_MODE="${BENCHMARK_MODE%_}"  # 移除末尾的下划线

# 遍历不同的 agent_prompt_num 和 normal_agent_concurrency 值（两重循环）
for agent_num in ${AGENT_PROMPT_NUM}; do
    for concurrency in ${NORMAL_AGENT_CONCURRENCY}; do
        echo "=========================================="
        echo "开始运行 benchmarks: ${BENCHMARK_MODE}"
        echo "  agent_prompt_num=${agent_num}, normal_agent_concurrency=${concurrency}"
        echo "=========================================="
        
        # 构建日志文件名：n{0/1}_b{0/1}_c{0/1}_参数配置
        log_filename="n${ENABLE_NORMAL_AGENT}_b${ENABLE_BURST_AGENT}_c${ENABLE_CHATBOT}_m${ENABLE_MULTIROUND_CHATBOT}"
        log_filename="${log_filename}_normal_p${agent_num}_c${concurrency}"
        log_filename="${log_filename}_burst_d${BURST_DELAY}_p${BURST_PROMPT_NUM}_c${BURST_AGENT_CONCURRENCY}"
        log_filename="${log_filename}_chat_p${CHAT_PROMPT_NUM}_c${CHAT_CONCURRENCY}_t${MAX_TOKENS_CHAT}"
        log_filename="${log_filename}_inflight_64"
        log_file="${LOG_BASE_DIR}/${log_filename}.txt"
        
        # 构建基本命令
        cmd="python3 /home/scratch.fredw_sw/trt-llm-github-3/TensorRT-LLM/examples/scaffolding/benchmark_agent_chat.py"
        cmd="${cmd} --model \"${MODEL}\""
        
        # 添加 benchmark enable flags
        if [ "$ENABLE_NORMAL_AGENT" = "1" ]; then
            cmd="${cmd} --enable_normal_agent"
            cmd="${cmd} --agent_prompt_num ${agent_num}"
            cmd="${cmd} --normal_agent_concurrency ${concurrency}"
        fi
        
        if [ "$ENABLE_CHATBOT" = "1" ]; then
            cmd="${cmd} --enable_chatbot"
            cmd="${cmd} --chat_prompt_num ${CHAT_PROMPT_NUM}"
            cmd="${cmd} --chat_concurrency ${CHAT_CONCURRENCY}"
            cmd="${cmd} --max_tokens_chat ${MAX_TOKENS_CHAT}"
        fi

        if [ "$ENABLE_MULTIROUND_CHATBOT" = "1" ]; then
            cmd="${cmd} --enable_multiround_chatbot"
            cmd="${cmd} --model_dir ${MODEL_DIR}"
            cmd="${cmd} --chat_multiround_rounds ${MULTI_ROUND_ROUNDS}"
            cmd="${cmd} --chat_prompt_num ${CHAT_PROMPT_NUM}"
            cmd="${cmd} --chat_concurrency ${CHAT_CONCURRENCY}"
        fi
        
        if [ "$ENABLE_BURST_AGENT" = "1" ]; then
            cmd="${cmd} --enable_burst_agent"
            cmd="${cmd} --burst_delay ${BURST_DELAY}"
            cmd="${cmd} --burst_prompt_num ${BURST_PROMPT_NUM}"
            cmd="${cmd} --burst_agent_concurrency ${BURST_AGENT_CONCURRENCY}"
        fi
        
        # 添加通用选项
        if [ "$ENABLE_STATISTICS" = "1" ]; then
            cmd="${cmd} --enable_statistics"
        fi
        
        if [ "$ENABLE_QUERY_COLLECTOR" = "1" ]; then
            cmd="${cmd} --enable_query_collector"
        fi
        
        # 执行命令并输出到终端和日志文件
        cmd="${cmd} 2>&1 | tee \"${log_file}\""
        echo "执行命令: $cmd"
        eval "$cmd"
        
        # 检查执行状态
        if [ $? -eq 0 ]; then
            echo "✓ ${BENCHMARK_MODE}, agent_prompt_num=${agent_num}, concurrency=${concurrency} 完成"
            echo "  日志保存到: ${log_file}"
        else
            echo "✗ ${BENCHMARK_MODE}, agent_prompt_num=${agent_num}, concurrency=${concurrency} 执行失败！"
        fi
        
        echo ""
    done
done

echo "=========================================="
echo "所有测试完成！"
echo "日志文件保存在 ${LOG_BASE_DIR} 目录下："
ls -lh ${LOG_BASE_DIR}/*.txt 2>/dev/null || echo "没有找到日志文件"
echo "=========================================="


# 示例命令:
# ./script/run_deep_research.sh --enable_agent_hierarchy 1 --enable_statistics 1 --enable_normal_agent 1 --agent_prompt_num "1" --normal_agent_concurrency "20" 2>&1 | tee client_output.txt
# ./script/run_deep_research.sh --enable_agent_hierarchy 1 --enable_statistics 1 --enable_normal_agent 1 --agent_prompt_num "1" --normal_agent_concurrency "20" --enable_burst_agent 1 --burst_delay 120  --burst_prompt_num 10 --burst_agent_concurrency 10 2>&1 | tee client_output.txt
# ./script/run_deep_research.sh --enable_agent_hierarchy 1 --enable_statistics 1 --enable_normal_agent 1 --agent_prompt_num "50" --normal_agent_concurrency "20" --enable_burst_agent 1 --burst_delay 240  --burst_prompt_num 10 --burst_agent_concurrency 10 --enable_chatbot 1 --chat_prompt_num 5 --chat_concurrency 1 2>&1 | tee client_output.txt
# ./script/run_deep_research.sh --model gpt-oss-120b --enable_agent_hierarchy 0 --set_unique_id_zero 1 --enable_statistics 1 --enable_normal_agent 1 --agent_prompt_num "100" --normal_agent_concurrency "32" --enable_burst_agent 1 --burst_delay 300  --burst_prompt_num 32 --burst_agent_concurrency 32 --enable_chatbot 1 --chat_prompt_num 100 --chat_concurrency 32 --max_tokens_chat 16384 2>&1 | tee client_output.txt

