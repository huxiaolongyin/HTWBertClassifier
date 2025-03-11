#!/bin/bash

# 创建logs目录（如果不存在）
mkdir -p logs

# 获取当前日期时间作为日志文件名的一部分
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/api_${TIMESTAMP}.log"

echo "Starting BERT Classification API on http://0.0.0.0:6565"
echo "Logs will be saved to ${LOG_FILE}"

# 激活虚拟环境
source .venv/bin/activate

# 启动服务并将输出重定向到日志文件
uvicorn api.main:app --host 0.0.0.0 --port 6565 --log-level info >> ${LOG_FILE} 2>&1