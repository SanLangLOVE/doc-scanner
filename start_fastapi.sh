#!/bin/bash

PORT=8699
HOST=0.0.0.0
LOG=fastapi.log

# 获取本机所有IP
IPS=$(ifconfig | grep 'inet ' | awk '{print $2}' | grep -v 127.0.0.1)

echo "[INFO] 启动 FastAPI (Uvicorn) 服务..."
echo "[INFO] 本地访问: http://localhost:$PORT"
for ip in $IPS; do
  echo "[INFO] 局域网访问: http://$ip:$PORT"
done
echo "[INFO] 日志输出到: $LOG"

uvicorn docscanner_api:app --host $HOST --port $PORT --reload --log-level debug | tee $LOG 