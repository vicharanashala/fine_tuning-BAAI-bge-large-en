#!/bin/bash

# Configuration
MODEL_PATH="/home/models/gemma4-e4b"
ENV_ACTIVATE="/home/kshitij/Kshitij/embedding-model-fine-tuning/.venv_vllm/bin/activate"

echo "--- Starting Dual vLLM Cluster ---"

# 1. Launch Worker 0 (GPU 0)
echo "Launching Worker 0 on Port 8000 (GPU 0)..."
CUDA_VISIBLE_DEVICES=0 nohup python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --port 8000 \
    --gpu-memory-utilization 0.4 \
    --trust-remote_code \
    --max-model-len 4096 > /home/kshitij/vllm_worker0.log 2>&1 &

# 2. Launch Worker 1 (GPU 1)
echo "Launching Worker 1 on Port 8001 (GPU 1)..."
CUDA_VISIBLE_DEVICES=1 nohup python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --port 8001 \
    --gpu-memory-utilization 0.4 \
    --trust-remote_code \
    --max-model-len 4096 > /home/kshitij/vllm_worker1.log 2>&1 &

echo "Dual Workers launched. Waiting 30s for initialization..."
sleep 30

# 3. Quick Health Check
echo "Checking Worker Health..."
curl -s localhost:8000/v1/models | grep -q "gemma" && echo "Worker 0: READY" || echo "Worker 0: INITIALIZING..."
curl -s localhost:8001/v1/models | grep -q "gemma" && echo "Worker 1: READY" || echo "Worker 1: INITIALIZING..."
