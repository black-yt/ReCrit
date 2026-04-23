#!/bin/bash
set -euo pipefail

# User configuration.
export LLM_API_KEY="${LLM_API_KEY:-YOUR_LLM_API_KEY}"
export LLM_BASE_URL="${LLM_BASE_URL:-https://your-judge-endpoint/v1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

MODEL_PATH="${MODEL_PATH:-/path/to/model}"
TRAIN_DATASET="${TRAIN_DATASET:-/path/to/train.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-output}"

mkdir -p "${OUTPUT_DIR}"

IFS=',' read -ra _GPUS <<< "${CUDA_VISIBLE_DEVICES}"
WORLD_SIZE=${#_GPUS[@]}

TRAIN_ARGS=(
    --model_path          "${MODEL_PATH}"
    --train_dataset       "${TRAIN_DATASET}"
    --output_dir          "${OUTPUT_DIR}"
    --judge_mode          both
    --judge_model         gemini-3-flash-preview-nothinking
    --num_generations     8
    --num_turns           2
    --completion_ratio    0.75
    --per_device_train_batch_size 2
    --learning_rate       2e-6
    --max_new_tokens      4096
    --max_seq_length      8192
    --num_train_epochs    3
    --temperature         1.0
    --vllm_gpu_memory_utilization 0.22
    --vllm_max_model_len  65536
    --repetition_n_grams  8
    --soft_max_length     4096
    --soft_cache_length   1024
    --w_correction        1.0
    --w_robustness        0.6
    --w_sycophancy        1.0
    --w_boundary          0.1
    --epsilon             0.2
    --kl_beta             0.04
    --warmup_ratio        0.05
    --gradient_accumulation_steps 4
    --think_fmt_weight    0.2
    --repetition_weight   0.2
    --overlong_weight     0.2
    --train_micro_batch_size 1
)

if [[ "${WORLD_SIZE}" -eq 1 ]]; then
    echo "Single-GPU mode (GPU: ${CUDA_VISIBLE_DEVICES})"
    RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 python -m train "${TRAIN_ARGS[@]}"
else
    echo "Multi-GPU mode (WORLD_SIZE=${WORLD_SIZE}, GPUs: ${CUDA_VISIBLE_DEVICES})"
    export MASTER_ADDR=127.0.0.1
    export MASTER_PORT="${MASTER_PORT:-29500}"

    PIDS=()
    for ((rank=0; rank<WORLD_SIZE; rank++)); do
        CUDA_VISIBLE_DEVICES=${_GPUS[$rank]} \
        RANK=${rank} LOCAL_RANK=0 WORLD_SIZE=${WORLD_SIZE} \
            python -m train "${TRAIN_ARGS[@]}" &
        PIDS+=($!)
    done
    wait "${PIDS[@]}"
fi
