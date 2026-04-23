#!/bin/bash
set -euo pipefail

# ============================================================
# User configuration section (modify the following variables as needed)
# ============================================================
export LLM_API_KEY="sk-xxxxx"
export LLM_BASE_URL="base-url"

# Visible GPU IDs. Use one value for single-GPU and comma-separated values for multi-GPU (for example: 0,1,2,3)
export CUDA_VISIBLE_DEVICES=0,1

MODEL_PATH="/path/to/model"
TRAIN_DATASET="/path/to/train.jsonl"
OUTPUT_DIR="output"
# ============================================================

mkdir -p "${OUTPUT_DIR}"

# Automatically derive the GPU count from CUDA_VISIBLE_DEVICES
IFS=',' read -ra _GPUS <<< "${CUDA_VISIBLE_DEVICES}"
WORLD_SIZE=${#_GPUS[@]}

TRAIN_ARGS=(
    --model_path          "${MODEL_PATH}"
    --train_dataset       "${TRAIN_DATASET}"
    --output_dir          "${OUTPUT_DIR}"
    --judge_mode          both              # close / open / both (the dataset must include a judge_mode field when using both)
    --judge_model         gemini-3-flash-preview-nothinking  # LLM model used by Judge evaluation
    # --add_format_prompt                   # Uncomment to append a format-requirements prompt when the dataset does not already provide one
    --num_generations     8
    --num_turns           2
    --completion_ratio    0.75              # Stopping threshold for asynchronous rollout (1.0 = synchronous mode)
    --per_device_train_batch_size 2         # Number of prompts per GPU (total rollout prompts = this value x GPU count)
    --learning_rate       2e-6
    --max_new_tokens      4096
    --max_seq_length      8192
    --num_train_epochs    3
    --temperature         1.0
    --vllm_gpu_memory_utilization 0.22  # Fraction of GPU memory reserved by vLLM; smaller values leave more memory for training
    --vllm_max_model_len  65536
    --repetition_n_grams  8
    --soft_max_length     4096
    --soft_cache_length   1024
    --w_correction        1.0
    --w_robustness        0.6
    --w_sycophancy        1.0
    --w_boundary          0.1
    --epsilon             0.2
    --kl_beta             0.04              # KL penalty coefficient (anchors the initial SFT model; 0 = disabled)
    --warmup_ratio        0.05
    --gradient_accumulation_steps 4   # Gradient accumulation steps (effective batch = per_device x num_gen x accum x GPU count)
    --think_fmt_weight    0.2         # Auxiliary-reward scaling weight (repetition/overlong/think_fmt all default to 0.2)
    --repetition_weight   0.2
    --overlong_weight     0.2
    --train_micro_batch_size 1          # Process only 1 sample per micro-step; this halves peak memory usage (useful when large models OOM)
)

if [[ "${WORLD_SIZE}" -eq 1 ]]; then
    echo "Single-GPU mode (GPU: ${CUDA_VISIBLE_DEVICES})"
    RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 python -m train "${TRAIN_ARGS[@]}"
else
    echo "Multi-GPU mode (WORLD_SIZE=${WORLD_SIZE}, GPUs: ${CUDA_VISIBLE_DEVICES})"
    export MASTER_ADDR=127.0.0.1
    export MASTER_PORT=29500

    # Each rank process only sees one GPU (CUDA_VISIBLE_DEVICES restricts it to a single card),
    # so both vLLM and the training model run on cuda:0, avoiding CUDA-context conflicts.
    # LOCAL_RANK is always 0 because each process only sees one GPU.
    PIDS=()
    for ((rank=0; rank<WORLD_SIZE; rank++)); do
        CUDA_VISIBLE_DEVICES=${_GPUS[$rank]} \
        RANK=${rank} LOCAL_RANK=0 WORLD_SIZE=${WORLD_SIZE} \
            python -m train "${TRAIN_ARGS[@]}" &
        PIDS+=($!)
    done
    wait "${PIDS[@]}"
fi
