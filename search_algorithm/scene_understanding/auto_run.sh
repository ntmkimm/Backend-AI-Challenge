#!/bin/bash 
#* shebang
MIN_MEMORY=3000
SKIP_GPU_LIST=(0 3 4 5)
MAX_GPUS=2
cnt=0
used_gpus=()

check_gpu_memory() {
    nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | while IFS=',' read -r index memory; do
        index=$(echo "$index" | tr -d '[:space:]')
        memory=$(echo "$memory" | tr -d '[:space:]')
        echo "$index:$memory"
    done
}

is_gpu_skipped() {
    local gpu="$1"
    for skip in "${SKIP_GPU_LIST[@]}"; do
        if [ "$gpu" = "$skip" ]; then
            return 0  # true: nên bỏ qua
        fi
    done
    return 1  # false: không nằm trong list skip
}

is_gpu_used() {
    local gpu="$1"
    for used in "${used_gpus[@]}"; do
        if [ "$gpu" = "$used" ]; then
            return 0  # true: đã dùng
        fi
    done
    return 1  # false: chưa dùng
}

run_base_search() {
    local gpu="$1"
    export CUDA_VISIBLE_DEVICES="$gpu"
    echo "🎬 Running on GPU $gpu (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)..."
    python scene_en.py &
}

echo "🚀 Starting job dispatcher for up to $MAX_GPUS GPUs..."

while [ "$cnt" -lt "$MAX_GPUS" ]; do
    available_gpu=""
    while IFS=':' read -r index memory; do
        index=$(echo "$index" | tr -d '[:space:]')
        memory=$(echo "$memory" | tr -d '[:space:]')

        if [ -n "$index" ] && [ -n "$memory" ]; then
            is_gpu_skipped "$index" && continue
            is_gpu_used "$index" && continue
            if [ "$memory" -ge "$MIN_MEMORY" ]; then
                available_gpu="$index"
                break  # lấy GPU đầu tiên chưa dùng và đủ RAM
            fi
        fi
    done < <(check_gpu_memory)

    if [ -n "$available_gpu" ]; then
        echo "✅ Found GPU $available_gpu with $memory MB free"
        used_gpus+=("$available_gpu")  # đánh dấu đã dùng
        run_base_search "$available_gpu"
        cnt=$((cnt + 1))
        echo "🧠 Jobs started: $cnt/$MAX_GPUS"
    else
        echo "⏳ No suitable GPU found yet. Retrying in 2s..."
        sleep 2
    fi
done

echo "✅ All $MAX_GPUS jobs dispatched. Waiting for them to finish..."
wait
echo "🏁 All jobs completed."
