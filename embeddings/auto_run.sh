#!/bin/bash

# ==== CONFIG ====
MIN_MEMORY=3000                      # MB bộ nhớ trống yêu cầu mỗi GPU
NUM_GPUS_REQUIRED=1                # Số GPU bạn muốn dùng: 2 hoặc 3
SKIP_GPU="7"                        # GPU cần bỏ qua, phân tách bằng dấu phẩy: "1,7"
PYTHON_SCRIPT="upload_milvus.py"   # Tên file Python (không cần "python ..." ở đây)
# =================

IFS=',' read -ra SKIP_ARRAY <<< "$SKIP_GPU"

# Kiểm tra bộ nhớ các GPU
check_gpu_memory() {
    nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | while IFS=',' read -r index memory; do
        echo "$index:$memory"
    done
}

# Chạy script Python với CUDA_VISIBLE_DEVICES
run_python_script() {
    local gpus_csv="$1"
    export CUDA_VISIBLE_DEVICES="$gpus_csv"
    echo "Using GPUs: $CUDA_VISIBLE_DEVICES"
    echo "Running: python $PYTHON_SCRIPT"
    python "$PYTHON_SCRIPT"
}

# Vòng lặp chính
echo "Searching for $NUM_GPUS_REQUIRED suitable GPU(s)..."
while true; do
    suitable_gpus=()

    while IFS=':' read -r index memory; do
        skip=false
        for skip_index in "${SKIP_ARRAY[@]}"; do
            if [ "$index" -eq "$skip_index" ]; then
                skip=true
                break
            fi
        done

        if ! $skip && [ "$memory" -ge "$MIN_MEMORY" ]; then
            suitable_gpus+=("$index")
            if [ "${#suitable_gpus[@]}" -ge "$NUM_GPUS_REQUIRED" ]; then
                break
            fi
        fi
    done < <(check_gpu_memory)

    if [ "${#suitable_gpus[@]}" -ge "$NUM_GPUS_REQUIRED" ]; then
        gpu_list_csv=$(IFS=','; echo "${suitable_gpus[*]}")
        echo "Found GPUs: $gpu_list_csv"
        run_python_script "$gpu_list_csv"
        echo "Script finished."
        break
    else
        echo "Waiting for $NUM_GPUS_REQUIRED GPU(s) with >= ${MIN_MEMORY}MB free..."
        sleep 5
    fi
done
