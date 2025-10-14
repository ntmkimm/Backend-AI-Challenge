#!/bin/bash 
#* shebang
MIN_MEMORY=000

check_gpu_memory() {
    nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | while IFS=',' read -r index memory; do
        # Trim whitespace from values
        index=$(echo "$index" | tr -d '[:space:]')
        memory=$(echo "$memory" | tr -d '[:space:]')
        echo "$index:$memory"
    done
}

run_uvicorn() {
    local gpu="$1"
    export CUDA_VISIBLE_DEVICES="$gpu"
    echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

    while true; do
        echo "Starting uvicorn server..."
        # uvicorn app:app --reload --port 5731 --host=0.0.0.0 --lifespan=auto --workers 1 \
        #     --limit-concurrency 1000 --backlog 2048 --http httptools --timeout-keep-alive 1000 \
        # --reload-exclude 'dependencies/*' --reload-exclude 'services/*' --reload-exclude 'core/*' --reload-exclude 'config/*'
        gunicorn app:app -w 1 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:5731 --timeout 1200 --keep-alive 1000 
        echo "Server crashed or exited. Restarting in 2s..."
        sleep 2
    done
}

echo "Searching for a suitable GPU..."
echo "Checking available GPU memory..."
while true; do
    available_gpu=""
    while IFS=':' read -r index memory; do
        # Trim whitespace from values again to be safe
        index=$(echo "$index" | tr -d '[:space:]')
        memory=$(echo "$memory" | tr -d '[:space:]')
        
        if [ -n "$index" ] && [ -n "$memory" ]; then
            # if { [ "$index" = "6" ] || [ "$index" = "7" ]; } && [ "$memory" -ge "$MIN_MEMORY" ]; then
            if { [ "$index" = "6" ]; } && [ "$memory" -ge "$MIN_MEMORY" ]; then
                # available_gpu="$index"
                available_gpu="6"
                break  # Chỉ lấy GPU 6 hoặc 7 có đủ bộ nhớ
            fi
        fi
    done < <(check_gpu_memory)

    if [ -n "$available_gpu" ]; then
        echo "Found suitable GPU: $available_gpu with free memory >= ${MIN_MEMORY}MB"
        run_uvicorn "$available_gpu"
        break
    else
        echo "No GPU 6 or 7 with ${MIN_MEMORY}MB free memory found. Waiting 2 seconds..."
        sleep 2
    fi
done

echo "Uvicorn server is running"
