set -eu
slot="$1"
shift
node_name="$(hostname -s)"
gpu_uuid="$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i "$slot")"
mkdir -p "$HOME/.gpu-leases" "$HOME/.research-allocations" outputs
exec 9>"$HOME/.gpu-leases/$node_name-$gpu_uuid.lock"
flock -n 9
exec 8>"$HOME/.research-allocations/$node_name-$gpu_uuid.lock"
flock -n 8
used_memory="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu_uuid")"
test "$used_memory" -lt 256
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$gpu_uuid")"
if scontrol show reservation -o 2>/dev/null | grep -q 'Nodes=.*avi-gn-fsk34'; then
    exit 1
fi
export PATH="$PWD/.venv/bin:/usr/local/cuda-13.1/bin:$PATH"
export CUDA_HOME=/usr/local/cuda-13.1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$slot"
actual_uuid="$(.venv/bin/python -c 'import torch; print("GPU-" + str(torch.cuda.get_device_properties(0).uuid))')"
test "$actual_uuid" = "$gpu_uuid"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export MAX_JOBS=4
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export VLLM_CACHE_ROOT="$PWD/.work/cache/gpu-$slot/vllm"
export TORCHINDUCTOR_CACHE_DIR="$PWD/.work/cache/gpu-$slot/inductor"
export TRITON_CACHE_DIR="$PWD/.work/cache/gpu-$slot/triton"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((29600+slot))
receipt="outputs/resource-$gpu_uuid.jsonl"
start_time="$(date +%s)"
printf '{"state":"RUNNING","host":"%s","gpu":"%s","start_epoch":%s,"pid":%s,"limit_seconds":10800}\n' "$node_name" "$gpu_uuid" "$start_time" "$$" >> "$receipt"
trap 'result=$?; end_time=$(date +%s); printf "{\"state\":\"RELEASED\",\"end_epoch\":%s,\"elapsed_seconds\":%s,\"exit_code\":%s}\n" "$end_time" "$((end_time-start_time))" "$result" >> "$receipt"' EXIT
timeout --signal=TERM 10800 "$@"
