#!/bin/bash
set -euo pipefail
: "${GLM53_NODE:?Set 22 for prefill, 23 or 24 for decode}"
: "${WORK_DIR:?Set an existing writable work directory}"
: "${ADAPTER_DIR:?Set adapter checkout path}"
: "${SGLANG_DIR:?Set the compatible official SGLang checkout path}"
: "${MODEL_DIR:?Set directory containing both GLM checkpoints}"
: "${HOST_IP:?Set this node service IP}"
: "${DECODE_MASTER_IP:?Set decode rank 0 IP}"
case "$GLM53_NODE" in 22|23|24) ;; *) exit 2 ;; esac
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PLUGIN_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)
SGLANG_PYTHON_DIR=$(python3 "$SCRIPT_DIR/verify_upstream.py" "$SGLANG_DIR")
export PYTHONPATH="$ADAPTER_DIR:$SGLANG_PYTHON_DIR:$PLUGIN_DIR"
export SGLANG_PLUGINS=hcu HCU_SLIMQUANT=0
export SGLANG_USE_FUSED_SILU_MUL_QUANT=1
unset W8A8_SUPPORT_METHODS
if [ "$GLM53_NODE" != 22 ]; then export W8A8_SUPPORT_METHODS=3; fi
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4
export HF_HOME="$WORK_DIR/cache/huggingface"
export SGLANG_PYSPY_DUMP_BEFORE_CRASH=0
export LD_LIBRARY_PATH=/opt/dtk/dcc/lib:/opt/dtk/.hyhal/rocm_smi/lib:/opt/dtk/hip/lib:/opt/dtk/lib:${LD_LIBRARY_PATH:-}
unset NCCL_TOPO_FILE
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-bond1}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-bond1}"
export NCCL_IB_HCA="${NCCL_IB_HCA:-shca_0,shca_1,shca_2,shca_3}"
export ROCSHMEM_ALLOWED_IBV_DEVICES="${ROCSHMEM_ALLOWED_IBV_DEVICES:-$NCCL_IB_HCA}"
export TRITON_CACHE_DIR="$WORK_DIR/cache/node$GLM53_NODE/triton"
export TORCH_EXTENSIONS_DIR="$WORK_DIR/cache/node$GLM53_NODE/torch_extensions"
export PYTHONPYCACHEPREFIX="$WORK_DIR/cache/node$GLM53_NODE/pycache"
unset GLM53_BOUNDARY_ARM GLM53_BOUNDARY_CONTROL GLM53_BOUNDARY_OUTPUT GLM53_BOUNDARY_ROLE
unset HCU_PLUGIN_SAMPLER_OBSERVER_CONFIG HCU_PLUGIN_LAYER0_OBSERVER_CONFIG
cd "$WORK_DIR"
ulimit -c 0
if [ "$GLM53_NODE" = 22 ]; then
 export HCU_PLUGIN_DRAFT_LM_HEAD_VP_SIZE=1
else
 export HCU_PLUGIN_DRAFT_LM_HEAD_VP_SIZE=8
fi
unset SGLANG_NCCL_ALL_GATHER_IN_OVERLAP_SCHEDULER_SYNC_BATCH SGLANG_SCHEDULER_SKIP_ALL_GATHER GLM53_D2H_PATCH_PATH SGLANG_AUTO_NUMA_BIND
export SGLANG_HCU_NATIVE_MHC=1
unset HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS=1
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=96
export HCU_CONFIG="$SCRIPT_DIR/adapter-config.json"
export SGLANG_DSA_HCU_USE_LIGHTOP_DECODE_GATHER=1
common=(--served-model-name glm53-flash-pd --reasoning-parser glm45 --trust-remote-code
 --host "$HOST_IP" --port 30822 --language-only
 --kv-cache-dtype fp8_e4m3 --context-length 1048576 --chunked-prefill-size 32768 --max-prefill-tokens 32768
 --page-size 64
 --disable-prefill-cuda-graph
 --disable-shared-experts-fusion --disable-custom-all-reduce
 --dsa-topk-backend torch --dsa-prefill-backend flashmla_sparse --dsa-decode-backend flashmla_kv
 --moe-a2a-backend deepep --moe-runner-backend deep_gemm --deepep-dispatcher-output-dtype int8
 --disaggregation-transfer-backend mooncake --disaggregation-ib-device shca_0,shca_1,shca_2,shca_3
 --disaggregation-bootstrap-port 30998 --enable-metrics
 --speculative-algorithm EAGLE --speculative-num-steps 5
 --speculative-eagle-topk 1 --speculative-num-draft-tokens 6
 --json-model-override-args '{"index_share_for_mtp_iteration": true}')
if [ "$GLM53_NODE" = 22 ]; then
 role=(--random-seed 42 --model-path "$MODEL_DIR/GLM-5.3-Flash-W8A8-INT8" --quantization w8a8_int8
 --tp-size 8 --ep-size 8 --moe-dense-tp-size 1 --attn-cp-size 8 --enable-prefill-cp --cp-strategy interleave
 --enable-single-batch-overlap --deepep-mode normal --deepep-config "$SCRIPT_DIR/deepep.json"
 --enable-dsa-cache-layer-split --enable-hierarchical-cache
 --hicache-size 25 --hicache-write-policy write_through --hicache-io-backend kernel
 --hicache-mem-layout layer_first --mamba-radix-cache-strategy extra_buffer
 --disable-chunked-prefix-cache
 --disable-cuda-graph --max-running-requests 16 --max-mamba-cache-size 96
 --disaggregation-mode prefill --mem-fraction-static 0.85 --nccl-port 30922)
else
 rank=$((GLM53_NODE-23))
 export HIP_GRAPH_ACCUMULATE_DISPATCH=0 HIP_GRAPH_USE_CMD_CACHE=0
 export DEEPEP_ENABLE_LL_LAYERED_OPT=1
 role=(--random-seed 648620208 --model-path "$MODEL_DIR/GLM-5.3-Flash-W4A8-INT4" --quantization slimquant_w4a8
 --tp-size 16 --dp-size 16 --ep-size 16 --nnodes 2 --node-rank "$rank"
 --dist-init-addr "$DECODE_MASTER_IP:30923" --dist-timeout 600 --watchdog-timeout 1200
 --enable-dp-attention --enable-dp-lm-head --moe-dense-tp-size 1 --deepep-mode low_latency
 --disable-radix-cache --enable-linear-replayssm-spec --mamba-radix-cache-strategy no_buffer
 --max-running-requests 128 --max-mamba-cache-size 320
 --cuda-graph-bs-decode 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 --cuda-graph-max-bs-decode 16
 --disaggregation-mode decode --mem-fraction-static 0.80)
fi
mkdir -p "$WORK_DIR/logs/glm5.3-flash"
log="$WORK_DIR/logs/glm5.3-flash/server-node${GLM53_NODE}-$(date +%Y%m%d-%H%M%S).log"
exec python3 -u -m hcu_sglang "${common[@]}" "${role[@]}" > "$log" 2>&1
