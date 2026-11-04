#!/bin/bash
set -euo pipefail
: "${WORK_DIR:?Set writable work directory}"
: "${ROUTER_HOST:?Set router bind IP}"
: "${PREFILL_URL:?Set http://prefill-ip:30822}"
: "${DECODE_URL:?Set http://decode-rank0-ip:30822}"
mkdir -p "$WORK_DIR/logs/glm5.3-flash"
exec python3 -m sglang_router.launch_router \
  --host "$ROUTER_HOST" --port "${ROUTER_PORT:-30881}" \
  --prometheus-port "${PROMETHEUS_PORT:-29081}" --pd-disaggregation \
  --prefill "$PREFILL_URL" "${BOOTSTRAP_PORT:-30998}" --decode "$DECODE_URL" \
  --policy round_robin --worker-startup-timeout-secs 3600 \
  --request-timeout-secs 7500 --disable-retries \
  > "$WORK_DIR/logs/glm5.3-flash/router-$(date +%Y%m%d-%H%M%S).log" 2>&1
