#!/usr/bin/env bash
# run_matrix.sh — llama-server 投机解码多配置基准（bash）
# 用法：./run_matrix.sh 配置名...（配置名：base / gpu_k1..k3 / cpu_k1..k3）
# 不依赖 curl / sudo，不 kill 自己启动之外的任何进程。
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BIN="${BIN:-$HOME/r670/llama.cpp/build-vk/bin}"
MODEL="${MODEL:-$HOME/r670/models/Qwen3.6-27B-Q4_K_M.gguf}"
PORT="${PORT:-18670}"
OUT="${OUT:-$HOME/r670/results/$(date +%Y%m%d-%H%M%S)}"
DRAFT_THREADS="${DRAFT_THREADS:-16}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

BENCH="$SCRIPT_DIR/bench.py"
COMPARE="$SCRIPT_DIR/compare.py"
PROMPTS="$SCRIPT_DIR/prompts.jsonl"
MAX_TOKENS="${MAX_TOKENS:-256}"
REPEAT="${REPEAT:-1}"
BENCH_TIMEOUT="${BENCH_TIMEOUT:-600}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-600}"

SERVER_PID=""

log() { echo "[run_matrix] $*"; }

cleanup() {
    if [ -n "${SERVER_PID:-}" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
        log "TERM server pid $SERVER_PID"
        kill -TERM "$SERVER_PID" 2>/dev/null || true
        for _ in $(seq 1 60); do
            kill -0 "$SERVER_PID" 2>/dev/null || break
            sleep 1
        done
        if kill -0 "$SERVER_PID" 2>/dev/null; then
            log "KILL server pid $SERVER_PID"
            kill -KILL "$SERVER_PID" 2>/dev/null || true
        fi
        wait "$SERVER_PID" 2>/dev/null || true
        SERVER_PID=""
    fi
}
trap cleanup EXIT INT TERM

wait_port_free() {
    for _ in $(seq 1 60); do
        if python3 -c "
import socket,sys
s=socket.socket()
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
try:
    s.bind(('127.0.0.1', $PORT))
except OSError:
    sys.exit(1)
finally:
    s.close()
sys.exit(0)
" 2>/dev/null; then
            return 0
        fi
        sleep 1
    done
    return 1
}

health_wait() {
    local deadline=$(( $(date +%s) + HEALTH_TIMEOUT ))
    while [ "$(date +%s)" -lt "$deadline" ]; do
        if python3 -c "
import sys, urllib.request
try:
    with urllib.request.urlopen('http://127.0.0.1:$PORT/health', timeout=5) as r:
        sys.exit(0 if r.status == 200 else 1)
except Exception:
    sys.exit(1)
" 2>/dev/null; then
            return 0
        fi
        sleep 2
    done
    return 1
}

config_args() {
    case "$1" in
        base)
            echo "--spec-type none"
            ;;
        gpu_k1|gpu_k2|gpu_k3)
            echo "--spec-type draft-mtp --spec-draft-n-max ${1##*_k}"
            ;;
        cpu_k1|cpu_k2|cpu_k3)
            echo "--spec-type draft-mtp --spec-draft-n-max ${1##*_k} -md $MODEL -devd none -ngld 0 -td $DRAFT_THREADS"
            ;;
        *)
            echo ""
            return 1
            ;;
    esac
}

if [ "$#" -lt 1 ]; then
    log "usage: ./run_matrix.sh 配置名...（base gpu_k1 gpu_k2 gpu_k3 cpu_k1 cpu_k2 cpu_k3）"
    exit 2
fi

CONFIGS=()
for c in "$@"; do
    if ! config_args "$c" >/dev/null; then
        log "未知配置：$c（可用：base gpu_k1 gpu_k2 gpu_k3 cpu_k1 cpu_k2 cpu_k3）"
        exit 2
    fi
    CONFIGS+=("$c")
done

mkdir -p "$OUT"
log "OUT=$OUT"

for cfg in "${CONFIGS[@]}"; do
    log "=== 配置 $cfg ==="
    cleanup
    if ! wait_port_free; then
        log "端口 $PORT 未释放，跳过 $cfg"
        continue
    fi

    SPEC_ARGS="$(config_args "$cfg")"
    LOGF="$OUT/server-$cfg.log"

    # shellcheck disable=SC2086
    "$BIN/llama-server" -m "$MODEL" --host 127.0.0.1 --port "$PORT" \
        -np 1 -c 8192 -ngl 99 -fa on $SPEC_ARGS $EXTRA_ARGS > "$LOGF" 2>&1 &
    SERVER_PID=$!
    log "server pid=$SERVER_PID log=$LOGF"

    if ! health_wait; then
        log "health 等待超时（${HEALTH_TIMEOUT}s），$LOGF 尾部 40 行："
        tail -n 40 "$LOGF" 2>/dev/null || true
        cleanup
        continue
    fi
    log "health OK"

    if python3 "$BENCH" --base "http://127.0.0.1:$PORT" --prompts "$PROMPTS" \
        --tag "$cfg" --out "$OUT/$cfg.json" --max-tokens "$MAX_TOKENS" \
        --repeat "$REPEAT" --timeout "$BENCH_TIMEOUT" --warmup; then
        log "bench OK -> $OUT/$cfg.json"
    else
        log "bench 存在失败项 -> $OUT/$cfg.json"
    fi
done

cleanup

BASELINE="${CONFIGS[0]}"
INPUTS=()
for cfg in "${CONFIGS[@]}"; do
    [ -f "$OUT/$cfg.json" ] && INPUTS+=("$OUT/$cfg.json")
done
if [ "${#INPUTS[@]}" -gt 0 ]; then
    python3 "$COMPARE" --inputs "${INPUTS[@]}" --baseline "$BASELINE" --out "$OUT/compare.md"
    log "compare -> $OUT/compare.md"
else
    log "没有可用的 json 输出，跳过 compare"
fi

log "全部完成"
