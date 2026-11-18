#!/usr/bin/env bash
# Yee 3.5.2 mtp_bench: does the CPU keep its memory speed while the GPU decodes (and the other way round)?
# Run it before putting an MTP draft head on the CPU next to a GPU target.
#
#   BIN=<llama.cpp build/bin> MODEL=<model.gguf> ./contention.sh
#
# BIN and MODEL are required. OUT (result directory, default ./contention-<time>), THREADS (CPU threads,
# default 16) and LOAD_WAIT (seconds for the CPU llama-bench to load and repack the model, default 75) are
# optional. Only the llama-bench processes this script starts are stopped.
set -u
BIN="${BIN:?set BIN to the llama.cpp build/bin directory}"
MODEL="${MODEL:?set MODEL to a .gguf file}"
OUT="${OUT:-./contention-$(date +%Y%m%d-%H%M%S)}"
THREADS="${THREADS:-16}"
LOAD_WAIT="${LOAD_WAIT:-75}"
mkdir -p "$OUT"
cd "$OUT" || exit 1

run_gpu() { "$BIN/llama-bench" -m "$MODEL" -ngl 99 -fa 1 -p 4 -n 32 -r 5 -o csv 2>/dev/null; }
run_cpu() { "$BIN/llama-bench" -m "$MODEL" -ngl 0 -dev none -t "$THREADS" -fa 1 -p 0 -n "$1" -r 1 -o csv 2>/dev/null; }

echo "[contention] GPU alone"
run_gpu > gpu_alone.csv
run_cpu 4000 > cpu_load.csv &
CP=$!
sleep "$LOAD_WAIT"
echo "[contention] GPU while the CPU decodes"
run_gpu > gpu_with_cpu.csv
kill "$CP" 2>/dev/null
wait "$CP" 2>/dev/null
echo "[contention] CPU alone"
run_cpu 64 > cpu_alone.csv
"$BIN/llama-bench" -m "$MODEL" -ngl 99 -fa 1 -p 0 -n 4000 -r 1 -o csv > gpu_load.csv 2>/dev/null &
GP=$!
sleep 30
echo "[contention] CPU while the GPU decodes"
run_cpu 64 > cpu_with_gpu.csv
kill "$GP" 2>/dev/null
wait "$GP" 2>/dev/null

python3 - <<'PY'
import csv


def speeds(path):
    out = {}
    try:
        rows = list(csv.DictReader(open(path)))
    except OSError:
        return out
    for r in rows:
        key = ("pp%s" % r["n_prompt"]) if r.get("n_prompt", "0") != "0" else ("tg%s" % r["n_gen"])
        out[key] = float(r["avg_ts"])
    return out


ga, gc = speeds("gpu_alone.csv"), speeds("gpu_with_cpu.csv")
ca, cg = speeds("cpu_alone.csv"), speeds("cpu_with_gpu.csv")
for k, v in ga.items():
    w = gc.get(k, 0.0)
    print("GPU %-6s alone %8.2f t/s, with the CPU decoding %8.2f t/s (%+.1f%%)" % (k, v, w, (w / v - 1) * 100))
for k, v in ca.items():
    w = cg.get(k, 0.0)
    print("CPU %-6s alone %8.2f t/s, with the GPU decoding %8.2f t/s (%+.1f%%)" % (k, v, w, (w / v - 1) * 100))
print("A CPU draft head can only overlap the GPU verify if the CPU keeps most of its speed here. "
      "mx4 (Ryzen AI Max+ 395, shared LPDDR5X): the GPU lost 11.7%, the CPU kept 28%.")
PY
