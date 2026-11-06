#!/usr/bin/env bash
# Run the TP=2 gather benchmark on the pair: rank 1 in a short-lived container on
# the worker, rank 0 here. Neither touches the serving containers.
#
#   ./run.sh                 # both paths, default size sweep
#   SIZES=4194304 ./run.sh   # one size (per-rank bytes)
#   MODE=nccl ./run.sh       # NCCL only
#   NCCL_SOCKET_NTHREADS=4 NCCL_NSOCKS_PERTHREAD=4 ./run.sh   # tuned NCCL
#
# Results: $OUT/rank0.json, $OUT/rank1.json and a merged summary.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
WORKER=${WORKER:-test@192.168.100.39}
IMAGE=${IMAGE:-deepinfer1017-tp2:runtime-0.1.3-arm64}
PORT=${PORT:-29600}
CONNS=${CONNS:-8}
ITERS=${ITERS:-20}
WARMUP=${WARMUP:-5}
MODE=${MODE:-both}
SIZES=${SIZES:-}
LOCAL_RAILS=${LOCAL_RAILS:-192.168.100.38,192.168.101.38,192.168.102.38,192.168.103.38}
PEER_RAILS=${PEER_RAILS:-192.168.100.39,192.168.101.39,192.168.102.39,192.168.103.39}
MASTER=${MASTER:-${LOCAL_RAILS%%,*}}
RDIR=${RDIR:-/tmp/tp_gather_bench}
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
OUT=${OUT:-$HERE/results/$STAMP}
NAME=tp-gather-bench-$$

log() { printf '[bench] %s\n' "$*"; }

# NCCL knobs to forward into both containers, when set in this shell.
PASS=()
for v in NCCL_SOCKET_NTHREADS NCCL_NSOCKS_PERTHREAD NCCL_MIN_NCHANNELS NCCL_MAX_NCHANNELS \
         NCCL_CHANNELS NCCL_ALGO NCCL_PROTO NCCL_DEBUG NCCL_SOCKET_FAMILY; do
  [[ -n "${!v:-}" ]] && PASS+=(-e "$v=${!v}")
done

run_args() { # $1 rank, $2 json path; each rank is told its own addresses
  local rank=$1 json=$2
  local rails=$LOCAL_RAILS peers=$PEER_RAILS
  (( rank == 1 )) && { rails=$PEER_RAILS; peers=$LOCAL_RAILS; }
  local -a args=(
    --rank "$rank" --world 2 --master "$MASTER" --port "$PORT"
    --conns "$CONNS" --iters "$ITERS" --warmup "$WARMUP" --mode "$MODE"
    --rails "$rails" --peer-rails "$peers"
    --json "$json"
  )
  [[ -n "$SIZES" ]] && args+=(--sizes "$SIZES")
  printf '%s\n' "${args[@]}"
}

log "checking the worker and the image"
ssh -o BatchMode=yes -o ConnectTimeout=10 "$WORKER" "docker image inspect $IMAGE >/dev/null" \
  || { log "worker $WORKER has no $IMAGE"; exit 1; }

log "copying the benchmark to $WORKER:$RDIR"
tar -C "$HERE" -cf - bench_tp_gather.py \
  | ssh -o BatchMode=yes "$WORKER" "rm -rf '$RDIR' && mkdir -p '$RDIR' && tar -C '$RDIR' -xf -"

mkdir -p "$OUT"
mapfile -t R1 < <(run_args 1 /bench/rank1.json)
mapfile -t R0 < <(run_args 0 /out/rank0.json)

cleanup() {
  ssh -o BatchMode=yes "$WORKER" "docker rm -f $NAME >/dev/null 2>&1 || true" >/dev/null 2>&1 || true
}
trap cleanup EXIT

log "rank 1 on $WORKER (container $NAME)"
ssh -o BatchMode=yes "$WORKER" "docker rm -f $NAME >/dev/null 2>&1 || true; \
  docker run --rm --name $NAME --network host --ipc host --runtime=nvidia \
    -e NVIDIA_VISIBLE_DEVICES=all -e PYTHONUNBUFFERED=1 ${PASS[*]:-} \
    -v '$RDIR':/bench --entrypoint python3 '$IMAGE' /bench/bench_tp_gather.py ${R1[*]} \
    > '$RDIR/rank1.out' 2>&1" &
SSH_PID=$!

sleep 2
log "rank 0 here"
docker run --rm --network host --ipc host --runtime=nvidia \
  -e NVIDIA_VISIBLE_DEVICES=all -e PYTHONUNBUFFERED=1 "${PASS[@]}" \
  -v "$HERE":/bench -v "$OUT":/out --entrypoint python3 "$IMAGE" /bench/bench_tp_gather.py "${R0[@]}" \
  | tee "$OUT/rank0.out"

log "waiting for rank 1"
wait "$SSH_PID" || true
scp -q -o BatchMode=yes "$WORKER:$RDIR/rank1.out" "$OUT/rank1.out" 2>/dev/null || true
scp -q -o BatchMode=yes "$WORKER:$RDIR/rank1.json" "$OUT/rank1.json" 2>/dev/null || true

log "summary"
python3 - "$OUT" <<'PY'
import glob, json, os, sys
out = sys.argv[1]
reports = []
for path in sorted(glob.glob(os.path.join(out, 'rank*.json'))):
    with open(path) as f:
        reports.append(json.load(f))
if not reports:
    print('no rank json; see rank0.out / rank1.out')
    sys.exit(1)
base = reports[0]
print(f"torch {base['torch']} cuda {base['cuda']} nccl {base['nccl']} on {base['gpu']}")
print(f"ifname {base['nccl_socket_ifname']}  threads {base['nccl_socket_nthreads']}  "
      f"nsocks {base['nccl_nsocks_perthread']}  conns {base['conns']}")
print(f"{'rows':>5} {'KiB':>8} {'nccl us':>9} {'nccl GB/s':>10} {'nccl chain':>11} "
      f"{'tcp us':>9} {'tcp GB/s':>9} {'rails rx MiB':>28}")
for i, entry in enumerate(base['results']):
    n = entry['bytes_per_rank']
    npath = entry['paths'].get('nccl')
    tpath = entry['paths'].get('tcp')
    def f(p, k, fmt):
        return format(p[k], fmt) if p else '-'
    rails = ''
    if tpath and tpath.get('rails'):
        per = [v['rx_bytes'] / 2**20 for v in tpath['rails'].values() if v]
        rails = ' '.join(f'{x:6.0f}' for x in per)
    print(f"{entry['rows']:>5} {n >> 10:>8} {f(npath,'median_us','9.1f')} {f(npath,'wire_GBps','10.2f')} "
          f"{f(npath,'chained_us','11.1f')} {f(tpath,'median_us','9.1f')} {f(tpath,'wire_GBps','9.2f')} {rails}")
# rank 1 sees the mirror image; flag asymmetry above 10%.
if len(reports) > 1:
    other = reports[1]
    for a, b in zip(base['results'], other['results']):
        for path in ('nccl', 'tcp'):
            pa, pb = a['paths'].get(path), b['paths'].get(path)
            if pa and pb:
                ratio = max(pa['median_us'], pb['median_us']) / min(pa['median_us'], pb['median_us'])
                if ratio > 1.10:
                    print(f"note: rank asymmetry on {path} at {a['rows']} rows: "
                          f"{pa['median_us']:.1f} vs {pb['median_us']:.1f} us")
PY
log "wrote $OUT"
