#!/usr/bin/env bash
# One command: check both Sparks, build the engine image and send it to the worker, check the weights, start one rank
# on each Spark, wait for the API and smoke-test it.
# Usage: ./start.sh [restart]
# Configuration: scripts/config.sh defaults < .env < scripts/local.sh < exported environment.
# DRY_RUN=1 ./start.sh prints the full plan, including the link it finds, and changes nothing.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
# shellcheck source=scripts/prepare.sh
source ./scripts/prepare.sh
# shellcheck source=scripts/banner.sh
source ./scripts/banner.sh
case "${1:-}" in
  '') MODE=start ;;
  restart) MODE=restart ;;
  help|-h|--help) sed -n '2,7s/^# //p' "$0"; exit 0 ;;
  *) fail E_CONFIG 'Usage: ./start.sh [restart]. Set serving options in the environment or scripts/local.sh.' ;;
esac
(( $# <= 1 )) || fail E_CONFIG 'Usage: ./start.sh [restart].'
validate_config
set_image
inspect_model

step() { printf '\n[%s/6] %s\n' "$1" "$2"; }
bits() { case "$1" in bf16) echo 16 ;; q8) echo 8 ;; q6) echo 6 ;; q4) echo 4 ;; esac; }
API_HOST="$HOST"
[[ "$HOST" != 0.0.0.0 && "$HOST" != :: ]] || API_HOST=127.0.0.1
[[ "$API_HOST" != *:* ]] || API_HOST="[$API_HOST]"
URL="http://$API_HOST:$PORT"

# --- the context window ----------------------------------------------------------------------------------------------
FIT_FILE="$STATE_DIR/context-fit.tsv"
fit_key() {
  printf '%s|' "$PATCHES_HASH" "$MODEL_DIR" "$WORKER" "$TP" "$PARALLEL" "$KV" "$KV_PAGES" "$DENSE" "$HEAD" "$VISION" "$VISION_MAX_IMAGES" \
    "$DRAFTER" "$DRAFTER_BITS" "$MEMORY_RESERVE_GIB" "$PROMPT_CACHE_GIB" "$CACHE_FIRST" "$HOST_CACHE_GIB" "$LEARN_GIB" "$KV_SHARD" "$ROCE" "$MGBE_ZC" "$GLM_MM_CUBLAS" "$CONTEXT_MAX" "$SERVE_EXTRA_ARGS" \
    | sha256sum | cut -c1-16
}
fit_boot_id() {
  if [[ -n "${FIT_BOOT_ID:-}" ]]; then printf '%s\n' "$FIT_BOOT_ID"
  elif [[ -r /proc/sys/kernel/random/boot_id ]]; then cat /proc/sys/kernel/random/boot_id
  fi
}
remembered_context() {
  [[ -f "$FIT_FILE" ]] || return 0
  local boot; boot=$(fit_boot_id)
  awk -F'\t' -v k="$1" -v max="$CONTEXT_MAX" -v boot="$boot" '
    $1 == k { v = $2; b = $4 }
    END {
      if (v == "") exit
      if (v + 0 >= max + 0 || (boot != "" && b == boot)) print v
    }' "$FIT_FILE"
}
remember_context() {
  local tmp="$FIT_FILE.tmp" boot
  boot=$(fit_boot_id)
  mkdir -p "$STATE_DIR"
  { if [[ -f "$FIT_FILE" ]]; then awk -F'\t' -v k="$1" '$1 != k' "$FIT_FILE"; fi
    printf '%s\t%s\t%s\t%s\n' "$1" "$2" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$boot"; } > "$tmp"
  mv -- "$tmp" "$FIT_FILE"
}
# A smaller window the engine's memory refusals in the ranks' logs allow (nothing when they show none). The native
# engine states the bytes, not the window: the streams' caches at the asked window and the room left on that rank.
# Both Sparks' figures count and the tightest wins -- rank 0 also carries the HTTP server and the vision tower, and
# whatever else its Spark runs comes out of the same unified memory. A stream has a fixed part (KDA state, rings:
# CONTEXT_FIT_FIXED_GIB) besides its per-token caches; CONTEXT_FIT_MARGIN leaves room for the rest.
fit_context() {
  python3 - "$1" "$2" "$CONTEXT_FIT_MARGIN" "$CONTEXT" "${CONTEXT_FIT_FIXED_GIB:-0.4}" <<'PY'
import re, sys
text = open(sys.argv[1], errors="replace").read()
asked, margin, auto, fixed = int(sys.argv[2]), float(sys.argv[3]), sys.argv[4] == "auto", float(sys.argv[5])
num = r"([0-9]+(?:\.[0-9]+)?)"
w = None
named = re.findall(r"largest fitting prompt-plus-reply window: ([0-9]+) tokens", text)
if named:
    w = min(int(x) for x in named)
if w is None:  # windows a stream (--kv-cache fp8): the engine's own stream is in the model's bytes, so N share room + one
    m = re.findall(r"--parallel ([0-9]+) needs " + num + r" GiB for its streams at a ([0-9]+)-token window, "
                   r"and the CUDA memory budget leaves " + num + " GiB", text)
    fits = []
    for g in m:
        n, x, w0, r = int(g[0]), float(g[1]), int(g[2]), float(g[3])
        s = x / max(n - 1, 1)                    # one stream at w0
        f = min(fixed, s / 2)                    # its fixed part
        if s > f > 0 or s > 0:
            per_token = (s - f) / w0             # GiB a token
            fits.append(((r + s) / n - f) / per_token * margin)
    if fits:
        w = min(fits)
if w is None:  # shared KV pages (bf16, or fp8 with KV_PAGES=on): the engine's own window first, the other streams' fixed parts after
    m = re.findall(r"one at a ([0-9]+)-token window takes " + num + r" GiB of KV pages, and the budget leaves " + num + " GiB", text)
    if m:
        w0, p, r = int(m[-1][0]), float(m[-1][1]), float(m[-1][2])
        if p > 0:
            w = w0 * r / p * margin * 0.9
if w is None:
    m = re.findall(r"one at a ([0-9]+)-token window takes " + num + r" GiB, and " + num + " GiB is left after", text)
    if m:
        w0, s, r = int(m[-1][0]), float(m[-1][1]), float(m[-1][2])
        if s > 0:
            w = w0 * r / s * margin
if w is None and auto and re.search(r"\(OutOfMemory\)|CUDA_ERROR_OUT_OF_MEMORY", text):
    w = asked / 2  # the engine's own window did not allocate: no figures to scale by
if w is not None:
    w = int(w) // 1024 * 1024
    if 8192 <= w < asked:
        print(w)
PY
}
FIT_KEY=$(fit_key)
if [[ "$CONTEXT" == auto ]]; then
  CTX=$(remembered_context "$FIT_KEY")
  if [[ -n "$CTX" ]]; then
    if (( CTX < CONTEXT_MAX )); then CTX_NOTE="auto: cached shrunk fit from this boot ($FIT_FILE; delete it to retry $CONTEXT_MAX)"
    else CTX_NOTE="auto: the window that served these settings last time ($FIT_FILE)"; fi
  else CTX=$CONTEXT_MAX; CTX_NOTE="auto: try $CONTEXT_MAX, then once the largest window the engine's memory refusal allows"; fi
else
  CTX=$CONTEXT; CTX_NOTE=explicit
fi

# --- what every rank is given --------------------------------------------------------------------------------------
case "$CHAT_TEMPLATE" in
  auto) TEMPLATE_FILE=; [[ "$VISION" != 1 ]] || TEMPLATE_FILE="$PWD/templates/glm-5.3-flash-vision.jinja" ;;
  none) TEMPLATE_FILE= ;;
  /*) TEMPLATE_FILE=$CHAT_TEMPLATE ;;
  *) fail E_CONFIG 'CHAT_TEMPLATE must be auto, none or an absolute file path.' ;;
esac
[[ -z "$TEMPLATE_FILE" || -f "$TEMPLATE_FILE" ]] || fail E_CONFIG "Chat template not found: $TEMPLATE_FILE"
# EFFORT_LINE=thinking: the served template is the chosen one (the checkpoint's own with VISION=0) with its effort line
# set from reasoning_effort, which the engine passes with thinking on only (null with thinking off: prompt.zig), so a
# thinking-off prompt no longer carries `<|system|>Reasoning Effort: Max`. Written to STATE_DIR every start; the base
# line must be the one the checkpoint ships, else the start stops rather than serve a template it did not understand.
# With VISION=0 the base is the checkpoint's own file, which the default MODEL_SOURCE=download fetches in step 3: until
# that download is complete (weights_ready), step 3 writes the template after it (EFFORT_AFTER_DOWNLOAD). Every other
# start checks and writes it here, before anything runs or a serving pair is stopped.
effort_template() {
  if [[ "$DRY_RUN" == 1 && "$EFFORT_AFTER_DOWNLOAD" == 1 && ! -f "$EFFORT_BASE" ]]; then
    log "Then write $STATE_DIR/chat_template.jinja: the downloaded $EFFORT_BASE with its effort line taken from reasoning_effort (EFFORT_LINE=thinking)."
    return 0
  fi
  [[ -f "$EFFORT_BASE" ]] || fail E_CONFIG "Chat template not found: $EFFORT_BASE (EFFORT_LINE=thinking rewrites its effort line)."
  [[ "$DRY_RUN" == 1 ]] || mkdir -p "$STATE_DIR"
  python3 - "$EFFORT_BASE" "$STATE_DIR/chat_template.jinja" "$DRY_RUN" <<'PY' || fail E_CONFIG "EFFORT_LINE=thinking: $EFFORT_BASE has no effort line this recipe knows; set EFFORT_LINE=always or CHAT_TEMPLATE to a file."
import sys
base = "{%- set effective_reasoning_effort = reasoning_effort if reasoning_effort is defined and reasoning_effort in ['low', 'high'] else 'max' -%}"
line = "{%- set effective_reasoning_effort = (reasoning_effort if reasoning_effort in ['low', 'high'] else 'max') if (reasoning_effort is defined and reasoning_effort is not none) else none -%}"
src = open(sys.argv[1], encoding="utf-8").read()
if src.count(base) != 1:
    sys.exit(1)
if sys.argv[3] != "1":  # DRY_RUN=1 checks the base and writes nothing
    open(sys.argv[2], "w", encoding="utf-8").write(src.replace(base, line))
PY
}
EFFORT_AFTER_DOWNLOAD=0
if [[ "$CHAT_TEMPLATE" == auto && "$EFFORT_LINE" == thinking ]]; then
  EFFORT_BASE=${TEMPLATE_FILE:-$MODEL_DIR/chat_template.jinja}
  if [[ -z "$TEMPLATE_FILE" && "$MODEL_SOURCE" == download ]] && ! weights_ready "$MODEL_DIR" "$MODEL_REVISION"; then
    EFFORT_AFTER_DOWNLOAD=1
  else
    effort_template
  fi
  TEMPLATE_FILE="$STATE_DIR/chat_template.jinja"
fi
WORKER_TEMPLATE="$WORKER_STATE_DIR/chat_template.jinja"

build_serve_args() {
  local -a extra=()
  SERVE_ARGS=(--name "$SERVED_NAME" --host "$HOST" --port "$PORT" --context "$1" --parallel "$PARALLEL"
    --max-tokens "$MAX_TOKENS" --kv-cache "$KV" --no-update-check)
  if [[ "$THINKING" == 1 ]]; then SERVE_ARGS+=(--thinking); else SERVE_ARGS+=(--no-thinking); fi
  [[ -z "$REASONING_EFFORT" ]] || SERVE_ARGS+=(--reasoning-effort "$REASONING_EFFORT")
  [[ "$VISION" != 1 ]] || SERVE_ARGS+=(--vision --vision-max-images "$VISION_MAX_IMAGES")
  [[ -z "$TEMPLATE_FILE" ]] || SERVE_ARGS+=(--chat-template /templates/chat_template.jinja)
  case "$DRAFTER" in
    dflash2) SERVE_ARGS+=(--drafter /dflash2 --drafter-bits "$DRAFTER_BITS") ;;
    none) SERVE_ARGS+=(--no-drafts) ;;
  esac
  if [[ "$DRAFTER" != none ]]; then
    [[ -z "$MTP_DRAFTS" ]] || SERVE_ARGS+=(--mtp-drafts "$MTP_DRAFTS")
    [[ -z "$MTP_CONFIDENCE" ]] || SERVE_ARGS+=(--mtp-confidence "$MTP_CONFIDENCE")
  fi
  [[ -z "$PROMPT_CACHE_GIB" ]] || SERVE_ARGS+=(--prompt-cache-gib "$PROMPT_CACHE_GIB")
  # LEARN_GIB: engine patch 0113 keeps the prompt states on disk too (rank 0's CACHE_DIR/learn holds every rank's part)
  [[ "$LEARN_GIB" == 0 ]] || SERVE_ARGS+=(--learn-conversations --learn-dir /state/learn --learn-gib "$LEARN_GIB"
    --learn-min-free-gib "$LEARN_MIN_FREE_GIB")
  read -r -a extra <<< "$SERVE_EXTRA_ARGS"
  SERVE_ARGS+=("${extra[@]}")
}

# Recipe-managed names; any other exported TF_*, TENSORFOLD_* or NCCL_* variable reaches both ranks by name (values,
# TENSORFOLD_API_KEY's included, are never printed). The per-rank ones are added by launch().
declare -a PASS_ENV=()
WORKER_ENV_FILE=''
MANAGED_ENV=' HOME HF_HUB_OFFLINE CUDA_DEVICE_ORDER CUDA_VISIBLE_DEVICES TENSORFOLD_NO_UPDATE_CHECK TENSORFOLD_NO_LIVE
 TENSORFOLD_MEMORY_RESERVE_GIB TF_GLM_DENSE_BITS TF_GLM_HEAD_BITS TF_GLM_CLEAR_THINKING TF_GLM_RANK TF_GLM_WORLD TF_CUDA_DEVICE
 TF_KV_PAGES TF_GLM_CE_EXCHANGE TF_GLM_ABSORB_MMA TF_GLM_W8A8 TF_GLM_HOST_CACHE_GIB TENSORFOLD_PREFILL_SHARE TENSORFOLD_PREFILL_BUDGET TF_PROMPT_CACHE_FIRST TF_GLM_KV_SHARD TF_GLM_L2PF TF_EXTRA_EOS TF_THINK_OFF_SPLIT TF_GLM_MM_CUBLAS TF_GLM_PREFILL_REDUCE
 TF_ROCE TF_ROCE_MAX_KB TF_ROCE_HCA TF_MGBE_ZC TF_MGBE_ZC_MAX_KB TF_MGBE_ZC_EAGER_KB TF_MGBE_ZC_PORTS TF_MGBE_ZC_CPUS TF_BIN TF_REPO TF_REF
 NCCL_COMM_ID NCCL_SOCKET_IFNAME NCCL_IB_DISABLE NCCL_IB_HCA NCCL_IB_GID_INDEX NCCL_MIN_NCHANNELS NCCL_MAX_NCHANNELS '
# The flags both ranks share. Everything that places a rank is added per rank, so the two fingerprints agree on the rest.
build_env_args() {
  local key pr
  ENV_ARGS=(-e HOME=/state/home -e HF_HUB_OFFLINE=1 -e TENSORFOLD_NO_UPDATE_CHECK=1 -e TENSORFOLD_NO_LIVE=1
    -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e "TENSORFOLD_MEMORY_RESERVE_GIB=$MEMORY_RESERVE_GIB"
    -e "TF_GLM_DENSE_BITS=$(bits "$DENSE")" -e "TF_GLM_HEAD_BITS=$(bits "$HEAD")" -e "TF_GLM_CLEAR_THINKING=$CLEAR_THINKING"
    -e "TF_KV_PAGES=$KV_PAGES" -e "TF_GLM_CE_EXCHANGE=$CE_EXCHANGE" -e "TF_GLM_ABSORB_MMA=$ABSORB_MMA" -e "TF_GLM_W8A8=$W8A8"
    -e "TF_CUDA_DEVICE=0" -e "TF_GLM_WORLD=$TP" -e "NCCL_COMM_ID=$LINK_SELF:$NCCL_PORT"
    -e "TF_ROCE=$([[ "$ROCE" == on ]] && echo 1 || echo 0)" -e "TF_ROCE_MAX_KB=$ROCE_MAX_KB"
    -e "TF_MGBE_ZC=$([[ "$MGBE_ZC" == on ]] && echo 1 || echo 0)"
    -e "TF_BIN=$TF_BIN" -e "TENSORFOLD_PREFILL_SHARE=$PREFILL_SHARE")
  if [[ "$MGBE_ZC" == on ]]; then
    ENV_ARGS+=(-e "TF_MGBE_ZC_MAX_KB=$MGBE_ZC_MAX_KB" -e "TF_MGBE_ZC_EAGER_KB=$MGBE_ZC_EAGER_KB")
    [[ -z "$MGBE_ZC_CPUS" ]] || ENV_ARGS+=(-e "TF_MGBE_ZC_CPUS=$MGBE_ZC_CPUS")
  fi
  pr=$TP2_PREFILL_REDUCE
  [[ "$pr" != auto ]] || pr=$([[ "$MGBE_ZC" == on ]] && echo gather || echo split)
  [[ "$pr" == split ]] || ENV_ARGS+=(-e "TF_GLM_PREFILL_REDUCE=$pr")
  [[ "$GLM_MM_CUBLAS" != 1 ]] || ENV_ARGS+=(-e "TF_GLM_MM_CUBLAS=1")
  [[ -z "$HOST_CACHE_GIB" ]] || ENV_ARGS+=(-e "TF_GLM_HOST_CACHE_GIB=$HOST_CACHE_GIB")
  [[ -z "$PREFILL_BUDGET" ]] || ENV_ARGS+=(-e "TENSORFOLD_PREFILL_BUDGET=$PREFILL_BUDGET")
  # CACHE_FIRST=1: engine patch 0112 sets PROMPT_CACHE_GIB aside before the streams' windows (both ranks alike)
  [[ "$CACHE_FIRST" != 1 ]] || ENV_ARGS+=(-e "TF_PROMPT_CACHE_FIRST=1")
  # KV_SHARD=1: engine patch 0114 keeps each stream's latent rows on its owner rank (both ranks must agree)
  [[ "$KV_SHARD" != 1 ]] || ENV_ARGS+=(-e "TF_GLM_KV_SHARD=1")
  # L2PF: each decode window's next weights read into L2 (both ranks must agree). Named either way: unset is on on GB10
  ENV_ARGS+=(-e "TF_GLM_L2PF=$L2PF")
  # EXTRA_EOS: engine patch 0120 ends a reply at these tokens too (both ranks alike)
  [[ "$EXTRA_EOS" == none ]] || ENV_ARGS+=(-e "TF_EXTRA_EOS=$EXTRA_EOS")
  # THINK_OFF_SPLIT: engine patch 0121 moves a thinking-off draft closed by </think> into reasoning_content
  [[ "$THINK_OFF_SPLIT" == 0 ]] || ENV_ARGS+=(-e "TF_THINK_OFF_SPLIT=$THINK_OFF_SPLIT")
  # Variables you exported yourself. Rank 0 takes them by name, so no value is printed or put on a command line.
  # Rank 1 cannot: `docker run -e KEY` on the worker would read the worker's environment, where they are not set, and
  # the two ranks would then serve with different settings. Their values go to the worker in an env file instead
  # (over the ssh channel, mode 600), which keeps them off the remote command line too.
  PASS_ENV=()
  while IFS= read -r key; do
    [[ "$MANAGED_ENV" != *" $key "* ]] || continue
    case "$key" in TF_*|TENSORFOLD_*|NCCL_*) ENV_ARGS+=(-e "$key"); PASS_ENV+=("$key") ;; esac
  done < <(compgen -e | LC_ALL=C sort)
}

# The worker's copy of the exported variables, as a root-readable-only env file; echoes its path, or nothing.
write_worker_env() {
  WORKER_ENV_FILE=''
  (( ${#PASS_ENV[@]} )) || return 0
  WORKER_ENV_FILE="$WORKER_STATE_DIR/rank.env"
  if [[ "$DRY_RUN" == 1 ]]; then
    log "  (${#PASS_ENV[@]} exported variable(s) — ${PASS_ENV[*]} — are sent to $(worker_host) in $WORKER_ENV_FILE, values not shown)"
    return 0
  fi
  local key
  { for key in "${PASS_ENV[@]}"; do printf '%s=%s\n' "$key" "${!key}"; done; } \
    | wssh sh -c "umask 077; cat > $(printf '%q' "$WORKER_ENV_FILE")" \
    || fail E_START "Could not place the environment file on $(worker_host)."
}

# Rank r's own placement and link settings.
rank_env_args() {
  local r=$1
  RANK_ENV=(-e "TF_GLM_RANK=$r")
  if [[ "$LINK_KIND" == tcp ]]; then
    if (( r == 0 )); then
      RANK_ENV+=(-e "NCCL_SOCKET_IFNAME=$LINK_DEV" -e NCCL_IB_DISABLE=1)
      local zp="${MGBE_ZC_PORTS:-$LINK_ZC_PORTS}"
      [[ "$MGBE_ZC" != on || -z "$zp" ]] || RANK_ENV+=(-e "TF_MGBE_ZC_PORTS=$zp")
    else
      RANK_ENV+=(-e "NCCL_SOCKET_IFNAME=$WORKER_DEV" -e NCCL_IB_DISABLE=1)
      local zp="${MGBE_ZC_PORTS:-$WORKER_ZC_PORTS}"
      [[ "$MGBE_ZC" != on || -z "$zp" ]] || RANK_ENV+=(-e "TF_MGBE_ZC_PORTS=$zp")
    fi
  elif (( r == 0 )); then
    RANK_ENV+=(-e "NCCL_SOCKET_IFNAME=$LINK_DEV" -e "NCCL_IB_HCA=$LINK_HCA" -e "NCCL_IB_GID_INDEX=$LINK_GID")
    [[ "$ROCE" != on ]] || RANK_ENV+=(-e "TF_ROCE_HCA=$LINK_HCA")
  else
    RANK_ENV+=(-e "NCCL_SOCKET_IFNAME=$WORKER_DEV" -e "NCCL_IB_HCA=$WORKER_HCA" -e "NCCL_IB_GID_INDEX=$WORKER_GID")
    [[ "$ROCE" != on ]] || RANK_ENV+=(-e "TF_ROCE_HCA=$WORKER_HCA")
  fi
  RANK_ENV+=(-e "NCCL_MIN_NCHANNELS=$NCCL_CHANNELS" -e "NCCL_MAX_NCHANNELS=$NCCL_CHANNELS")
}

# Per-node container runtime flags (GPU passthrough, CUDA floor compat, /dev/infiniband, /dev/mgbe_zc + udev group).
build_run_args() {
  RUN_ARGS=(--ipc=host --network host --shm-size 16g --cap-add IPC_LOCK
            --ulimit memlock=-1 --ulimit stack=67108864)
  [[ ! -e /dev/infiniband ]] || RUN_ARGS+=(--device /dev/infiniband)
  local -a gpu_local=() gpu_worker=() zc_local=() zc_worker=()
  mapfile -t gpu_local < <(gpu_args '')
  mapfile -t gpu_worker < <(gpu_args worker)
  if [[ "$MGBE_ZC" == on ]]; then
    zc_local=(--device /dev/mgbe_zc)
    [[ -z "${MGBE_ZC_GID_LOCAL:-}" ]] || zc_local+=(--group-add "$MGBE_ZC_GID_LOCAL")
    zc_worker=(--device /dev/mgbe_zc)
    [[ -z "${MGBE_ZC_GID_WORKER:-}" ]] || zc_worker+=(--group-add "$MGBE_ZC_GID_WORKER")
  fi
  RUN_ARGS_LOCAL=("${gpu_local[@]}" "${RUN_ARGS[@]}" "${zc_local[@]}")
  RUN_ARGS_WORKER=("${gpu_worker[@]}" "${RUN_ARGS[@]}" "${zc_worker[@]}")
}

# The ranks' command lines, as each Spark starts them.
show_ranks() {
  local r
  for (( r = TP - 1; r >= 0; r-- )); do
    rank_env_args "$r"
    if (( r == 0 )); then
      printf '  rank 0 here (%s), HTTP server and vision tower:\n    ' "$(hostname)"
    else
      printf '  rank %s on %s (follows rank 0'"'"'s steps, binds no port):\n    ' "$r" "$(worker_host)"
    fi
    shq env "TF_GLM_RANK=$r" "TF_GLM_WORLD=$TP" TF_CUDA_DEVICE=0 "NCCL_COMM_ID=$LINK_SELF:$NCCL_PORT" \
      "$TF_BIN" serve /model "${SERVE_ARGS[@]}"
    printf '\n'
  done
}

# --- logs ------------------------------------------------------------------------------------------------------------
# Each rank is PID 1 in its own container, so its output is Docker's to keep: nothing wraps it and SIGTERM reaches the
# engine itself. collect_logs copies both ranks' output into the run directory, for the fit parser and the archive.
collect_logs() {
  [[ -n "${RUN_LOG_DIR:-}" && -d "$RUN_LOG_DIR" ]] || return 0
  docker logs "$CONTAINER_NAME" > "$RUN_LOG_DIR/rank0.log" 2>&1 || true
  wssh docker logs "$CONTAINER_NAME" > "$RUN_LOG_DIR/rank1.log" 2>&1 || true
}

# --- the memory guard ------------------------------------------------------------------------------------------------
# A Spark's weights, KV pool, prompt cache and everything else it runs share one pool, and a box driven into reclaim
# can stop answering long before the OOM killer picks a victim. The guard samples both Sparks and stops the server
# when one stays under the floor, so the pair comes down in order instead of being killed.
start_memguard() {
  [[ "$MEMGUARD" == 1 ]] || return 0
  if [[ "$DRY_RUN" == 1 ]]; then
    log "Memory guard: sample MemAvailable (plus parked pages) on both Sparks every ${MEMGUARD_INTERVAL_S} s and run"
    log "./stop.sh when one stays under ${MEMGUARD_FLOOR_GIB} GiB for ${MEMGUARD_HOLD_S} s."
    return 0
  fi
  # The guard outlives start.sh, so it takes nothing of start.sh's: not the lock (fd 8), and not the caller's
  # stdin, stdout or stderr. Holding them kept `ssh spark './start.sh'` from returning after LIVE (issue 7); its one
  # message goes to memguard.log in the run directory.
  ( exec 8>&-                     # never hold start.sh's lock
    under_here=0; under_there=0; tick=0; there=''
    # The worker's figure is taken about every 10 s (it costs an ssh round trip); this Spark's at every tick.
    every=$(( 10 / MEMGUARD_INTERVAL_S )); (( every > 0 )) || every=1
    while sleep "$MEMGUARD_INTERVAL_S"; do
      [[ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER_NAME" 2>/dev/null || true)" == true ]] || exit 0
      here=$(mem_avail_gb | awk '{print $1}')
      if (( tick % every == 0 )); then there=$(worker_avail_gib || true); fi
      tick=$(( tick + 1 ))
      if [[ -n "$here" ]] && ! at_least "$here" "$MEMGUARD_FLOOR_GIB"; then
        under_here=$(( under_here + MEMGUARD_INTERVAL_S ))
      else under_here=0; fi
      if [[ -n "$there" ]] && ! at_least "$there" "$MEMGUARD_FLOOR_GIB"; then
        under_there=$(( under_there + MEMGUARD_INTERVAL_S ))
      else under_there=0; fi
      # A worker sample is only as fresh as `every` ticks, so its hold is reached a sample at a time.
      if (( under_here >= MEMGUARD_HOLD_S )) || (( under_there >= MEMGUARD_HOLD_S )); then
        who=$(hostname); low=$here
        (( under_there < MEMGUARD_HOLD_S )) || { who=$(worker_host); low=$there; }
        printf '%s [memguard] %s: %s GiB available, under the %s GiB floor for %s s; stopping the server\n' \
          "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$who" "$low" "$MEMGUARD_FLOOR_GIB" "$MEMGUARD_HOLD_S"
        MEMGUARD=0 "$REPO_ROOT/stop.sh" 2>&1 || true
        exit 0
      fi
    done ) </dev/null >> "$RUN_LOG_DIR/memguard.log" 2>&1 &
  MEMGUARD_PID=$!
  log "[I_MEMGUARD] Watching both Sparks: a stop below ${MEMGUARD_FLOOR_GIB} GiB held ${MEMGUARD_HOLD_S} s (pid $MEMGUARD_PID, its log $RUN_LOG_DIR/memguard.log; MEMGUARD=0 turns it off)."
}
stop_memguard() { [[ -z "${MEMGUARD_PID:-}" ]] || { kill "$MEMGUARD_PID" 2>/dev/null || true; wait "$MEMGUARD_PID" 2>/dev/null || true; }; }

# The optional key is passed by environment name and as a header from a file descriptor; never printed.
api_curl() {
  if [[ -n "${TENSORFOLD_API_KEY:-}" ]]; then
    curl --header @<(printf 'Authorization: Bearer %s\n' "${TENSORFOLD_API_KEY%%,*}") "$@"
  else
    curl "$@"
  fi
}

# --- launch ----------------------------------------------------------------------------------------------------------
# Rank 1 on the worker first: it retries its line to rank 0 for up to 300 s. Then rank 0 here, which serves the API.
launch() {
  local ctx=$1
  local -a user_args=() worker_user_args=() mounts=() cmd=() wcmd=()
  local worker_ug='' unwritable
  build_serve_args "$ctx"
  build_env_args
  build_run_args
  [[ -z "$CONTAINER_USER" ]] || user_args=(--user "$CONTAINER_USER")
  worker_user_args=("${user_args[@]}")
  # The worker's own login when CONTAINER_USER is this Spark's (the default, or set to the same value): the two logins
  # can have different ids, and rank 1 writes to a state folder its own login owns. Root or another
  # UID:GID crosses as given.
  if [[ "$CONTAINER_USER" == "$(id -u):$(id -g)" ]]; then
    if worker_ug=$(wssh sh -c 'printf "%s:%s\n" "$(id -u)" "$(id -g)"') && [[ "$worker_ug" =~ ^[0-9]+:[0-9]+$ ]]; then
      worker_user_args=(--user "$worker_ug")
    elif [[ "$DRY_RUN" == 1 ]]; then  # a placeholder, as step 4 shows for the link; only a real start reads worker_ug
      warn W_WORKER "Cannot read the login's ids on $(worker_host), so the plan shows a placeholder for rank 1's --user."
      worker_ug=''; worker_user_args=(--user '<worker uid:gid>')
    else
      fail E_WORKER "Could not read the login's ids on $(worker_host)."
    fi
  fi
  RUN_LOG_DIR="$LOG_DIR/run-$(date -u +%Y%m%dT%H%M%SZ)-$$-$ATTEMPT"
  log "Ranks:"
  show_ranks
  if [[ "$DRY_RUN" != 1 ]]; then
    mkdir -p "$RUN_LOG_DIR" "$CACHE_DIR/home" "$CACHE_DIR/learn"
    ln -sfn -- "$RUN_LOG_DIR" "$LOG_DIR/current"
    wssh mkdir -p "$WORKER_CACHE_DIR/home" "$WORKER_STATE_DIR" \
      || fail E_START "Could not create $WORKER_CACHE_DIR on $(worker_host). It defaults to this Spark's home; when the logins differ, set WORKER_CACHE_DIR to a path under $(worker_host)'s own home in local.sh."
    # With different ids, a folder an earlier rank made in its HOME under this Spark's ids stops rank 1 with only AccessDenied.
    if [[ -n "$worker_ug" && "$worker_ug" != "$CONTAINER_USER" ]]; then
      unwritable=$(wssh find "$WORKER_CACHE_DIR/home" -type d ! -writable -print -quit 2>/dev/null) || true
      [[ -z "$unwritable" ]] || fail E_WORKER "$unwritable on $(worker_host) is not writable by the login rank 1 runs as ($worker_ug). Give $WORKER_CACHE_DIR/home to that login: sudo chown -R $worker_ug $WORKER_CACHE_DIR/home"
    fi
    [[ -z "$TEMPLATE_FILE" ]] || wssh tee "$WORKER_TEMPLATE" < "$TEMPLATE_FILE" >/dev/null \
      || fail E_START "Could not place the chat template on $(worker_host)."
  fi

  # rank 1 on the worker
  rank_env_args 1
  write_worker_env
  # Only the -e NAME=VALUE pairs cross as they are; the bare -e NAME ones become the env file above.
  local -a worker_env=()
  local a
  for a in "${ENV_ARGS[@]}"; do
    if [[ "$a" == -e ]]; then continue; fi
    [[ "$a" == *=* ]] && worker_env+=(-e "$a")
  done
  [[ -z "$WORKER_ENV_FILE" ]] || worker_env+=(--env-file "$WORKER_ENV_FILE")
  wcmd=(docker run -d --name "$CONTAINER_NAME" --label tf.recipe=glm53-spark --label "tf.patches=$PATCHES_HASH"
    --init --restart=no "${RUN_ARGS_WORKER[@]}" "${worker_user_args[@]}" "${worker_env[@]}" "${RANK_ENV[@]}"
    "${WORKER_MOUNTS[@]}" -v "$WORKER_CACHE_DIR:/state")
  [[ -z "$TEMPLATE_FILE" ]] || wcmd+=(-v "$WORKER_TEMPLATE:/templates/chat_template.jinja:ro")
  wcmd+=(--entrypoint "$TF_BIN" "$IMAGE" serve /model "${SERVE_ARGS[@]}")
  if [[ "$DRY_RUN" == 1 ]]; then
    log "On $(worker_host), over ssh:"
    plan "${wcmd[@]}"
  else
    wssh "${wcmd[@]}" >/dev/null || fail E_START "Could not start rank 1 on $(worker_host). Check its Docker and the image."
    log "Started rank 1 on $(worker_host)."
  fi

  # rank 0 here
  rank_env_args 0
  mounts=(-v "$MODEL_DIR:/model:ro" "${OVERLAY_MOUNT[@]}")
  [[ "$DRAFTER" != dflash2 ]] || mounts+=(-v "$DFLASH2_DIR:/dflash2:ro")
  [[ -z "$TEMPLATE_FILE" ]] || mounts+=(-v "$TEMPLATE_FILE:/templates/chat_template.jinja:ro")
  mounts+=(-v "$CACHE_DIR:/state")
  cmd=(docker run -d --name "$CONTAINER_NAME" --label tf.recipe=glm53-spark --label "tf.patches=$PATCHES_HASH"
    --init --restart=no "${RUN_ARGS_LOCAL[@]}" "${user_args[@]}" "${ENV_ARGS[@]}" "${RANK_ENV[@]}" "${mounts[@]}"
    --entrypoint "$TF_BIN" "$IMAGE" serve /model "${SERVE_ARGS[@]}")
  if [[ "$DRY_RUN" == 1 ]]; then log 'Here:'; plan "${cmd[@]}"; return 0; fi
  "${cmd[@]}" >/dev/null || { stop_both; fail E_START 'Docker could not launch rank 0. Check its error and rerun.'; }
  log "Started rank 0 here (logs: $RUN_LOG_DIR)."
}

# Both containers down, rank 0 first so it ends the follower's steps; used between fit attempts and on failure.
stop_both() {
  # A failed stop aborts the restart: layering a fresh rank onto a container that would not die is how
  # spark4 wedged on 2026-11-03. stop.sh escalates stop -> kill and verifies both ranks are gone;
  # a nonzero status here means a rank is still there and a human must look before anything starts.
  GLM_STOP_LOCK_HELD=1 MEMGUARD=0 "$REPO_ROOT/stop.sh" >/dev/null 2>&1 \
    || fail E_STOP 'The ranks would not stop (stop, then kill, both verified absent). Clean up by hand: docker ps on both Sparks.'
}

stop_follower() { kill "${log_follower:-}" 2>/dev/null || true; wait "${log_follower:-}" 2>/dev/null || true; trap - EXIT; }
# 0 once /health answers (rank 0 listens only after every rank loaded and agreed); 1 when a container stopped.
wait_ready() {
  local started=$SECONDS heartbeat=$SECONDS
  docker logs --follow --tail 5 "$CONTAINER_NAME" 8>&- & # closes fd 8: a follower left behind must not hold start.lock
  log_follower=$!
  trap stop_follower EXIT
  until curl --fail --silent --max-time 5 "$URL/health" >/dev/null 2>&1; do
    if [[ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER_NAME" 2>/dev/null || true)" != true ]]; then
      stop_follower; return 1
    fi
    # The worker is asked every sixth turn (about 30 s): rank 0 stops by itself when its follower goes, so this only
    # has to notice a rank 1 that never came up at all.
    if (( (SECONDS - started) % 30 < 5 )) && ! wssh docker inspect -f '{{.State.Running}}' "$CONTAINER_NAME" 2>/dev/null | grep -qx true; then
      stop_follower
      warn W_RANK "Rank 1 on $(worker_host) is no longer running."
      return 1
    fi
    if (( SECONDS - started >= WAIT_TIMEOUT )); then
      stop_follower
      collect_logs
      fail E_API "The API did not come up within $WAIT_TIMEOUT s. Inspect $LOG_DIR/current/rank*.log; raise WAIT_TIMEOUT for a first start (each rank hashes the checkpoint once)."
    fi
    if (( SECONDS - heartbeat >= 30 )); then
      log "Waiting for the API: $((SECONDS - started)) s (first start: each rank hashes the checkpoint once; then the load and graphs)."
      heartbeat=$SECONDS
    fi
    sleep 5
  done
  stop_follower
}
show_failure() {
  local f
  collect_logs
  for f in "$RUN_LOG_DIR"/rank*.log; do
    [[ -f "$f" ]] || continue
    printf '\n--- %s (last 15 lines) ---\n' "$f" >&2
    tail -n 15 -- "$f" >&2
  done
}

# The device prompt cache a rank's startup line ("… N streams at once, …, a 9.7 GiB prompt cache; …"; "1 stream" for one)
# funds, in GiB:
# 0 for "no prompt cache" or a cache in host memory, empty when the log has no startup line.
funded_cache_gib() {
  local line
  line=$(grep -hE ' streams? at once' "$1" 2>/dev/null | tail -1) || true
  [[ -n "$line" ]] || return 0
  if [[ "$line" != *"GiB prompt cache in host memory"* && "$line" =~ ,\ a\ ([0-9][0-9.]*)\ GiB\ prompt\ cache ]]; then
    printf '%s\n' "${BASH_REMATCH[1]}"
  else
    printf '0\n'
  fi
}
# The newest log archive stop.sh wrote (it removes the run directory once archived): the names carry a UTC stamp, so
# the last in glob order is the newest.
last_archive() {
  local a=("$LOG_DIR"/glm53-spark-*.tar.gz)
  if [[ -e "${a[-1]}" ]]; then printf '%s\n' "${a[-1]}"; else printf 'none written (see %s)\n' "$LOG_DIR"; fi
}
# PROMPT_CACHE_MIN_GIB: both ranks stop before LIVE when a startup line funds less (the cap is not a floor, issue 16).
check_cache_min() {
  [[ -n "$PROMPT_CACHE_MIN_GIB" ]] || return 0
  local r f got
  collect_logs
  for r in 0 1; do
    f="$RUN_LOG_DIR/rank$r.log"
    got=$(funded_cache_gib "$f")
    if [[ -z "$got" ]]; then
      if (( r == 0 )); then
        stop_both
        fail E_CACHE "PROMPT_CACHE_MIN_GIB=$PROMPT_CACHE_MIN_GIB: rank 0's log has no startup line (\"N streams at once\"), so the start cannot be checked; both ranks are stopped. Logs: $(last_archive)."
      fi
      # Rank 1 follows rank 0's plan and prints no startup line on the pair: rank 0's line is the check.
      continue
    fi
    if ! at_least "$got" "$PROMPT_CACHE_MIN_GIB"; then
      stop_both
      fail E_CACHE "Rank $r funds a $got GiB prompt cache, under PROMPT_CACHE_MIN_GIB=$PROMPT_CACHE_MIN_GIB; both ranks are stopped. The cache is what the room leaves, capped by PROMPT_CACHE_GIB: lower PARALLEL or CONTEXT, or with CACHE_FIRST=1 try a larger PROMPT_CACHE_GIB (it can cost the streams a whole window). Logs: $(last_archive)."
    fi
    log "Prompt cache: rank $r funds $got GiB (PROMPT_CACHE_MIN_GIB=$PROMPT_CACHE_MIN_GIB)."
  done
}

live_message() {
  brand_style
  printf '%sGLM-5.3-Flash-EXL3 is now LIVE! on port %s%s\n%sEndpoint: %s/v1%s\n' \
    "$AE_SILVER" "$PORT" "$AE_RESET" "$AE_WHITE" "$URL" "$AE_RESET"
}

SMOKE_JSON=$(python3 - "$SERVED_NAME" <<'PY'
import json, sys
print(json.dumps({"model": sys.argv[1], "messages": [{"role": "user", "content": "Reply with exactly: ready"}],
                  "temperature": 0, "max_tokens": 32, "chat_template_kwargs": {"enable_thinking": False}}))
PY
)

finish_start() {
  if [[ "$DRY_RUN" == 1 ]]; then
    log "Poll every 5 s for up to $WAIT_TIMEOUT s (rank 0 listens once both ranks have loaded and agreed):"
    plan curl --fail --silent --max-time 5 "$URL/health"
    if [[ "$CONTEXT_RETRY" == 1 ]]; then
      log "If a rank exits first and its log holds the engine's memory refusal (streams' caches vs room left), both"
      log "containers stop and the pair starts once more with the window the figures allow (x$CONTEXT_FIT_MARGIN)."
    fi
    [[ -z "${TENSORFOLD_API_KEY:-}" ]] || log '  (requests below carry Authorization: Bearer from TENSORFOLD_API_KEY; not shown)'
    plan curl --fail --silent --max-time 10 "$URL/v1/models"
    plan curl --fail --silent --show-error --max-time "$SMOKE_TIMEOUT" "$URL/v1/chat/completions" -H 'Content-Type: application/json' -d "$SMOKE_JSON"
    log 'Would check that /v1/models lists the served name and the reply has nonempty content, then print:'
    live_message
    return 0
  fi
  collect_logs
  api_curl --fail --silent --show-error --max-time 10 "$URL/v1/models" > "$RUN_LOG_DIR/models.json" \
    || fail E_API "GET /v1/models failed (an API key mismatch?). Inspect $LOG_DIR/current/rank0.log."
  python3 -c 'import json,sys; ids=[m.get("id") for m in json.load(open(sys.argv[1])).get("data", [])]; sys.exit(0 if sys.argv[2] in ids else 1)' \
    "$RUN_LOG_DIR/models.json" "$SERVED_NAME" || fail E_API "/v1/models does not list $SERVED_NAME (see $RUN_LOG_DIR/models.json)."
  api_curl --fail --silent --show-error --max-time "$SMOKE_TIMEOUT" "$URL/v1/chat/completions" \
    -H 'Content-Type: application/json' -d "$SMOKE_JSON" > "$RUN_LOG_DIR/smoke.json" \
    || fail E_SMOKE "The smoke request failed. Inspect $LOG_DIR/current/rank0.log and retry."
  python3 - "$RUN_LOG_DIR/smoke.json" <<'PY' || fail E_SMOKE "The smoke test returned no answer, or one character repeated (the weights load but the model does not answer). Inspect $LOG_DIR/current/smoke.json and rank0.log."
import json, sys
reply = json.load(open(sys.argv[1], encoding="utf-8"))
# Explicit checks, not asserts: PYTHONOPTIMIZE would skip those. curl --fail has already refused an HTTP error.
if reply.get("error") is not None: sys.exit("an error reply")
message = reply["choices"][0]["message"]
content = message.get("content")
if message.get("role") != "assistant" or not isinstance(content, str) or not content.strip(): sys.exit("no answer from the assistant")
text = content.strip()
if len(text) >= 8 and len(set(text)) <= 2: sys.exit("one character repeated")
print("Smoke test passed: " + text[:120])
PY
  printf '\n'
  grep -h -m1 -E 'CUDA sm_[0-9]+ device' "$RUN_LOG_DIR/rank0.log" 2>/dev/null || true
  grep -h -m1 -E '^.*serving .* at http' "$RUN_LOG_DIR/rank0.log" 2>/dev/null || true
  grep -h -m1 -E 'roce|p2p:' "$RUN_LOG_DIR/rank0.log" 2>/dev/null || true
  printf 'Health: %s/health\nLogs: %s/current/rank*.log\n' "$URL" "$LOG_DIR"
  live_message
}

banner
printf '\n%s\n%s at once | context %s (%s)\n%s dense, %s head, %s KV | %s drafts | vision %s | reserve %s GiB | port %s\nrank 0 here, rank 1 on %s | RoCE %s\n' \
  "$MODEL_DESC" "$PARALLEL" "$CTX" "$CTX_NOTE" "$DENSE" "$HEAD" "$KV$([[ "$KV_PAGES" != on ]] || echo ' (one shared page pool; context = one request'"'"'s cap)')" \
  "$DRAFTER" "$VISION" "$MEMORY_RESERVE_GIB" "$PORT" "$(worker_host)" "$ROCE"

step 1 'Check both Sparks: Linux aarch64, Docker, the NVIDIA runtime, memory, disk and ports'
if [[ "$DRY_RUN" == 1 ]]; then
  log '[I_DRY_RUN] Plan only: nothing is built, started, downloaded, sent or changed. The link below is probed read-only.'
  log "Check Linux aarch64, docker group access and the NVIDIA runtime here and on $(worker_host) over ssh; a GB10 on each;"
  log "$MIN_AVAIL_GIB GiB available on each (MemAvailable plus parked pages, as the engine counts it); disk for what is missing;"
  log "port $PORT on $HOST and $NCCL_PORT-$((NCCL_PORT + 1)) on this Spark's link address. Refuse if another start or stop holds $STATE_DIR/start.lock."
  if [[ "$MODE" == restart ]]; then DRY_RUN=1 "$REPO_ROOT/stop.sh"; fi
else
  check_platform
  mkdir -p "$STATE_DIR"
  exec 8>"$STATE_DIR/start.lock"
  flock -n 8 || fail E_LOCK 'Another start, stop or preparation is in progress.'
  running=$(docker inspect -f '{{.State.Running}}' "$CONTAINER_NAME" 2>/dev/null || true)
  if [[ "$running" == true && "$MODE" == start ]]; then
    log "$CONTAINER_NAME is already running. Endpoint: $URL/v1"
    log 'Use ./start.sh restart to apply changed settings, or ./stop.sh to stop it.'
    RUN_LOG_DIR=$(docker inspect -f '{{range .Mounts}}{{if eq .Destination "/logs"}}{{.Source}}{{end}}{{end}}' "$CONTAINER_NAME")
    [[ -n "$RUN_LOG_DIR" && -d "$RUN_LOG_DIR" ]] || RUN_LOG_DIR=$(readlink -f "$LOG_DIR/current" 2>/dev/null || true)
    [[ -d "$RUN_LOG_DIR" ]] || fail E_LOG 'The running container has no accessible log directory. Inspect its mounts.'
    detect_link
    wait_ready || { show_failure; fail E_START "A rank exited. Inspect $LOG_DIR/current/rank*.log; ./stop.sh archives them."; }
    finish_start
    exit 0
  fi
  # Settings are validated before an existing server is stopped; its logs are archived first.
  if docker container inspect "$CONTAINER_NAME" >/dev/null 2>&1 || wssh docker container inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
    GLM_STOP_LOCK_HELD=1 "$REPO_ROOT/stop.sh"
  fi
  check_gpu
  check_memory
  check_disk
  python3 - "$HOST" "$PORT" "$NCCL_PORT" <<'PY' || fail E_PORT "Port $PORT or $NCCL_PORT-$((NCCL_PORT + 1)) is in use (another server?). Stop its owner or change PORT / NCCL_PORT."
import socket, sys
host, port, nccl = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
for h, p in ((host, port), ("0.0.0.0", nccl), ("0.0.0.0", nccl + 1)):
    family = socket.AF_INET6 if ":" in h else socket.AF_INET
    with socket.socket(family, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((h, p))
PY
  log "[I_PORT] Port $PORT and $NCCL_PORT-$((NCCL_PORT + 1)) are free."
fi

check_weights_access

step 2 "Build the TensorFold native image with $PATCH_COUNT engine patches for sm_$SM, and send it to $(worker_host)"
prepare_image
ship_image

step 3 'Download or check the checkpoint and the drafter weights on both Sparks'
prepare_weights
model_overlay
[[ "$EFFORT_AFTER_DOWNLOAD" == 0 ]] || effort_template

step 4 "Find the link between the Sparks and the RDMA devices it runs on"
check_link
prepare_worker_weights

step 5 "Start rank 1 on $(worker_host), then rank 0 here"
ATTEMPT=1
launch "$CTX"

step 6 'Wait for the API and run a greedy smoke test'
if [[ "$DRY_RUN" != 1 ]] && ! wait_ready; then
  # Up to CONTEXT_RETRIES more starts, each at the window the last refusal allows (the estimate tightens every time).
  while :; do
    collect_logs
    new=''
    # Both ranks' logs: when one rank leaves first, the other can be stopped before its own refusal line is written.
    [[ "$CONTEXT_RETRY" != 1 ]] || new=$(fit_context <(cat "$RUN_LOG_DIR"/rank*.log 2>/dev/null) "$CTX" || true)
    # No figures at all under auto on the first refusal: the newest window any settings served, when smaller.
    if [[ -z "$new" && "$CONTEXT_RETRY" == 1 && "$CONTEXT" == auto && "$ATTEMPT" == 1 && -f "$FIT_FILE" ]]; then
      new=$(sort -t$'\t' -k3,3 "$FIT_FILE" | awk -F'\t' -v asked="$CTX" 'END { if ($2 + 0 >= 8192 && $2 + 0 < asked) print $2 }')
    fi
    # On KV pages the plan sets the engine's own --context window aside first, then the other streams' fixed parts. A
    # refusal there names what fits ("fits: --parallel N, TENSORFOLD_MEMORY_RESERVE_GIB=G (now G), --context T"): under
    # CONTEXT=auto take its --context; otherwise, or when it names none, say so and stop.
    if grep -qh "for its other streams' fixed parts" "$RUN_LOG_DIR"/rank*.log 2>/dev/null; then
      fits=$(grep -ho 'fits: .*' "$RUN_LOG_DIR"/rank*.log 2>/dev/null | tail -1)
      t=$(grep -o -- '--context [0-9]*' <<< "$fits" | grep -o '[0-9]*$' || true)
      if [[ "$CONTEXT" == auto && -n "$t" ]] && (( t >= 8192 && t < CTX )); then
        new=$(( t * 99 / 100 / 1024 * 1024 ))
      else
        show_failure
        fail E_START "The shared KV page pool can't hold $PARALLEL streams' fixed parts beside one $CTX-token window ($(grep -ho "needs [0-9.]* GiB for its other streams' fixed parts.*" "$RUN_LOG_DIR"/rank*.log | tail -1)). ${fits:+The engine says $fits. }Lower CONTEXT or MEMORY_RESERVE_GIB (now $MEMORY_RESERVE_GIB), or PARALLEL."
      fi
    fi
    if [[ -z "$new" ]] || (( ATTEMPT > ${CONTEXT_RETRIES:-3} )); then
      show_failure
      (( ATTEMPT == 1 )) && fail E_START "A rank exited during startup. Inspect $LOG_DIR/current/rank*.log; ./stop.sh archives them."
      fail E_START "Start $ATTEMPT (--context $CTX) failed too. Inspect $LOG_DIR/current/rank*.log; lower CONTEXT or PARALLEL, or raise MEMORY_RESERVE_GIB."
    fi
    warn W_CONTEXT "This start's memory holds less than a $CTX-token window at --parallel $PARALLEL; starting once more with --context $new."
    stop_both
    ATTEMPT=$(( ATTEMPT + 1 ))
    CTX=$new
    launch "$CTX"
    wait_ready && break
  done
fi
[[ "$DRY_RUN" == 1 ]] || check_cache_min
start_memguard
finish_start
if [[ "$DRY_RUN" != 1 ]]; then
  if [[ "$CONTEXT" == auto ]]; then
    remember_context "$FIT_KEY" "$CTX"
    log "Context: $CTX tokens (kept for the next start with these settings; delete $FIT_FILE to probe again)."
    # Windows (KV_PAGES=off, or auto with KV=fp8) have no pages line, so no pool to report and no pipefail exit.
    pool=$(grep -ho 'taking [0-9]* pages of 512 rows' "$RUN_LOG_DIR/rank0.log" 2>/dev/null | tail -1 | grep -o '[0-9]*' | head -1) || true
    [[ -z "$pool" ]] || log "KV pool: $pool pages = $(( pool * 512 )) tokens shared by $PARALLEL streams (one request up to $CTX)."
    # A few MiB more fixed memory can quietly drop a stream. One stream reads "1 stream at once"; auto names no count.
    [[ "$PARALLEL" == auto ]] || grep -qE "(^|[^0-9])$PARALLEL streams? at once" "$RUN_LOG_DIR/rank0.log" 2>/dev/null ||
      warn W_STREAMS "The engine serves fewer than PARALLEL=$PARALLEL streams at once; see the startup line in $RUN_LOG_DIR/rank0.log."
  elif (( ATTEMPT == 2 )); then
    log "Context: $CTX tokens, not the $CONTEXT asked; set CONTEXT=$CTX in scripts/local.sh to skip the refused start."
  fi
fi
