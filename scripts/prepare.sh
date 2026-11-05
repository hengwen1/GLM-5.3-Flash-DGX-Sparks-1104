#!/usr/bin/env bash
# Build the engine image and check the weights once. Also sourced by start.sh and stop.sh for its helpers.
# Usage: scripts/prepare.sh [--image|--weights|--check|--label]
#   (none)     image and weights          --image    the image only       --weights  the weights only
#   --check    platform, both Sparks' memory and the link between them (needs the built image)
#   --label    print the image hash (tf.patches label) and exit
# DRY_RUN=1 prints the plan without Docker, GPUs, downloads or file changes.
set -euo pipefail
# shellcheck source=scripts/config.sh
source "$(dirname -- "${BASH_SOURCE[0]}")/config.sh"

log() { printf '%s\n' "$*"; }
fail() { printf '[%s] %s\n' "$1" "$2" >&2; exit 1; }
warn() { printf '[%s] %s\n' "$1" "$2" >&2; }
# Words as a shell would read them back: plain when safe, else single-quoted.
shq() { local a; for a in "$@"; do if [[ "$a" =~ ^[A-Za-z0-9_./:=,@%+-]+$ ]]; then printf '%s ' "$a"; else printf "'%s' " "${a//\'/\'\\\'\'}"; fi; done; }
plan() { printf '  '; shq "$@"; printf '\n'; }
run() { if [[ "$DRY_RUN" == 1 ]]; then plan "$@"; else "$@"; fi; }
is_number() { [[ "$1" =~ ^[0-9]+([.][0-9]+)?$ ]]; }
at_least() { python3 -c 'import sys; sys.exit(0 if float(sys.argv[1]) >= float(sys.argv[2]) else 1)' "$1" "$2"; }

# The two Sparks: ssh, the link, the image and the worker's weights.
# shellcheck source=scripts/nodes.sh
source "$(dirname -- "${BASH_SOURCE[0]}")/nodes.sh"

# The GPU's compute capability as the engine's -Dsm: 121 on a GB10 (DGX Spark), 110 on a Jetson AGX Thor. Empty or
# auto asks the driver on this node; an explicit SM wins (comma-separated lists are the engine's own syntax).
detect_sm() {
  local cc
  cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')
  [[ "$cc" =~ ^[0-9]+\.[0-9]+$ ]] || return 1
  printf '%s\n' "${cc//./}"
}
resolve_arch() {
  [[ -z "$SM" || "$SM" == auto ]] || return 0
  local detected
  if detected=$(detect_sm); then SM=$detected
  else
    SM=121
    warn W_SM "Cannot read the GPU's compute capability; building for sm_121 (the engine's default). Set SM to override."
  fi
}

validate_config() {
  local key path
  resolve_arch
  [[ "$DRY_RUN" =~ ^[01]$ ]] || fail E_CONFIG 'DRY_RUN must be 0 or 1.'
  [[ "$MODEL_SOURCE" =~ ^(local|download)$ ]] || fail E_CONFIG 'MODEL_SOURCE must be local or download.'
  [[ "$DRAFTER" =~ ^(dflash2|mtp|none)$ ]] || fail E_CONFIG 'DRAFTER must be dflash2, mtp or none.'
  [[ "$KV" =~ ^(fp8|bf16)$ ]] || fail E_CONFIG 'KV must be fp8 or bf16 (--kv-cache).'
  [[ "$KV_PAGES" =~ ^(on|off|auto)$ ]] || fail E_CONFIG 'KV_PAGES must be on, off or auto (TF_KV_PAGES).'
  [[ "$CE_EXCHANGE" =~ ^[01]$ && "$ABSORB_MMA" =~ ^[01]$ && "$W8A8" =~ ^[01]$ ]] || fail E_CONFIG 'CE_EXCHANGE, ABSORB_MMA and W8A8 must be 0 or 1.'
  if [[ "$DENSE" == fp8 ]]; then
    warn W_DENSE 'The native engine has no FP8 dense weights: DENSE=fp8 is served as DENSE=q8 (8-bit codes, groups of 64).'
    DENSE=q8
  fi
  for key in DENSE HEAD; do
    [[ "${!key}" =~ ^(bf16|q8|q6|q4)$ ]] || fail E_CONFIG "$key must be bf16, q8, q6 or q4 (TF_GLM_${key}_BITS 16, 8, 6 or 4)."
  done
  for key in MAX_TOKENS PORT NCCL_PORT LOG_KEEP WAIT_TIMEOUT SMOKE_TIMEOUT STOP_TIMEOUT \
             MIN_AVAIL_GIB IMAGE_FREE_GIB MODEL_FREE_GIB DFLASH2_FREE_GIB CONTEXT_MAX SM NCCL_CHANNELS ROCE_MAX_KB; do
    [[ "${!key}" =~ ^[1-9][0-9]*$ ]] || fail E_CONFIG "$key must be a positive integer."
  done
  [[ "$PARALLEL" == auto || ( "$PARALLEL" =~ ^[1-9][0-9]*$ && PARALLEL -le 64 ) ]] || fail E_CONFIG 'PARALLEL must be auto or 1-64.'
  (( CONTEXT_MAX <= 1048576 )) || fail E_CONFIG 'CONTEXT_MAX must be at most 1048576 (the model window).'
  [[ "$CONTEXT" == auto || ( "$CONTEXT" =~ ^[1-9][0-9]*$ && CONTEXT -le 1048576 ) ]] || fail E_CONFIG 'CONTEXT must be auto or a token count from 1 to 1048576.'
  [[ "$CONTEXT_RETRY" =~ ^[01]$ ]] || fail E_CONFIG 'CONTEXT_RETRY must be 0 or 1.'
  is_number "$CONTEXT_FIT_MARGIN" && python3 -c 'import sys; sys.exit(0 if 0 < float(sys.argv[1]) <= 1 else 1)' "$CONTEXT_FIT_MARGIN" \
    || fail E_CONFIG 'CONTEXT_FIT_MARGIN must be a fraction in (0, 1].'
  (( PORT <= 65535 && NCCL_PORT < 65535 && PORT != NCCL_PORT && PORT != NCCL_PORT + 1 )) \
    || fail E_CONFIG 'PORT and NCCL_PORT must be distinct ports (the ranks also use NCCL_PORT+1).'
  [[ "$VISION_MAX_IMAGES" =~ ^[1-9][0-9]*$ && VISION_MAX_IMAGES -le 64 ]] || fail E_CONFIG 'VISION_MAX_IMAGES must be 1-64.'
  for key in THINKING VISION CLEAR_THINKING; do [[ "${!key}" =~ ^[01]$ ]] || fail E_CONFIG "$key must be 0 or 1."; done
  [[ -z "$REASONING_EFFORT" || "$REASONING_EFFORT" =~ ^(low|medium|high|xhigh)$ ]] || fail E_CONFIG 'REASONING_EFFORT must be low, medium, high, xhigh or empty.'
  [[ "$DRAFTER_BITS" =~ ^[04]$ ]] || fail E_CONFIG 'DRAFTER_BITS must be 4 or 0 (bf16).'
  [[ -z "$MTP_DRAFTS" || ( "$MTP_DRAFTS" =~ ^[0-9]+$ && MTP_DRAFTS -le 64 ) ]] || fail E_CONFIG 'MTP_DRAFTS must be empty or 0-64.'
  [[ -z "$MTP_CONFIDENCE" ]] || { is_number "$MTP_CONFIDENCE" && python3 -c 'import sys; sys.exit(0 if 0 <= float(sys.argv[1]) <= 1 else 1)' "$MTP_CONFIDENCE"; } \
    || fail E_CONFIG 'MTP_CONFIDENCE must be empty or a probability in [0, 1].'
  # GB10 is a unified-memory device, where the engine's own reserve floor is 2 GiB (cuda_memory.zig's floor(unified)).
  is_number "$MEMORY_RESERVE_GIB" && at_least "$MEMORY_RESERVE_GIB" 2 \
    || fail E_CONFIG 'MEMORY_RESERVE_GIB must be a number of GiB, at least 2 (the engine floor on a unified-memory device).'
  [[ -z "$HOST_CACHE_GIB" || "$HOST_CACHE_GIB" == auto ]] || is_number "$HOST_CACHE_GIB" \
    || fail E_CONFIG 'HOST_CACHE_GIB must be a number of GiB, 0 (off), auto or empty (TF_GLM_HOST_CACHE_GIB).'
  [[ "$ROCE" =~ ^(on|off|auto)$ ]] || fail E_CONFIG 'ROCE must be on, off or auto (TF_ROCE).'
  [[ "$GPU_RUNTIME" =~ ^(auto|gpus|runtime)$ ]] || fail E_CONFIG 'GPU_RUNTIME must be auto, gpus or runtime.'
  [[ "$FABRIC_MIN_SPEED_MBPS" =~ ^[0-9]+$ ]] || fail E_CONFIG 'FABRIC_MIN_SPEED_MBPS must be a link speed in Mb/s (0: no floor).'
  [[ "$MEMGUARD" =~ ^[01]$ ]] || fail E_CONFIG 'MEMGUARD must be 0 or 1.'
  is_number "$MEMGUARD_FLOOR_GIB" || fail E_CONFIG 'MEMGUARD_FLOOR_GIB must be a number of GiB.'
  for key in MEMGUARD_HOLD_S MEMGUARD_INTERVAL_S; do
    [[ "${!key}" =~ ^[1-9][0-9]*$ ]] || fail E_CONFIG "$key must be a positive integer."
  done
  [[ "$WORKER_WEIGHTS" =~ ^(rsync|nfs|check)$ ]] || fail E_CONFIG 'WORKER_WEIGHTS must be rsync, nfs or check.'
  [[ -z "$PROMPT_CACHE_GIB" ]] || is_number "$PROMPT_CACHE_GIB" || fail E_CONFIG 'PROMPT_CACHE_GIB must be empty or a number of GiB (0: none).'
  [[ "$CONFIG_OVERLAY" =~ ^(auto|0)$ ]] || fail E_CONFIG 'CONFIG_OVERLAY must be auto or 0.'
  if [[ "${GLM_TP4:-0}" == 1 ]]; then
    [[ "$TP" == 4 ]] || fail E_CONFIG "TP=$TP: ./start-tp4.sh runs four Sparks (TP=4); ./start.sh runs two."
    for key in WORKER WORKER2 WORKER3; do
      [[ -n "${!key}" ]] || fail E_CONFIG "$key is empty: set WORKER, WORKER2 and WORKER3 to the other three Sparks' ssh targets, in the ring's order from this Spark, in scripts/local.sh."
      [[ "${!key}" =~ ^[A-Za-z0-9_.@:-]+$ ]] || fail E_CONFIG "$key must be an ssh target such as user@host, with no spaces."
    done
    [[ "$WORKER" != "$WORKER2" && "$WORKER" != "$WORKER3" && "$WORKER2" != "$WORKER3" ]] || fail E_CONFIG 'WORKER, WORKER2 and WORKER3 must name three different Sparks.'
    [[ "$PREFILL_REDUCE" =~ ^(gather|ring)$ ]] || fail E_CONFIG 'PREFILL_REDUCE must be gather or ring (TF_GLM_PREFILL_REDUCE on a ring; split cannot run there).'
    [[ "$RING_NCCL_ALGO" =~ ^[A-Za-z,^]+$ ]] || fail E_CONFIG 'RING_NCCL_ALGO must be an NCCL_ALGO value (Ring).'
    for key in MASTER_ADDR SOCKET_IFNAME NFS_SERVER1 NFS_SERVER2 NFS_SERVER3; do
      [[ -z "${!key}" || "${!key}" =~ ^[A-Za-z0-9.:_-]+$ ]] || fail E_CONFIG "$key must be an address, host name or netdev."
    done
    for key in NFS_MODEL_PATH NFS_DFLASH2_PATH; do
      [[ -z "${!key}" || ( "${!key}" == /* && "${!key}" != *:* && "${!key}" != *,* ) ]] || fail E_CONFIG "$key must be an absolute path without colons or commas."
    done
    [[ "$WORKER_WEIGHTS" =~ ^(nfs|check)$ ]] || fail E_CONFIG 'WORKER_WEIGHTS must be nfs or check at four Sparks (rsync is not offered there).'
  else
    [[ "$TP" == 2 ]] || fail E_CONFIG "TP=$TP: ./start.sh wires two Sparks; four cabled as a ring start with ./start-tp4.sh (README \"4 Sparks\")."
    [[ -n "$WORKER" ]] || fail E_CONFIG 'WORKER is empty: set it to the second Spark'"'"'s ssh target (e.g. WORKER=zurih@10.0.34.4) in scripts/local.sh.'
    [[ "$WORKER" =~ ^[A-Za-z0-9_.@:-]+$ ]] || fail E_CONFIG 'WORKER must be an ssh target such as user@host, with no spaces.'
  fi
  for key in FABRIC_PEER FABRIC_SELF NFS_SERVER; do
    [[ -z "${!key}" || "${!key}" =~ ^[A-Za-z0-9.:_-]+$ ]] || fail E_CONFIG "$key must be an address or host name."
  done
  [[ -z "$TF_ROCE_HCA" || "$TF_ROCE_HCA" =~ ^[A-Za-z0-9_]+(,[A-Za-z0-9_]+)*$ ]] || fail E_CONFIG 'TF_ROCE_HCA must be comma-separated RDMA device names.'
  [[ -z "$NCCL_IB_GID_INDEX" || "$NCCL_IB_GID_INDEX" =~ ^[0-9]+$ ]] || fail E_CONFIG 'NCCL_IB_GID_INDEX must be a number.'
  [[ -z "$NCCL_SOCKET_IFNAME" || "$NCCL_SOCKET_IFNAME" =~ ^[A-Za-z0-9_.-]+(,[A-Za-z0-9_.-]+)*$ ]] || fail E_CONFIG 'NCCL_SOCKET_IFNAME must be a netdev name or a comma-separated list of them.'
  [[ "$CONTAINER_NAME" =~ ^[a-zA-Z0-9][a-zA-Z0-9_.-]*$ ]] || fail E_CONFIG 'CONTAINER_NAME contains unsupported characters.'
  [[ "$CONTAINER_NAME" != tensorfold-zig-r0 && "$CONTAINER_NAME" != tensorfold-zig-r1 && "$CONTAINER_NAME" != dsv41-1017-exl3-tf ]] \
    || fail E_CONFIG "$CONTAINER_NAME is a DeepSeek recipe's container on these Sparks; give this one another CONTAINER_NAME."
  [[ -z "$CONTAINER_USER" || "$CONTAINER_USER" =~ ^[a-zA-Z0-9_][a-zA-Z0-9_.-]*(:[a-zA-Z0-9_][a-zA-Z0-9_.-]*)?$ ]] || fail E_CONFIG 'CONTAINER_USER must be UID[:GID], a user name, root or empty.'
  [[ "$SERVED_NAME" =~ ^[^[:space:]]+$ ]] || fail E_CONFIG 'SERVED_NAME must be one word.'
  [[ "$HOST" =~ ^[A-Za-z0-9.:_-]+$ ]] || fail E_CONFIG 'HOST must be an address or host name.'
  [[ "$IMAGE" == '' || "$IMAGE" =~ ^[a-z0-9][a-z0-9._/:@-]*$ ]] || fail E_CONFIG 'IMAGE is not a valid image reference.'
  [[ "$TF_BIN" == /* ]] || fail E_CONFIG 'TF_BIN must be an absolute path inside the image.'
  for key in DATA_DIR MODEL_DIR DFLASH2_DIR CACHE_DIR STATE_DIR LOG_DIR WORKER_MODEL_DIR WORKER_DFLASH2_DIR WORKER_CACHE_DIR WORKER_STATE_DIR WORKER_OVERLAY; do
    path=${!key}
    [[ "$path" == /* && "$path" != *:* && "$path" != *$'\n'* && "$path" != *,* ]] || fail E_CONFIG "$key must be an absolute path without colons, commas or newlines."
  done
  [[ "$ABLIT" =~ ^[01]$ ]] || fail E_CONFIG 'ABLIT must be 0 or 1.'
  [[ "$ABLIT" == 0 || "$MODEL_SOURCE" == download ]] || fail E_CONFIG 'ABLIT=1 fetches the Ablit weights, so it needs MODEL_SOURCE=download. To serve a copy you already have, set ABLIT=0, MODEL_SOURCE=local and MODEL_DIR.'
  [[ "$MODEL_SOURCE" != download || -n "$MODEL_REVISION" ]] || fail E_CONFIG "MODEL_ID=$MODEL_ID has no pinned revision: set MODEL_REVISION."
  # A download never overwrites another checkpoint or revision: their files share names, so the result would be a mix.
  if [[ "$MODEL_SOURCE" == download && -f "$MODEL_DIR/.revision" && "$(cat "$MODEL_DIR/.revision")" != "$MODEL_REVISION" ]]; then
    fail E_CONFIG "MODEL_DIR=$MODEL_DIR holds revision $(cat "$MODEL_DIR/.revision"), not $MODEL_ID @ $MODEL_REVISION. Point MODEL_DIR at another directory (unset it to use the default for this checkpoint)."
  fi
  # A download never writes into a checkpoint it did not fetch (a local quant named by MODEL_DIR in local.sh). An
  # interrupted download leaves .downloading, so the next start resumes it.
  if [[ "$MODEL_SOURCE" == download && -e "$MODEL_DIR" && ! -f "$MODEL_DIR/.revision" && ! -f "$MODEL_DIR/.downloading" ]] && [[ -n "$(ls -A -- "$MODEL_DIR" 2>/dev/null)" ]]; then
    fail E_CONFIG "MODEL_SOURCE=download would write into MODEL_DIR=$MODEL_DIR, which holds files this recipe did not download. Point MODEL_DIR at an empty or recipe-made directory (default: $DATA_DIR/model)."
  fi
  if [[ "$DRAFTER" == none && ( -n "$MTP_DRAFTS" || -n "$MTP_CONFIDENCE" ) ]]; then
    warn W_CONFIG 'DRAFTER=none serves without drafts: MTP_DRAFTS and MTP_CONFIDENCE are not passed.'
  fi
}

# The image tag and its tf.patches label: docker/Dockerfile, the patches/engine/*.patch files (sorted by name; the
# build context holds nothing else) and the pins. Editing scripts or docs never forces a rebuild.
image_inputs() {
  (cd "$REPO_ROOT" && { [[ ! -f docker/Dockerfile ]] || printf 'docker/Dockerfile\0'
    find patches/engine -maxdepth 1 -type f -name '*.patch' -print0 2>/dev/null | LC_ALL=C sort -z; } | tr '\0' '\n')
}
set_image() {
  local f
  mapfile -t IMAGE_FILES < <(image_inputs)
  PATCH_COUNT=$(printf '%s\n' "${IMAGE_FILES[@]}" | grep -c '^patches/engine/' || true)
  PATCHES_HASH=$({
    printf 'TF_REPO=%s\nTF_REF=%s\nZIG_VERSION=%s\nZIG_SHA256=%s\nSM=%s\nZIG_CPU=%s\nNCCL_WHEEL=%s\n' "$TF_REPO" "$TF_REF" "$ZIG_VERSION" "$ZIG_SHA256" "$SM" "$ZIG_CPU" "$NCCL_WHEEL"
    for f in "${IMAGE_FILES[@]}"; do [[ -n "$f" ]] || continue; printf '\0%s\0' "$f"; cat -- "$REPO_ROOT/$f"; done
  } | sha256sum | cut -c1-12)
  IMAGE=${IMAGE:-$IMAGE_REPO:$PATCHES_HASH}
}
image_ready() { [[ "$(docker image inspect -f '{{index .Config.Labels "tf.patches"}}' "$IMAGE" 2>/dev/null || true)" == "$PATCHES_HASH" ]]; }

# Docker can hand a container the GPU. Two mechanisms are in the field and both are fine: an `nvidia` runtime
# registered with the daemon (JetPack on a Jetson AGX Thor, where `--gpus` is refused outright: "invoking the NVIDIA
# Container Runtime Hook directly ... is not supported"), or the NVIDIA Container Toolkit's prestart hook, which
# `--gpus` uses with plain runc -- what DGX OS ships on a Spark, where `docker info` lists no nvidia runtime at all.
# Pass "worker" to ask the worker. GPU_RUNTIME picks one; auto prefers the registered runtime and falls back to
# `--gpus`, so each node of a mixed pair gets the mechanism it supports.
gpu_mode() {
  local where=${1:-} runtimes
  if [[ "$where" == worker ]]; then
    runtimes=$(wssh docker info --format '{{json .Runtimes}}' 2>/dev/null || true)
  else
    runtimes=$(docker info --format '{{json .Runtimes}}' 2>/dev/null || true)
  fi
  case "$GPU_RUNTIME" in
    runtime) echo runtime ;;
    gpus) echo gpus ;;
    *) [[ "$runtimes" == *'"nvidia"'* ]] && echo runtime || echo gpus ;;
  esac
}

# gpu_args <local|worker>: the docker flags that give a container this node's GPU.
gpu_args() {
  case "$(gpu_mode "$1")" in
    runtime) printf '%s\n' --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all ;;
    *) printf '%s\n' --gpus all ;;
  esac
  cuda_compat_args "$1"
}

# The CUDA version a node's driver reports ("13.2" on JetPack 7.2 for a Thor).
driver_cuda() {
  local s
  if [[ "$1" == worker ]]; then s=$(wssh nvidia-smi 2>/dev/null || true); else s=$(nvidia-smi 2>/dev/null || true); fi
  sed -n 's/.*CUDA Version: *\([0-9][0-9.]*\).*/\1/p' <<< "$s" | head -1
}

# The CUDA floor an image asks the container runtime for (NVIDIA_REQUIRE_CUDA, e.g. "cuda>=13.3 brand=... driver>=595").
image_cuda_require() {
  local env
  if [[ "$1" == worker ]]; then env=$(wssh docker image inspect -f '{{json .Config.Env}}' "$IMAGE" 2>/dev/null || true)
  else env=$(docker image inspect -f '{{json .Config.Env}}' "$IMAGE" 2>/dev/null || true); fi
  python3 -c 'import json, sys
try: env = json.loads(sys.argv[1]) or []
except Exception: sys.exit(0)
print(next((e.split("=", 1)[1] for e in env if e.startswith("NVIDIA_REQUIRE_CUDA=")), ""))' "$env"
}

# cuda_compat_args <local|worker>: the NVIDIA container runtime refuses an image whose NVIDIA_REQUIRE_CUDA floor is
# newer than the driver's CUDA version -- JetPack 7.2 on a Thor reports CUDA 13.2 (driver 595.78) against an image
# built on nvidia/cuda:13.3.1, which asks for 13.3. CUDA 13.x minor version compatibility makes that run supported;
# the toolkit's floor is the only thing in the way, so it is moved to the driver's own version and the image's other
# requirements are kept. Nothing is passed when the driver is new enough.
cuda_compat_args() {
  local req drv out
  req=$(image_cuda_require "$1"); drv=$(driver_cuda "$1")
  [[ -n "$req" && -n "$drv" ]] || return 0
  out=$(python3 - "$req" "$drv" <<'PY'
import re, sys
req, drv = sys.argv[1], sys.argv[2]
m = re.search(r"cuda>=([0-9]+(?:\.[0-9]+)?)", req)
if not m: sys.exit(0)
key = lambda v: tuple(int(x) for x in v.split("."))
if key(drv) >= key(m.group(1)): sys.exit(0)
print(re.sub(r"cuda>=[0-9.]+", "cuda>=" + drv, req, count=1))
PY
  ) || true
  [[ -z "$out" ]] || printf '%s\n' -e "NVIDIA_REQUIRE_CUDA=$out"
}

nvidia_docker_ready() {
  local where=${1:-} runtimes
  if [[ "$where" == worker ]]; then
    runtimes=$(wssh docker info --format '{{json .Runtimes}}' 2>/dev/null || true)
    [[ "$runtimes" != *'"nvidia"'* ]] || return 0
    wssh test -x /usr/bin/nvidia-container-runtime-hook >/dev/null 2>&1 && return 0
    wssh test -x /usr/bin/nvidia-ctk >/dev/null 2>&1 && return 0
    return 1
  fi
  runtimes=$(docker info --format '{{json .Runtimes}}' 2>/dev/null || true)
  [[ "$runtimes" != *'"nvidia"'* ]] || return 0
  [[ ! -x /usr/bin/nvidia-container-runtime-hook ]] || return 0
  [[ ! -x /usr/bin/nvidia-ctk ]] || return 0
  return 1
}

check_platform() {
  local cmd
  [[ "$(uname -s)" == Linux && "$(uname -m)" == aarch64 ]] || fail E_PLATFORM 'This recipe requires Linux aarch64 (DGX Spark GB10 or Jetson AGX Thor).'
  for cmd in docker curl python3 flock gzip tar sha256sum nvidia-smi ssh rsync ip awk; do
    command -v "$cmd" >/dev/null || fail E_DEPENDENCY "Missing command: $cmd. See README requirements."
  done
  docker info >/dev/null 2>&1 || fail E_DOCKER 'Cannot talk to Docker. Start the daemon and check socket permissions.'
  if (( EUID != 0 )) && ! id -nG | tr ' ' '\n' | grep -qx docker; then
    fail E_DOCKER_GROUP 'Your login is not in the docker group. Add it and start a new login session.'
  fi
  nvidia_docker_ready '' || fail E_NVIDIA_RUNTIME 'Docker cannot give a container the GPU: no nvidia runtime registered and no NVIDIA Container Toolkit hook (nvidia-container-runtime-hook / nvidia-ctk). Install NVIDIA Container Toolkit.'
  log '[I_PLATFORM] Linux aarch64, Docker access and GPU passthrough are ready.'
  worker_reachable || fail E_WORKER "Cannot reach $WORKER over ssh with $SSH_OPTS. Set up key-based login (ssh-copy-id) and check WORKER."
  local wname wmachine
  wmachine=$(wssh uname -m 2>/dev/null || true)
  [[ "$wmachine" == aarch64 ]] || fail E_WORKER "$(worker_host) reports $wmachine, not aarch64: both ranks run the same aarch64 image."
  wssh docker info >/dev/null 2>&1 || fail E_WORKER "$(worker_host) cannot talk to Docker (is your login there in the docker group?)."
  nvidia_docker_ready worker || fail E_WORKER "$(worker_host) cannot give a container the GPU: no nvidia runtime registered and no NVIDIA Container Toolkit hook. Install NVIDIA Container Toolkit there."
  wname=$(wssh hostname 2>/dev/null || echo "$(worker_host)")
  log "[I_WORKER] $wname reachable over ssh, aarch64, Docker with the NVIDIA runtime."
  worker_home_defaults
}

# The state defaults live under this Spark's home (config.sh's DATA_DIR). On a pair whose logins differ, that home is
# not the worker's, and the worker cannot create the path (issue 3: acer-spark here, nvidia there). Re-root the
# untouched defaults under the login that runs there. An explicit WORKER_CACHE_DIR, DATA_DIR or CACHE_DIR wins.
worker_home_defaults() {
  local whome old=$WORKER_CACHE_DIR
  whome=$(wsh 'printf "%s\n" "${HOME:-}"' 2>/dev/null) || return 0
  [[ -n "$whome" && "$whome" != "$HOME" ]] || return 0
  [[ "$DATA_DIR" == "${XDG_CACHE_HOME:-$HOME/.cache}/aevonix-glm53" && "$CACHE_DIR" == "$DATA_DIR/zig-home" \
    && "$WORKER_CACHE_DIR" == "$CACHE_DIR" && "$WORKER_CACHE_DIR" == "$HOME"/* ]] || return 0
  WORKER_CACHE_DIR="$whome${WORKER_CACHE_DIR#"$HOME"}"
  [[ "$WORKER_STATE_DIR" == "$old/state" ]] && WORKER_STATE_DIR="$WORKER_CACHE_DIR/state"
  [[ "$WORKER_OVERLAY" == "$old/state/model-config.json" ]] && WORKER_OVERLAY="$WORKER_CACHE_DIR/state/model-config.json"
  log "[I_WORKER] The logins differ ($whome there, $HOME here): the worker's state defaults to $WORKER_CACHE_DIR."
}

# One GPU a node. nvidia-smi on a unified-memory device (GB10, Thor) answers [N/A] for memory.total, memory.used and
# memory.free and lists no compute apps -- the GPU's memory *is* the host's -- so there is nothing to check on the
# device side beyond that a supported GPU is there and the driver answers. What another workload holds shows up in
# check_memory instead.
check_gpu() {
  local names wnames
  names=$(nvidia-smi -L 2>/dev/null) || fail E_GPU 'Cannot list GPUs. Check the driver.'
  [[ "$(grep -c '^GPU' <<< "$names")" == 1 ]] || warn W_GPU "This node lists $(grep -c '^GPU' <<< "$names") GPUs; rank 0 opens the first."
  grep -Eq 'GB10|Thor' <<< "$names" || warn W_GPU "This node's GPU is not a GB10 or a Thor: $(head -1 <<< "$names")."
  log "[I_GPU] rank 0: $(sed -n '1s/ (UUID.*//p' <<< "$names") (sm_$SM)."
  wnames=$(wssh nvidia-smi -L 2>/dev/null) || fail E_GPU "Cannot list GPUs on $(worker_host). Check its driver."
  grep -Eq 'GB10|Thor' <<< "$wnames" || warn W_GPU "$(worker_host)'s GPU is not a GB10 or a Thor: $(head -1 <<< "$wnames")."
  log "[I_GPU] rank 1: $(sed -n '1s/ (UUID.*//p' <<< "$wnames") on $(worker_host)."
}

# Each selected GPU needs MIN_GPU_FREE_MIB free and less than MAX_GPU_USED_MIB in use. A desktop's own programs
# (the compositor, a browser) register as small compute processes on the display GPU: only larger ones count as busy.
check_gpus() {
  local gpu row free used name processes
  for gpu in "${GPU_IDS[@]}"; do
    row=$(nvidia-smi -i "$gpu" --query-gpu=name,memory.free,memory.used --format=csv,noheader,nounits) || fail E_GPU "Cannot query GPU $gpu. Check the driver and the GPUS selection."
    IFS=, read -r name free used <<< "$row"
    free=${free//[[:space:]]/}; used=${used//[[:space:]]/}
    [[ "$free" =~ ^[0-9]+$ && "$used" =~ ^[0-9]+$ ]] || fail E_GPU "GPU $gpu returned invalid memory information."
    (( free >= MIN_GPU_FREE_MIB )) || fail E_GPU_MEMORY "GPU $gpu ($name) has $free MiB free; at least $MIN_GPU_FREE_MIB MiB is required. Stop other GPU workloads (another TensorFold server?)."
    (( used < MAX_GPU_USED_MIB )) || fail E_GPU_BUSY "GPU $gpu uses $used MiB (MAX_GPU_USED_MIB=$MAX_GPU_USED_MIB). Stop other GPU workloads before launch."
    log "[I_GPU] GPU $gpu:$name, $free MiB free, $used MiB in use."
  done
  processes=$(nvidia-smi -i "$GPUS" --query-compute-apps=pid,used_memory --format=csv,noheader,nounits) || fail E_GPU 'Cannot list GPU compute processes.'
  processes=$(awk -F', *' -v max="$MAX_GPU_USED_MIB" '$2 ~ /^[0-9]+$/ && $2 + 0 >= max {print $1}' <<< "$processes")
  if [[ "$processes" =~ [0-9] ]]; then fail E_GPU_BUSY "A selected GPU has a compute process (PID ${processes//$'\n'/, }) of $MAX_GPU_USED_MIB MiB or more. Stop it before launch."; fi
}

# Memory on both Sparks, counted the way the engine's startup budget counts it on a unified-memory device
# (scripts/memavail.sh). The model is about 82 GiB a rank; MEMORY_RESERVE_GIB and the KV pool come on top, so a Spark
# still running another engine fails here with a figure that says so.
check_memory() {
  local mine parked theirs total
  read -r mine parked <<< "$(mem_avail_gb)"
  total=$(mem_total_gb)
  [[ -n "$mine" ]] || { warn W_RAM 'Cannot read MemAvailable on this Spark.'; return 0; }
  log "[I_RAM] rank 0: $mine GiB available of $total GiB${parked:+ (${parked} GiB of it parked on the per-CPU page lists)}."
  at_least "$mine" "$MIN_AVAIL_GIB" \
    || fail E_MEMORY "This Spark has $mine GiB available; MIN_AVAIL_GIB=$MIN_AVAIL_GIB is required. Stop whatever else holds its memory (another TensorFold rank?) and try again."
  theirs=$(worker_avail_gib)
  if [[ -z "$theirs" ]]; then
    warn W_RAM "Cannot read $(worker_host)'s available memory; the start goes on."
    return 0
  fi
  log "[I_RAM] rank 1: $theirs GiB available on $(worker_host)."
  at_least "$theirs" "$MIN_AVAIL_GIB" \
    || fail E_MEMORY "$(worker_host) has $theirs GiB available; MIN_AVAIL_GIB=$MIN_AVAIL_GIB is required. Stop whatever else holds its memory and try again."
}

check_disk() {
  local path ancestor device free need label docker_root i
  local -a paths=() sizes=() labels=()
  if [[ "$MODEL_SOURCE" == download ]] && ! weights_ready "$MODEL_DIR" "$MODEL_REVISION"; then
    need=${MODEL_FREE_GIB%.*}
    # Resuming: what an interrupted download already holds doesn't have to fit again.
    if [[ -f "$MODEL_DIR/.downloading" ]]; then
      need=$(( need - $(du -s --block-size=1G -- "$MODEL_DIR" | cut -f1) )); (( need > 0 )) || need=1
    fi
    paths+=("$MODEL_DIR"); sizes+=("$need"); labels+=(checkpoint)
  fi
  if [[ "$DRAFTER" == dflash2 ]] && ! drafter_ready; then paths+=("$DFLASH2_DIR"); sizes+=("$DFLASH2_FREE_GIB"); labels+=(DFlash2); fi
  if ! image_ready; then
    docker_root=$(docker info --format '{{.DockerRootDir}}') || fail E_DISK 'Cannot locate Docker storage.'
    paths+=("$docker_root"); sizes+=("$IMAGE_FREE_GIB"); labels+=(image)
  fi
  local -A need_by_device=() free_by_device=() names_by_device=()
  for (( i=0; i<${#paths[@]}; i++ )); do
    path=${paths[i]}; need=${sizes[i]}; label=${labels[i]}; ancestor=$path
    while [[ ! -e "$ancestor" ]]; do ancestor=$(dirname -- "$ancestor"); done
    read -r device free < <(df -Pk -- "$ancestor" | awk 'END {print $1, $4}')
    [[ "$free" =~ ^[0-9]+$ ]] || fail E_DISK "Cannot inspect available disk space for $label."
    need_by_device[$device]=$(( ${need_by_device[$device]:-0} + need * 1024 * 1024 ))
    free_by_device[$device]=$free
    names_by_device[$device]="${names_by_device[$device]:-} $label"
  done
  for device in "${!need_by_device[@]}"; do
    (( free_by_device[$device] >= need_by_device[$device] )) || fail E_DISK "Need $((need_by_device[$device]/1024/1024)) GiB free for${names_by_device[$device]}; only $((free_by_device[$device]/1024/1024)) GiB available on that filesystem."
  done
  log '[I_DISK] Enough disk space on this Spark for what is still missing.'
  check_worker_disk
}

# The worker needs room for the image it loads, and for the checkpoint when it is sent over the link.
check_worker_disk() {
  local need free root
  need=0
  if ! wssh docker image inspect "$IMAGE" >/dev/null 2>&1; then need=$(( need + IMAGE_FREE_GIB )); fi
  if [[ "$WORKER_WEIGHTS" == rsync ]] && ! worker_weights_ready "$WORKER_MODEL_DIR" "$MODEL_REVISION"; then
    need=$(( need + ${MODEL_FREE_GIB%.*} ))
  fi
  (( need > 0 )) || { log "[I_DISK] $(worker_host) needs no new space (image and weights are in place)."; return 0; }
  root=$(wssh docker info --format '{{.DockerRootDir}}' 2>/dev/null || echo /var/lib/docker)
  free=$(worker_free_gib "$root")
  [[ "$free" =~ ^[0-9]+$ ]] || { warn W_DISK "Cannot read $(worker_host)'s free disk space; the start goes on."; return 0; }
  (( free >= need )) || fail E_DISK "$(worker_host) needs about $need GiB free (image and/or checkpoint); it has $free GiB on $root."
  log "[I_DISK] $(worker_host) has $free GiB free, about $need GiB is needed."
}

prepare_image() {
  local -a build_flags=()
  read -r -a build_flags <<< "$DOCKER_BUILD_FLAGS"
  if [[ ! -f "$REPO_ROOT/docker/Dockerfile" ]]; then
    [[ "$DRY_RUN" == 1 ]] || fail E_IMAGE 'docker/Dockerfile is missing: the image cannot be built.'
    warn W_DOCKERFILE 'docker/Dockerfile is missing; the plan shows the build it would run.'
  fi
  if [[ "$DRY_RUN" != 1 ]]; then
    if image_ready; then log "Image is ready: $IMAGE (tf.patches=$PATCHES_HASH)"; return; fi
    local other
    other=$(docker images -q --filter "label=tf.patches=$PATCHES_HASH" | head -n1)
    if [[ -n "$other" ]]; then
      docker tag "$other" "$IMAGE" || fail E_IMAGE "Cannot tag image $other as $IMAGE."
      log "Image is ready: $other tagged $IMAGE (tf.patches=$PATCHES_HASH)"; return
    fi
  else
    log "Skipped when an image labelled tf.patches=$PATCHES_HASH exists (docker image inspect $IMAGE):"
  fi
  log "Build $IMAGE: TensorFold ${TF_REF:0:12} with $PATCH_COUNT engine patches, Zig $ZIG_VERSION, sm_$SM."
  run docker build -f "$REPO_ROOT/docker/Dockerfile" -t "$IMAGE" --label "tf.patches=$PATCHES_HASH" \
    --build-arg "TF_REPO=$TF_REPO" --build-arg "TF_REF=$TF_REF" --build-arg "ZIG_VERSION=$ZIG_VERSION" \
    --build-arg "ZIG_SHA256=$ZIG_SHA256" --build-arg "SM=$SM" --build-arg "PATCHES_HASH=$PATCHES_HASH" \
    --build-arg "ZIG_CPU=$ZIG_CPU" ${NCCL_WHEEL:+--build-arg "NCCL_WHEEL=$NCCL_WHEEL"} \
    "${build_flags[@]}" "$REPO_ROOT" || fail E_IMAGE 'Image build failed. Check disk space, network access and the first build error.'
  run docker run --rm --network none --entrypoint "$TF_BIN" "$IMAGE" --version \
    || fail E_IMAGE "The built image has no working $TF_BIN."
}

# --- weights -------------------------------------------------------------------------------------------------------
weights_ready() { [[ -f "$1/config.json" && -f "$1/.revision" && "$(cat "$1/.revision")" == "$2" ]]; }
drafter_ready() { [[ -f "$DFLASH2_DIR/config.json" ]] && compgen -G "$DFLASH2_DIR/*.safetensors" >/dev/null; }

# Reads MODEL_DIR's config.json (read-only, also in a dry run): MODEL_DESC, and NEEDS_OVERLAY=1 when an EXL3
# quantization_config names no bits (a mixed-width quant such as the 3.5 bpw GLMQ one).
inspect_model() {
  NEEDS_OVERLAY=0; MODEL_DESC="$MODEL_DIR"
  if [[ ! -f "$MODEL_DIR/config.json" ]]; then
    [[ "$MODEL_SOURCE" == download ]] && { MODEL_DESC="$MODEL_ID@${MODEL_REVISION:0:12} (to download)"; return 0; }
    local why="MODEL_DIR has no config.json: $MODEL_DIR. Set MODEL_DIR to a GLM-5.3-Flash EXL3 checkpoint, or MODEL_SOURCE=download."
    [[ "$DRY_RUN" == 1 ]] || fail E_MODEL "$why"
    warn E_MODEL "$why"; return 0
  fi
  local out
  out=$(python3 - "$MODEL_DIR" <<'PY'
import glob, json, os, sys
d = sys.argv[1]
c = json.load(open(os.path.join(d, "config.json")))
mt = c.get("model_type")
if mt != "glm5_next":
    print(f"E_MODEL\tmodel_type {mt!r}: this recipe serves GLM-5.3-Flash (glm5_next)"); sys.exit()
q = c.get("quantization_config") or (c.get("text_config") or {}).get("quantization_config") or {}
if not glob.glob(os.path.join(d, "*.safetensors")):
    print("E_MODEL\tno *.safetensors files"); sys.exit()
if not os.path.exists(os.path.join(d, "tokenizer.json")):
    print("E_MODEL\tno tokenizer.json"); sys.exit()
method = q.get("quant_method", "none")
if method != "exl3":
    print(f"W_MODEL\t{method} checkpoint: the recipe is written for EXL3; served as stored\t0"); sys.exit()
bits = q.get("bits"); avg = q.get("avg_bits")
width = f"{bits}-bit" if bits is not None else (f"mixed {avg} bpw" if avg is not None else "unknown width")
vision = "vision" if c.get("vision_config") else "no vision tower"
print(f"OK\tEXL3 {width} experts, {q.get('codebook', '?')} codebook, {vision}\t{1 if bits is None else 0}")
PY
  ) || fail E_MODEL "Cannot read $MODEL_DIR/config.json."
  local kind desc overlay
  IFS=$'\t' read -r kind desc overlay <<< "$out"
  case "$kind" in
    OK) MODEL_DESC="$(basename -- "$MODEL_DIR"): $desc" ;;
    W_MODEL) warn W_MODEL "$desc"; MODEL_DESC="$(basename -- "$MODEL_DIR")" ;;
    *) fail E_MODEL "$MODEL_DIR: $desc." ;;
  esac
  [[ "$CONFIG_OVERLAY" == auto && "${overlay:-0}" == 1 ]] && NEEDS_OVERLAY=1
  return 0
}

# The overlay: config.json with quantization_config.bits 4 added, mounted read-only over /model/config.json.
# The engine reads each expert matrix's width from its trellis shape (with the engine patches); only the family's
# admission check reads bits. The checkpoint itself is never changed.
model_overlay() {
  OVERLAY_MOUNT=()
  (( NEEDS_OVERLAY == 1 )) || return 0
  local overlay="$STATE_DIR/model-config.json"
  if [[ "$DRY_RUN" != 1 ]]; then
    mkdir -p "$STATE_DIR"
    python3 - "$MODEL_DIR/config.json" "$overlay" <<'PY' || fail E_MODEL 'Cannot write the config.json overlay.'
import json, os, sys
c = json.load(open(sys.argv[1]))
q = c.get("quantization_config")
if q is None:
    q = c["text_config"]["quantization_config"]
q["bits"] = 4
text = json.dumps(c, indent=2) + "\n"
old = open(sys.argv[2]).read() if os.path.exists(sys.argv[2]) else None
if old != text:  # rewritten only when it changes: the engine's digest cache keeps the file's identity
    tmp = sys.argv[2] + ".tmp"
    open(tmp, "w").write(text)
    os.replace(tmp, sys.argv[2])
PY
  fi
  log "config.json overlay: quantization_config.bits 4 added for the engine's EXL3 check ($overlay -> /model/config.json, read-only)."
  OVERLAY_MOUNT=(-v "$overlay:/model/config.json:ro")
}

# Downloads use the host's hf CLI (huggingface_hub). HF_TOKEN, when set, is read by hf from the environment and never printed.
fetch_weights() {
  local repo=$1 rev=$2 dir=$3
  if weights_ready "$dir" "$rev"; then log "Weights are ready: $repo @ ${rev:0:12} ($dir)"; return; fi
  [[ "$DRY_RUN" == 1 ]] || command -v hf >/dev/null || fail E_DEPENDENCY 'Missing command: hf (python3 -m pip install -U "huggingface_hub[cli]").'
  [[ "$DRY_RUN" == 1 ]] || { mkdir -p "$dir"; printf '%s\n' "$rev" > "$dir/.downloading"; }
  log "Download $repo at pinned revision $rev into $dir."
  run hf download "$repo" --revision "$rev" --local-dir "$dir" || fail E_DOWNLOAD 'Download failed. Check free space, access and HF_TOKEN; rerun to resume.'
  [[ "$DRY_RUN" == 1 ]] || { printf '%s\n' "$rev" > "$dir/.revision"; rm -f "$dir/.downloading"; }
}

# The gated Ablit weights need HF_TOKEN only while they still have to be fetched. One small file is asked for before the
# image work, so a refusal says why at once. The token goes to curl on stdin, never on a command line.
check_weights_access() {
  [[ "$MODEL_SOURCE" == download && "$MODEL_ID" == "$ABLIT_ID" ]] || return 0
  ! weights_ready "$MODEL_DIR" "$MODEL_REVISION" || return 0
  local steps code
  steps="1. Open https://huggingface.co/$MODEL_ID, log in and agree to its terms. 2. Create a token with read access at https://huggingface.co/settings/tokens (the same account). 3. Set HF_TOKEN=hf_... in scripts/local.sh, .env or the environment, and run this again. Or set ABLIT=0 to serve the published checkpoint."
  if [[ -z "${HF_TOKEN:-}" ]]; then
    [[ "$DRY_RUN" == 1 ]] && { warn W_HF_TOKEN "The Ablit weights are gated and HF_TOKEN is not set: a real start stops here."; return 0; }
    fail E_HF_TOKEN "The Ablit weights ($MODEL_ID) are gated on Hugging Face. $steps"
  fi
  [[ "$DRY_RUN" != 1 ]] || return 0
  code=$(printf 'Authorization: Bearer %s\n' "$HF_TOKEN" |
    curl -s -o /dev/null -w '%{http_code}' --max-time 20 -H @- "https://huggingface.co/$MODEL_ID/resolve/$MODEL_REVISION/config.json" || true)
  case "$code" in
    2??|3??) log "HF_TOKEN reaches the gated $MODEL_ID." ;;
    401) fail E_HF_TOKEN "Hugging Face does not accept HF_TOKEN (401). $steps" ;;
    403) fail E_HF_TOKEN "HF_TOKEN may not read $MODEL_ID yet (403): its account has not agreed to the terms, or a fine-grained token lacks read access to public gated repositories. $steps" ;;
    404) fail E_HF_TOKEN "Hugging Face has no $MODEL_ID @ $MODEL_REVISION (404). Check MODEL_REVISION." ;;
    *) warn W_HF_TOKEN "Could not check HF_TOKEN against $MODEL_ID (HTTP $code); trying the download." ;;
  esac
}

prepare_weights() {
  if [[ "$MODEL_SOURCE" == download ]]; then
    fetch_weights "$MODEL_ID" "$MODEL_REVISION" "$MODEL_DIR"
    [[ "$DRY_RUN" == 1 ]] || inspect_model
  else
    log "Checkpoint: $MODEL_DESC (local, mounted read-only; no download)."
  fi
  if [[ "$DRAFTER" == dflash2 ]]; then
    log 'DFlash2 drafter weights: CC BY-NC-ND 4.0, non-commercial. DRAFTER=mtp or DRAFTER=none serves without them.'
    if drafter_ready; then
      log "DFlash2 is ready: $DFLASH2_DIR"
      if [[ -f "$DFLASH2_DIR/.revision" && "$(cat "$DFLASH2_DIR/.revision")" != "$DFLASH2_REVISION" ]]; then
        warn W_DFLASH2 "$DFLASH2_DIR holds revision $(cat "$DFLASH2_DIR/.revision"), not the pinned $DFLASH2_REVISION."
      fi
    else
      fetch_weights "$DFLASH2_ID" "$DFLASH2_REVISION" "$DFLASH2_DIR"
    fi
  fi
}

# The link between the nodes: which ports carry it, which RDMA devices and GID index NCCL and the engine's RoCE path
# will be given, and -- when neither node has an RDMA device at all -- that NCCL's socket transport carries every
# exchange over the rails instead. detect_link resolves the fabric kind and ROCE's auto value.
check_link() {
  if [[ "$DRY_RUN" == 1 ]]; then
    log 'Find the link: the directly reached ports on both nodes, the RDMA devices and the RoCE v2 IPv4 GID index when'
    log 'both expose one; probe every rail with a full-size unfragmentable ping before using it.'
    # Read-only probing, so the plan below prints the real addresses. Placeholders when the worker is out of reach.
    if [[ -n "$WORKER" ]] && worker_reachable; then
      detect_link
    else
      LINK_SELF="<this node's link address>"; LINK_PEER="<the worker's link address>"
      LINK_DEV='<link netdev>'; WORKER_DEV='<worker link netdev>'
      LINK_HCA='<rdma devices>'; WORKER_HCA='<worker rdma devices>'
      LINK_GID='<gid>'; WORKER_GID='<gid>'
      LINK_KIND=$(rdma_present && echo ib || echo tcp)
      warn W_FABRIC "Cannot reach $WORKER over ssh, so the plan shows placeholders for the link."
    fi
    log "Then NCCL_COMM_ID=$LINK_SELF:$NCCL_PORT and the per-rank link variables ($([[ "${LINK_KIND:-ib}" == ib ]] && echo 'NCCL_SOCKET_IFNAME/NCCL_IB_HCA/NCCL_IB_GID_INDEX' || echo 'NCCL_SOCKET_IFNAME/NCCL_IB_DISABLE')),"
    log "TF_ROCE=$([[ "$ROCE" == on ]] && echo 1 || echo 0), TF_ROCE_MAX_KB=$ROCE_MAX_KB."
    return 0
  fi
  detect_link
  if [[ "$LINK_KIND" == ib ]]; then
    if [[ "$ROCE" == on ]]; then
      log "[I_ROCE] Both nodes expose an RDMA device; the ranks will try the RoCE all-gather path (TF_ROCE_MAX_KB=$ROCE_MAX_KB) and fall back to NCCL together if either cannot bring it up."
    else
      log '[I_ROCE] ROCE=off: NCCL carries every exchange.'
    fi
  else
    log "[I_ROCE] No RDMA device on both nodes: NCCL's socket transport carries every exchange over $LINK_DEV <-> $WORKER_DEV (the engine's RoCE all-gathers are off)."
  fi
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  cd "$REPO_ROOT"
  mode=${1:-}
  case "$mode" in
    --label) resolve_arch; set_image; printf '%s\n' "$PATCHES_HASH"; exit 0 ;;  # resolve_arch: SM is in the hash
    ''|--image|--weights|--check) ;;
    -h|--help) sed -n '2,7s/^# //p' "$0"; exit 0 ;;
    *) printf 'Usage: scripts/prepare.sh [--image|--weights|--check|--label]\n' >&2; exit 2 ;;
  esac
  validate_config
  set_image
  [[ "$mode" == --image ]] || inspect_model
  if [[ "$DRY_RUN" == 1 ]]; then log '[I_DRY_RUN] Plan only: no Docker, hardware access, downloads or file changes.'
  else
    check_platform
    mkdir -p "$STATE_DIR"
    exec 8>"$STATE_DIR/start.lock"
    flock -n 8 || fail E_LOCK 'Another start, stop or preparation is in progress.'
    [[ "$mode" == --check ]] || check_disk
  fi
  case "$mode" in
    --image) prepare_image; ship_image ;;
    --weights) check_weights_access; prepare_weights; model_overlay; prepare_worker_weights ;;
    --check) [[ "$DRY_RUN" == 1 ]] || { check_gpu; check_memory; image_ready || fail E_IMAGE "Build the image first: scripts/prepare.sh --image ($IMAGE)."; }; check_link ;;
    *) check_weights_access; prepare_image; ship_image; prepare_weights; model_overlay; check_link; prepare_worker_weights ;;
  esac
  log 'Prepared. Start the API with ./start.sh.'
fi
