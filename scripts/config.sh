#!/usr/bin/env bash
# Settings in precedence order: exported environment, scripts/local.sh, .env, the defaults below.
# local.sh is trusted Bash. .env is read as literal KEY=value lines, never executed.
# Sourced by scripts/prepare.sh, start.sh and stop.sh; it runs no Docker, GPU, ssh or network command.
[[ "${_GLM_CONFIG_LOADED:-0}" == 1 ]] && return 0
_GLM_CONFIG_LOADED=1
REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
if [[ -f "$REPO_ROOT/scripts/local.sh" ]]; then
  declare -A _cfg_env=()
  while IFS= read -r _cfg_key; do _cfg_env[$_cfg_key]=${!_cfg_key}; done < <(compgen -e)
  # shellcheck source=/dev/null
  source "$REPO_ROOT/scripts/local.sh"
  for _cfg_key in "${!_cfg_env[@]}"; do export "$_cfg_key=${_cfg_env[$_cfg_key]}"; done
  unset _cfg_env
fi
if [[ -f "$REPO_ROOT/.env" ]]; then
  while IFS= read -r _cfg_line || [[ -n "$_cfg_line" ]]; do
    [[ "$_cfg_line" =~ ^[[:space:]]*(export[[:space:]]+)?([A-Za-z_][A-Za-z0-9_]*)=(.*)$ ]] || continue
    _cfg_key=${BASH_REMATCH[2]}; _cfg_value=${BASH_REMATCH[3]}
    if [[ "$_cfg_value" =~ ^\"([^\"]*)\"[[:space:]]*(#.*)?$ || "$_cfg_value" =~ ^\'([^\']*)\'[[:space:]]*(#.*)?$ ]]; then
      _cfg_value=${BASH_REMATCH[1]}
    else
      _cfg_value=${_cfg_value%%#*}; _cfg_value=${_cfg_value%"${_cfg_value##*[![:space:]]}"}
    fi
    [[ -n "${!_cfg_key+set}" ]] || export "$_cfg_key=$_cfg_value"
  done < "$REPO_ROOT/.env"
fi
unset _cfg_key _cfg_line _cfg_value

# --- pinned engine build (hashed into the image tag with docker/ and patches/engine/*.patch) ----------------------
TF_REPO="${TF_REPO:-https://github.com/ashhart/TensorFold.git}"
TF_REF="${TF_REF:-f350cd3ea253a042e205746928271f9d8f71a82d}"
ZIG_VERSION="${ZIG_VERSION:-0.17.0}"
# sha256 of zig-aarch64-linux-0.17.0.tar.xz (ziglang.org/download/index.json).
ZIG_SHA256="${ZIG_SHA256:-9e8d11661d4ae3bd57702a3832781e23ad151dde5798e16a5ccd503f65234ff8}"
SM="${SM:-}"                                      # empty/auto: the GPU's own compute capability (GB10 sm_121, Jetson AGX Thor sm_110)
ZIG_CPU="${ZIG_CPU:-native}"                     # the image is built on a node and runs on it; baseline to move it
NCCL_WHEEL="${NCCL_WHEEL:-}"                     # non-empty: install this nvidia-nccl-cu13 version over the base image's
IMAGE="${IMAGE:-}"                               # empty: tensorfold-glm53-spark:<PATCHES_HASH>
IMAGE_REPO="${IMAGE_REPO:-tensorfold-glm53-spark}"
# How Docker hands a container the GPU. gpus: `--gpus all` (the NVIDIA Container Toolkit's prestart hook, what DGX OS
# ships on a Spark). runtime: `--runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all` (JetPack on a Jetson AGX Thor, where
# `--gpus` refuses: "invoking the NVIDIA Container Runtime Hook directly ... is not supported"). auto: runtime when the
# daemon lists an `nvidia` runtime, else gpus. Probed per node: a mixed pair takes each node's own answer.
GPU_RUNTIME="${GPU_RUNTIME:-auto}"
DOCKER_BUILD_FLAGS="${DOCKER_BUILD_FLAGS:-}"     # extra docker build flags, word-split (e.g. --progress=plain)
TF_BIN="${TF_BIN:-/usr/local/bin/tensorfold-native}"
CONTAINER_NAME="${CONTAINER_NAME:-glm53-spark}"  # never a DeepSeek recipe's container on these Sparks
CONTAINER_USER="${CONTAINER_USER-$(id -u):$(id -g)}"   # empty or root: run the ranks as root

# --- the two Sparks ---------------------------------------------------------------------------------------------------
# This machine serves rank 0 and the HTTP API; WORKER (an ssh target, key-based, no password) runs rank 1.
# Both ranks run the same image and the same flags; before the weights load they compare a fingerprint of their
# settings, the checkpoint and the binary, and name the field that differs.
TP="${TP:-2}"                                    # this recipe wires two Sparks; the engine also does TP=3 (unqualified here)
WORKER="${WORKER:-}"                             # e.g. zurih@10.0.34.4 or spark4; set it in scripts/local.sh
# One multiplexed connection carries the many small calls to the worker (docker inspect, the memory samples): a new
# ssh per call would cost a handshake each time and fill its auth log.
SSH_OPTS="${SSH_OPTS:--o BatchMode=yes -o ConnectTimeout=10 -o ControlMaster=auto -o ControlPath=$HOME/.ssh/cm-glm53-%C -o ControlPersist=300}"
# The ranks' link. Empty: start.sh asks the route to WORKER for this Spark's own ConnectX-7 netdev, the worker's address
# on it, the RoCE devices of that port (a Spark's one QSFP port reaches the GB10 over two PCIe links, so it shows up as
# two netdevs and two RDMA devices; both are used) and the RoCE v2 GID index that holds each side's IPv4 address.
FABRIC_PEER="${FABRIC_PEER:-}"                   # the worker's address on the link (auto)
FABRIC_SELF="${FABRIC_SELF:-}"                   # this Spark's address on the link (auto); rank 0's NCCL_COMM_ID address
NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-}"     # this node's link netdev(s), comma separated (auto); the worker's is found on the worker
TF_ROCE_HCA="${TF_ROCE_HCA:-}"                   # RDMA devices for the link, comma separated, at most two (auto)
NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX:-}"       # the RoCE v2 IPv4 GID index (auto; it differs between hosts' tables)
# Without an RDMA device on both nodes (a Jetson AGX Thor pair: its 100G QSFP cage is plain Ethernet over MGBE, no
# RoCE) the link is TCP only: NCCL's socket transport over every rail, which start.sh finds as the directly reached
# ports above FABRIC_MIN_SPEED_MBPS. NCCL pairs a rank's interface with the peer's on the same subnet, so each rail
# carries its own channels.
FABRIC_MIN_SPEED_MBPS="${FABRIC_MIN_SPEED_MBPS:-10000}"   # a link rail's speed floor; keeps a 1G/5G LAN out of the fabric
NCCL_CHANNELS="${NCCL_CHANNELS:-4}"              # NCCL_MIN_NCHANNELS and NCCL_MAX_NCHANNELS
PORT="${PORT:-8020}"
HOST="${HOST:-0.0.0.0}"
NCCL_PORT="${NCCL_PORT:-29583}"                  # NCCL_COMM_ID=FABRIC_SELF:NCCL_PORT; the ranks' watch line uses NCCL_PORT+1
SERVED_NAME="${SERVED_NAME:-GLM-5.3-Flash}"
# ROCE=on: the engine carries the decode and verify all-gathers up to ROCE_MAX_KB as one-shot RDMA writes over the
# Sparks' link (zig/src/cuda/roce.zig; 10.4 us a call at 4 KiB a rank against NCCL's 12.3, 17.4 against 51.5 at 64 KiB,
# measured upstream on two GB10s over one ConnectX-7 cable). Larger collectives stay on NCCL, and the size alone
# decides, so both ranks always take the same path. Same bytes in the same places as NCCL: the same bits.
# off (TF_ROCE=0 on either rank): NCCL carries everything.
# auto: on when both nodes expose an RDMA device (/dev/infiniband, /sys/class/infiniband), else off -- a Thor pair has
# no verbs device at all, so the whole fabric is NCCL over TCP.
ROCE="${ROCE:-auto}"                            # auto: on when both nodes expose an RDMA device, else off
ROCE_MAX_KB="${ROCE_MAX_KB:-4096}"               # TF_ROCE_MAX_KB, the largest shard the RDMA region holds

# --- four Sparks (./start-tp4.sh, ./stop-tp4.sh; README "4 Sparks") -----------------------------------------------------
# TP=4 over four Sparks cabled as a ring without a switch: this Spark (rank 0, the API) and WORKER, WORKER2, WORKER3
# (ranks 1-3) in the ring's order, each cabled to the next and the last back to this one, one subnet per cable. The two
# Sparks across the ring from each other reach each other only through a neighbour (an address routed over the cables,
# or the LAN). start.sh never reads these; start-tp4.sh sets TP=4 itself.
WORKER2="${WORKER2:-}"                           # rank 2's ssh target, the Spark across the ring (e.g. zurih@10.0.0.3)
WORKER3="${WORKER3:-}"                           # rank 3's ssh target, cabled to this Spark (e.g. zurih@10.0.41.4)
# NCCL_COMM_ID's address at TP=4 and the ranks' lines to rank 0: an address of this Spark every worker reaches (auto:
# what this Spark's host name resolves to, else its first address). The Spark across the ring reaches it through a
# neighbour. SOCKET_IFNAME: NCCL's bootstrap netdev on every rank (auto: each Spark's default-route netdev, the LAN).
MASTER_ADDR="${MASTER_ADDR:-}"
SOCKET_IFNAME="${SOCKET_IFNAME:-}"
# TF_GLM_PREFILL_REDUCE at TP=4. The engine's default, split, sends every rank's prompt rows straight to every other
# rank, which a ring cannot carry (the engine probes and refuses it). gather: the exchanges as all-gathers between
# neighbours, no point-to-point sends (the qualified mode here); ring: split, its exchanges passed along the ring.
PREFILL_REDUCE="${PREFILL_REDUCE:-gather}"
RING_NCCL_ALGO="${RING_NCCL_ALGO:-Ring}"         # NCCL_ALGO on the ring: Ring sends only to the next rank
# The checkpoint and drafter paths as the NFS server on this Spark exports them, when they differ from MODEL_DIR and
# DFLASH2_DIR (an NFS server in a container exports its own view: /huggingface/... for /home/you/.cache/huggingface/...).
NFS_MODEL_PATH="${NFS_MODEL_PATH:-}"
NFS_DFLASH2_PATH="${NFS_DFLASH2_PATH:-}"
# NFS_SERVER1 / NFS_SERVER2 / NFS_SERVER3: the address each worker mounts from (auto: this Spark's address on its route
# to that worker; NFS_SERVER, when set, for all three).
NFS_SERVER1="${NFS_SERVER1:-}"; NFS_SERVER2="${NFS_SERVER2:-}"; NFS_SERVER3="${NFS_SERVER3:-}"

# --- weights ---------------------------------------------------------------------------------------------------------
# Both Sparks read the whole checkpoint at load: each rank keeps its half, but every rank reads every shard, so the
# worker needs the same bytes as this one. The ranks compare a fingerprint of the checkpoint before they load, so a
# different revision on the worker is refused rather than mixed.
#   WORKER_WEIGHTS=rsync  send this Spark's copy over the link when the worker's is missing or a different revision
#                         (resumable, ~164 GiB, about 5 minutes at 600 MB/s). The worker then loads from local disk.
#   WORKER_WEIGHTS=nfs    the worker mounts this Spark's copy over NFS, as a read-only Docker volume per directory,
#                         and no bytes are duplicated. GLM holds all its weights in memory once loaded, so NFS costs
#                         load time only, not serving speed. Needs an NFS server on this Spark exporting MODEL_DIR
#                         (and DFLASH2_DIR) to the worker's address on the link; ./start.sh checks the export and says
#                         what to add when it is missing.
#   WORKER_WEIGHTS=check  refuse a start unless the worker already has them (nothing is copied or mounted).
WORKER_WEIGHTS="${WORKER_WEIGHTS:-rsync}"
NFS_SERVER="${NFS_SERVER:-}"                     # this Spark's address on the link as the worker reaches it (auto: FABRIC_SELF)
NFS_MODEL_VOLUME="${NFS_MODEL_VOLUME:-glm53-model}"      # the worker's Docker NFS volume for the checkpoint
NFS_DFLASH2_VOLUME="${NFS_DFLASH2_VOLUME:-glm53-dflash2}"
NFS_OPTS="${NFS_OPTS:-nfsvers=4.2,ro,nconnect=8,rsize=1048576,wsize=1048576,hard,timeo=600}"
DATA_DIR="${DATA_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/aevonix-glm53}"
MODEL_SOURCE="${MODEL_SOURCE:-download}"
# ABLIT=1 serves the Ablit weights (the same checkpoint, abliterated; README "Ablit weights") instead of the published
# one; 0 (default) the published one. The Ablit repository is gated: fetching it needs HF_TOKEN, from an account that
# accepted the terms on the model's page. It downloads beside the published checkpoint, which stays in the cache.
ABLIT="${ABLIT:-0}"
ABLIT_ID="Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold-Ablit"
_id="Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold"; _dir=model
if [[ "$ABLIT" == 1 ]]; then _id=$ABLIT_ID; _dir=model-ablit; fi
MODEL_ID="${MODEL_ID:-$_id}"
# The pins belong to the checkpoints above; another MODEL_ID needs its own MODEL_REVISION.
case "$MODEL_ID" in
  Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold) _rev=76c0b5173166d2795dd48860f45d8224817f894c ;;
  "$ABLIT_ID") _rev=57edefd2f5d9b371c8345883304d5af68b52fa24 ;;
  *) _rev="" ;;
esac
MODEL_REVISION="${MODEL_REVISION:-$_rev}"
MODEL_DIR="${MODEL_DIR:-$DATA_DIR/$_dir}"
unset _id _dir _rev
WORKER_MODEL_DIR="${WORKER_MODEL_DIR:-$MODEL_DIR}"
# HF_TOKEN from scripts/local.sh or .env reaches the hf CLI only when exported. It is never passed to the ranks.
[[ -z "${HF_TOKEN:-}" ]] || export HF_TOKEN
DFLASH2_ID="${DFLASH2_ID:-incoai/GLM-5.3-Flash-DFlash2}"          # CC BY-NC-ND 4.0: non-commercial
DFLASH2_REVISION="${DFLASH2_REVISION:-bf582e4eacc1810f76656d1811693ff6c6737d2a}"
DFLASH2_DIR="${DFLASH2_DIR:-$DATA_DIR/dflash2}"                   # downloaded at the pinned revision when missing
WORKER_DFLASH2_DIR="${WORKER_DFLASH2_DIR:-$DFLASH2_DIR}"          # the worker's copy (same path by default)
# A mixed-width EXL3 quant names avg_bits, not bits; the stock GLM check wants quantization_config.bits == 4.
# auto: mount a copy of config.json with bits 4 added over /model/config.json (only when bits is missing); 0: never.
CONFIG_OVERLAY="${CONFIG_OVERLAY:-auto}"
CACHE_DIR="${CACHE_DIR:-$DATA_DIR/zig-home}"     # the ranks' HOME: the engine's per-file checkpoint digest cache
WORKER_CACHE_DIR="${WORKER_CACHE_DIR:-$CACHE_DIR}"
# Where the worker keeps this recipe's own small files (the config.json overlay, when one is needed).
WORKER_STATE_DIR="${WORKER_STATE_DIR:-$WORKER_CACHE_DIR/state}"
WORKER_OVERLAY="${WORKER_OVERLAY:-$WORKER_STATE_DIR/model-config.json}"
STATE_DIR="${STATE_DIR:-$REPO_ROOT/.state}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/logs}"
LOG_KEEP="${LOG_KEEP:-10}"                       # newest compressed session logs kept

# --- serving --------------------------------------------------------------------------------------------------------
DRAFTER="${DRAFTER:-dflash2}"                    # dflash2 (--drafter DIR) | mtp (the checkpoint's MTP head) | none (--no-drafts)
DRAFTER_BITS="${DRAFTER_BITS:-4}"                # --drafter-bits: 4 (default) or 0 (bf16 drafter)
MTP_DRAFTS="${MTP_DRAFTS:-}"                     # --mtp-drafts N (empty: the engine's policy)
MTP_CONFIDENCE="${MTP_CONFIDENCE:-}"             # --mtp-confidence F (empty: off)
PARALLEL="${PARALLEL:-4}"                        # --parallel: streams at once (a number is required to fit; auto: up to 8)
# TENSORFOLD_PREFILL_SHARE: the share of wall time a long prompt's chunks (4,096 rows with patches 0094/0095, about
# 2.5 s each on this pair) get while other streams decode. 1: one decode round per chunk, so a stream decoding beside a
# long prefill gets one round every chunk. p: each chunk owes the decoders (1 - p) / p of its own time in decode rounds
# first (0.68: about 1.2 s, ~22 rounds; 0.5: as long as the chunk itself), and the long prompt's first token comes later
# by that much. Replies are the same at any value: it only moves when the next chunk runs. See README.
PREFILL_SHARE="${PREFILL_SHARE:-0.68}"
# TENSORFOLD_PREFILL_BUDGET: prompt rows filled in one visit while a stream waits to decode (empty: no cap)
PREFILL_BUDGET="${PREFILL_BUDGET:-}"
# auto: the largest window that fits. The first start tries CONTEXT_MAX (the model's 1,048,576), and when a rank
# refuses it with its memory figures, starts once more with the window they allow; the window that served is kept in
# STATE_DIR/context-fit.tsv for the next start with the same memory settings.
CONTEXT="${CONTEXT:-auto}"
CONTEXT_MAX="${CONTEXT_MAX:-1048576}"
CONTEXT_RETRY="${CONTEXT_RETRY:-1}"              # 1: one more start with a fitting window after a memory refusal
CONTEXT_FIT_MARGIN="${CONTEXT_FIT_MARGIN:-0.93}" # share of the computed window the retry asks for
KV="${KV:-fp8}"                                  # --kv-cache: fp8 (a window per stream, or one shared pool with KV_PAGES=on) | bf16 (shared KV pages)
# TF_KV_PAGES. on: every stream draws its KV from one shared pool of 512-row pages, and CONTEXT becomes one request's
# cap (prompt + max_tokens), up to 1048576; a request that doesn't fit waits for pages. off: with KV=fp8 a fixed window
# per stream. auto: the engine's default (bf16 on pages, fp8 windows). fp8 on pages needs engine patches 0017-0025.
KV_PAGES="${KV_PAGES:-on}"
# q4 is rejected on this checkpoint, not merely unused: NLL 0.8005 against q8's 0.7535 (about 40x the
# ABSORB_MMA-vs-off difference), it fails the same spelling question, and at 1,747 tok/s with ABSORB_MMA it is
# slower than q8 + ABSORB_MMA + W8A8. Dense weights are only ~1.8 GiB of 81.8 here (first_k_dense_replace 3,
# 288 routed experts), so narrowing them buys little and costs measurably.
DENSE="${DENSE:-q8}"                             # TF_GLM_DENSE_BITS: bf16 (16) | q8 | q6 | q4 (symmetric int codes, groups of 64)
HEAD="${HEAD:-bf16}"                             # TF_GLM_HEAD_BITS for the LM head, same choices
# TENSORFOLD_MEMORY_RESERVE_GIB. GB10 is a unified-memory device: the engine budgets from /proc/meminfo's MemAvailable
# (plus the pages parked on the kernel's per-CPU free lists, which MemAvailable leaves out), not from a card's free
# memory, and everything the Spark runs besides the rank comes out of the same pool. The engine's own default here is
# max(4 GiB, MemTotal/10), about 11.9 GiB on a 128 GB Spark; its floor is 2. Raise it when a long session pushes a
# Spark toward its floor: rank 0 also carries the HTTP server and the vision tower.
MEMORY_RESERVE_GIB="${MEMORY_RESERVE_GIB:-12}"
PROMPT_CACHE_GIB="${PROMPT_CACHE_GIB:-}"         # --prompt-cache-gib of device pages (empty: the engine's default)
# TF_GLM_HOST_CACHE_GIB, the kept prompt states' pinned host tier. On a discrete card that memory is free; on GB10 it
# is the very memory the KV pool and the weights draw on, and a state in it costs a copy to restore, so the tier is off
# here and PROMPT_CACHE_GIB (device pages) is the one to raise. Unset or auto would pin a quarter of MemAvailable,
# up to 16 GiB over the two ranks.
HOST_CACHE_GIB="${HOST_CACHE_GIB:-0}"
VISION="${VISION:-1}"                            # --vision: images (rank 0's tower)
VISION_MAX_IMAGES="${VISION_MAX_IMAGES:-50}"     # --vision-max-images: images per request (engine default 4, at most 64)
# --chat-template: auto = with VISION=1, templates/glm-5.3-flash-vision.jinja (the checkpoint's own template, whose media
# branch writes GLM's image and video markers instead of a "cannot see" note); none = the checkpoint's; or a file path
CHAT_TEMPLATE="${CHAT_TEMPLATE:-auto}"
THINKING="${THINKING:-0}"                        # --thinking / --no-thinking default; requests override
REASONING_EFFORT="${REASONING_EFFORT:-xhigh}"    # --reasoning-effort low|medium|high|xhigh (empty: omit); xhigh renders the checkpoint's trained "Reasoning Effort: Max" line
MAX_TOKENS="${MAX_TOKENS:-32768}"                # --max-tokens default reply limit
CLEAR_THINKING="${CLEAR_THINKING:-1}"            # TF_GLM_CLEAR_THINKING: 1 drops earlier turns' reasoning, 0 keeps it
# TF_GLM_CE_EXCHANGE moves a prompt chunk's split exchanges by copy engines into the peer's registered rows, which
# needs CUDA IPC between the ranks: that exists between two GPUs in one box, not between two Sparks. The engine falls
# back to NCCL by itself when it cannot map the peer, so 1 would only log that it did; off here.
CE_EXCHANGE="${CE_EXCHANGE:-0}"
# Prompt numerics (engine patches 0041 on): faster prefill, not bit-identical to stock. Both carried +6.6% and
# +10% claims from sm_120; measured on a Spark pair on 2026-11-02 (3 cold trials a point, fresh prompts so no
# prefix-cache hit, NLL over four held passages plus ten greedy answers):
#   ABSORB_MMA=1 (absorb_q/expand_v on tensor cores, an fp32 summation reorder): prefill 1,503 -> 1,623 tok/s at
#     23.5K and 1,530 -> 1,652 at 95K, +8.0%. All ten greedy answers byte-identical to it off, and NLL 0.7523
#     against 0.7535 -- equal or a shade better. Nothing is traded for it, so it is on by default here.
#   W8A8=1 (int8 activations in the prompt dense GEMMs): a further +10.5% (1,789 / 1,828 tok/s), which is the
#     v1.0.7 Spark build's prefill. It rounds activations: per-passage NLL moves +-0.03 (its total matching
#     baseline is cancellation across passages, not equivalence) and it answered "spell necessary backwards"
#     as yrassen for baseline's correct yrassecen. Off; turn it on only behind your own eval.
ABSORB_MMA="${ABSORB_MMA:-1}"                    # TF_GLM_ABSORB_MMA: measured free on GB10 (see above)
W8A8="${W8A8:-0}"                                # TF_GLM_W8A8: stashed (see above); a real rounding of activations
SERVE_EXTRA_ARGS="${SERVE_EXTRA_ARGS:-}"         # more tensorfold-native serve flags, word-split, given to every rank

# --- memory guard ----------------------------------------------------------------------------------------------------
# A Spark's weights, KV, prompt cache and everything else it runs share one 128 GB pool, and a box driven into the
# kernel's reclaim path can stop answering ssh long before the OOM killer picks a victim. MEMGUARD=1 samples
# MemAvailable on both Sparks and, when one stays under MEMGUARD_FLOOR_GIB for MEMGUARD_HOLD_S, stops the server (the
# same way ./stop.sh does) and says which Spark and when in the run's log. 0 turns it off.
MEMGUARD="${MEMGUARD:-1}"
MEMGUARD_FLOOR_GIB="${MEMGUARD_FLOOR_GIB:-2.5}"
MEMGUARD_HOLD_S="${MEMGUARD_HOLD_S:-4}"
MEMGUARD_INTERVAL_S="${MEMGUARD_INTERVAL_S:-2}"

# --- checks and timeouts -----------------------------------------------------------------------------------------------
DRY_RUN="${DRY_RUN:-0}"                          # 1: print the plan; no Docker, ssh, GPU, download or file change
WAIT_TIMEOUT="${WAIT_TIMEOUT:-2400}"             # seconds for the API to come up (digest, load, graphs)
SMOKE_TIMEOUT="${SMOKE_TIMEOUT:-180}"
STOP_TIMEOUT="${STOP_TIMEOUT:-45}"               # docker stop grace: rank 0 drains for TF_DRAIN_S (20 s) first
# Memory a Spark must have available before a start, counted as the engine counts it (MemAvailable plus the parked
# per-CPU pages; scripts/memavail.sh). The model is about 82 GiB a rank, and the reserve and the KV pool come on top.
MIN_AVAIL_GIB="${MIN_AVAIL_GIB:-100}"
IMAGE_FREE_GIB="${IMAGE_FREE_GIB:-40}"
MODEL_FREE_GIB="${MODEL_FREE_GIB:-180}"
DFLASH2_FREE_GIB="${DFLASH2_FREE_GIB:-10}"
