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
TF_REF="${TF_REF:-81de26f649677aa72ae0034d4af6e010cd947038}"
ZIG_VERSION="${ZIG_VERSION:-0.17.0}"
# sha256 of zig-aarch64-linux-0.17.0.tar.xz (ziglang.org/download/index.json).
ZIG_SHA256="${ZIG_SHA256:-9e8d11661d4ae3bd57702a3832781e23ad151dde5798e16a5ccd503f65234ff8}"
SM="${SM:-}"                                     # empty/auto: GPU compute cap (GB10 -> 121, Jetson AGX Thor -> 110a)
ZIG_CPU="${ZIG_CPU:-native}"                     # the image is built on a node and runs on it; baseline to move it
NCCL_WHEEL="${NCCL_WHEEL:-}"                     # non-empty: install this nvidia-nccl-cu13 version over the base image's
IMAGE="${IMAGE:-}"                               # empty: tensorfold-glm53-spark:<PATCHES_HASH>
# How Docker hands a container the GPU. gpus: `--gpus all` (DGX OS on Spark). runtime: `--runtime=nvidia` (JetPack on
# Thor, which refuses `--gpus`). auto asks each node's daemon.
GPU_RUNTIME="${GPU_RUNTIME:-auto}"
# Minimum link speed (Mb/s) for auto-discovered Ethernet rails when no RDMA device is present (keeps 1G LAN out).
FABRIC_MIN_SPEED_MBPS="${FABRIC_MIN_SPEED_MBPS:-10000}"
RAIL_TCP_BUFFER_BYTES="${RAIL_TCP_BUFFER_BYTES:-16777216}"
# WITH_VIDEO=1 builds libtfvideo.so (FFmpeg 9.0.2, static) into the image, so VISION=1 serves video_url parts too
# (issue 8). It adds a few minutes to a build and has its own image tag. 0 (default): video_url parts return HTTP 500.
WITH_VIDEO="${WITH_VIDEO:-0}"
IMAGE_REPO="${IMAGE_REPO:-tensorfold-glm53-spark}"
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
NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-}"     # this Spark's link netdev (auto); the worker's is found on the worker
TF_ROCE_HCA="${TF_ROCE_HCA:-}"                   # RDMA devices for the link, comma separated, at most two (auto)
NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX:-}"       # the RoCE v2 IPv4 GID index (auto; it differs between hosts' tables)
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
ROCE="${ROCE:-auto}"                             # auto (on on RDMA fabric, off on Ethernet Thor) | on | off
ROCE_MAX_KB="${ROCE_MAX_KB:-4096}"               # TF_ROCE_MAX_KB, the largest shard the RDMA region holds
# /dev/mgbe_zc 4x25GbE zero-copy TP=2 transport on Jetson AGX Thor (zig/src/cuda/mgbe_zc.zig, patch 0018).
# auto: enabled when LINK_KIND=tcp and /dev/mgbe_zc exists on both nodes; on: require it; off: NCCL TCP only.
MGBE_ZC="${MGBE_ZC:-auto}"
MGBE_ZC_GROUP="${MGBE_ZC_GROUP:-mgbe_zc}"        # host group owning /dev/mgbe_zc via udev rule (passed as --group-add)
MGBE_ZC_MAX_KB="${MGBE_ZC_MAX_KB:-16384}"        # TF_MGBE_ZC_MAX_KB: max shard per rank (covers decode + pieced prefill)
MGBE_ZC_EAGER_KB="${MGBE_ZC_EAGER_KB:-32}"       # TF_MGBE_ZC_EAGER_KB: per-rail eager (VDMA2) vs DDP (VDMA3) threshold
MGBE_ZC_PORTS="${MGBE_ZC_PORTS:-}"               # TF_MGBE_ZC_PORTS: comma list of MGBE port indices (auto from live rails)
MGBE_ZC_CPUS="${MGBE_ZC_CPUS:-}"                 # TF_MGBE_ZC_CPUS: optional CPU affinity per rail worker (e.g. 2,3,4,5)
TP2_PREFILL_REDUCE="${TP2_PREFILL_REDUCE:-auto}" # auto (gather when MGBE_ZC=on so prefill uses 4-rail DDP, else split) | gather | split
GLM_MM_CUBLAS="${GLM_MM_CUBLAS:-0}"              # TF_GLM_MM_CUBLAS: opt-in cuBLASLt for dense prompt linears >= 128 rows

# --- four Sparks (./start-tp4.sh, ./start-tp4-switchless.sh, ./stop-tp4.sh; README "4 Sparks") ----------------------
# TP=4 over four Sparks: this Spark (rank 0, the API) and WORKER, WORKER2, WORKER3 (ranks 1-3). On a switch
# (./start-tp4.sh) every Spark's ports share the switch's subnet and the workers may be in any order. Cabled as a ring
# without a switch (./start-tp4-switchless.sh) they go in the ring's order, each cabled to the next and the last back to
# this one, one subnet per cable; the two Sparks across the ring from each other reach each other only through a
# neighbour (an address routed over the cables, or the LAN). start.sh never reads these; the TP=4 starts set TP=4.
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
# ROCE_RING=on: the decode and verify all-gathers up to ROCE_MAX_KB a rank as one-shot RDMA writes to both
# neighbours, each relaying half of a shard to the rank across (engine patch 0108, TF_ROCE_RING=1, TF_ROCE_HCA naming
# the device toward the previous rank then the one toward the next); every rank must bring it up or all four fall
# back to NCCL together. off (the default until it is measured): NCCL carries every exchange.
ROCE_RING="${ROCE_RING:-off}"
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
# FP8=1 serves Z.AI's own release, zai-org/GLM-5.3-Flash, which is published in block FP8 (e4m3 codes, 128 x 128 scales;
# ~306 GiB, engine patch 0107; README "Block FP8 weights"). It fits four Sparks only (./start-tp4.sh, ./start-tp4-switchless.sh): ~81 GiB of
# weights a rank. unsloth/GLM-5.3-Flash-FP8 is a byte-identical mirror of it (every shard's sha256 the same).
FP8="${FP8:-0}"
FP8_ID="zai-org/GLM-5.3-Flash"
if [[ "$FP8" == 1 ]]; then _id=$FP8_ID; _dir=model-fp8; fi
MODEL_ID="${MODEL_ID:-$_id}"
# The pins belong to the checkpoints above; another MODEL_ID needs its own MODEL_REVISION.
case "$MODEL_ID" in
  Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold) _rev=76c0b5173166d2795dd48860f45d8224817f894c ;;
  "$ABLIT_ID") _rev=57edefd2f5d9b371c8345883304d5af68b52fa24 ;;
  "$FP8_ID") _rev=46a050bec13002ad29ee795c16e61cf4fa3c4a9c ;;
  unsloth/GLM-5.3-Flash-FP8) _rev=a160e2291674d9e3e92e98fd82faa2544a2867a3 ;;   # the mirror, the same bytes
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
# max(4 GiB, MemTotal/10), about 11.9 GiB on a 128 GB Spark; its floor is 2. 6.5 everywhere since 2026-11-11 (it was 12;
# the pair ran 9.5). Raise it when a long session pushes a Spark toward its floor: rank 0 also carries the HTTP server
# and the vision tower.
MEMORY_RESERVE_GIB="${MEMORY_RESERVE_GIB:-6.5}"
# CACHE_FIRST=1 (engine patch 0112, TF_PROMPT_CACHE_FIRST): PROMPT_CACHE_GIB is set aside before the room is counted
# in whole CONTEXT windows, so the PARALLEL streams share the pages left (one window at least) instead of the cache
# getting what whole windows leave. 0: windows first. On by default since 2026-11-11: at a 6.5 reserve windows-first
# would let the pair's page pool take a third 1M window and leave no prompt cache (the cliff RESULTS.md 2026-11-04
# records at 19.5 GiB of room); FP8 at four Sparks had none at 9.5.
CACHE_FIRST="${CACHE_FIRST:-1}"
# PROMPT_CACHE_GIB, device pages for kept prompt states: 5 on the pair (predicted to keep its two 1M windows, 4,096
# pages, beside it), 8 with EXL3 at four Sparks (measured: all eight windows, 16,392 pages, and an 8.0 GiB cache), 4
# with FP8=1 (measured: 6,147 pages and a 3.9 GiB cache).
if [[ "${GLM_TP4:-0}" == 1 && "$FP8" == 1 ]]; then PROMPT_CACHE_GIB="${PROMPT_CACHE_GIB:-4}"
elif [[ "${GLM_TP4:-0}" == 1 ]]; then PROMPT_CACHE_GIB="${PROMPT_CACHE_GIB:-8}"
else PROMPT_CACHE_GIB="${PROMPT_CACHE_GIB:-5}"; fi
# PROMPT_CACHE_MIN_GIB: the smallest prompt cache a start may serve with, in GiB. The engine funds the cache from the
# room each rank has at that start (it varies by a GiB or more between starts), and PROMPT_CACHE_GIB only caps it, even
# with CACHE_FIRST=1. With a value here, start.sh reads each rank's startup line and stops both ranks (E_CACHE) before
# LIVE when one funds less. Empty (default): no check.
PROMPT_CACHE_MIN_GIB="${PROMPT_CACHE_MIN_GIB:-}"
# TF_GLM_HOST_CACHE_GIB, the kept prompt states' pinned host tier. On a discrete card that memory is free; on GB10 it
# is the very memory the KV pool and the weights draw on, and a state in it costs a copy to restore, so the tier is off
# here and PROMPT_CACHE_GIB (device pages) is the one to raise. Unset or auto would pin a quarter of MemAvailable,
# up to 16 GiB over the two ranks.
HOST_CACHE_GIB="${HOST_CACHE_GIB:-0}"
# LEARN_GIB (engine patch 0113, --learn-conversations): kept prompt states also go to disk under CACHE_DIR/learn,
# every rank's part in rank 0's store, so a conversation resumes after its device copy is evicted or the pair
# restarts. The GiB of disk the states may take; 0 (default): off, as before. LEARN_MIN_FREE_GIB: no state is
# written while the disk has less free.
LEARN_GIB="${LEARN_GIB:-0}"
LEARN_MIN_FREE_GIB="${LEARN_MIN_FREE_GIB:-15}"
# KV_SHARD=1 (engine patch 0114, TF_GLM_KV_SHARD): each stream's FP8 latent rows live on one Spark (its owner) instead
# of both, so the same memory holds about twice the tokens; the other Spark's queries go to the owner for attention
# (1-4% decode). Needs KV=fp8, KV pages, DFlash2 or no drafter, PARALLEL above 1. The pool then takes most of the
# room: pair it with CACHE_FIRST=1 PROMPT_CACHE_GIB so long conversations stay kept. 0 (default): off, as before.
KV_SHARD="${KV_SHARD:-0}"
# L2PF (TF_GLM_L2PF, built into TensorFold 1.3.0): each decode window reads the next layer's mixer weights, this
# layer's router and KDA's o_proj into L2 on a side stream while the gathers and chains run (not the EXL3 routed
# experts). The engine turns it on by itself on a GB10 when the variable is unset, so the launchers always pass it.
# On the pair it gave +2% at one stream unsharded (2026-11-11) and +2.2-2.7% at one, +1.1% at four with KV_SHARD=1
# (issue #54), token hashes identical. Both ranks must agree (the fingerprint carries it). 1 (default): on; 0: off.
L2PF="${L2PF:-1}"
VISION="${VISION:-1}"                            # --vision: images (rank 0's tower)
VISION_MAX_IMAGES="${VISION_MAX_IMAGES:-50}"     # --vision-max-images: images per request (engine default 4, at most 64)
# --chat-template: auto = with VISION=1, templates/glm-5.3-flash-vision.jinja (the checkpoint's own template, whose media
# branch writes GLM's image and video markers instead of a "cannot see" note); none = the checkpoint's; or a file path
CHAT_TEMPLATE="${CHAT_TEMPLATE:-auto}"
# The template's "Reasoning Effort" line. The checkpoint's template (and the vision copy) sets it to Max unless a
# request's effort is low or high, so every prompt carries `<|system|>Reasoning Effort: Max`, thinking off included,
# and a thinking-off reply then reasons in its answer (a 64-token "digits only" question runs out of tokens mid-sum).
# TensorFold v0.6.0's recipe wrote the line with thinking on only. thinking (default): the same here -- the served
# template is written to STATE_DIR/chat_template.jinja from the chosen one, its line taken from reasoning_effort,
# which the engine passes with thinking on only; always: the template as it is. CHAT_TEMPLATE=auto only; a template
# of your own decides for itself.
EFFORT_LINE="${EFFORT_LINE:-thinking}"
# EXTRA_EOS (engine patch 0120, TF_EXTRA_EOS): tokens that end a reply besides the checkpoint's EOS ids
# (<|endoftext|>, <|user|>, <|observation|>), comma-separated. GLM-5.3 can close a reply with <|assistant|>, which
# its generation_config leaves out: the token then reaches the reply as text and the model writes a second turn of
# its own until MAX_TOKENS. none: the checkpoint's list only.
EXTRA_EOS="${EXTRA_EOS:-<|assistant|>}"
# THINK_OFF_SPLIT (engine patch 0121, TF_THINK_OFF_SPLIT): with thinking off, GLM-5.3 sometimes drafts in its reasoning
# voice anyway and writes </think> before its answer. A chat reply that writes that token within its first
# THINK_OFF_SPLIT tokens gets the text before it as reasoning_content and the rest as content, as a thinking reply
# does. While a reply streams, its content waits until the token, the end or THINK_OFF_SPLIT tokens (whichever comes
# first), so a long thinking-off reply's first content arrives up to that many tokens later. 0 (default): off, since
# that hold delays every streamed thinking-off reply; 1024 is the hold its author measured.
THINK_OFF_SPLIT="${THINK_OFF_SPLIT:-0}"
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
