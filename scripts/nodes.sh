# The two Sparks: reaching the worker, finding the link between them, and keeping the image and the weights in step.
# Sourced by scripts/prepare.sh after its log/fail/warn/plan/shq helpers, and so by start.sh and stop.sh.
#
# The link discovery (link_info, rails, rail_info, rail_ping, cx7_peer) is adapted from the 2x DGX Sparks recipe's
# scripts/nodes.sh (github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold, branch zig-engine), trimmed to
# two ranks. The lessons in it are that recipe's: the GID index must be the RoCE v2 entry that holds the node's own
# IPv4 address, and a DGX Spark's single QSFP port reaches the GB10 over two PCIe links, so one cable shows up as two
# netdevs and two RDMA devices and both have to be given to NCCL (and to the engine's RoCE path) for the port's full
# width.
[[ "${_GLM_NODES_LOADED:-0}" == 1 ]] && return 0
_GLM_NODES_LOADED=1
# shellcheck source=scripts/memavail.sh
source "$(dirname -- "${BASH_SOURCE[0]}")/memavail.sh"

# --- reaching the worker ---------------------------------------------------------------------------------------------
worker_host() { printf '%s' "${WORKER#*@}"; }

# wssh <words...>: run a command on the worker. The words are quoted for the remote shell, so paths with spaces work;
# $HOME is left for the remote shell to expand when a word is exactly that.
wssh() {
  local a remote=''
  for a in "$@"; do
    case "$a" in '$HOME'|'$HOME'/*) remote+=" \"$a\"" ;; *) remote+=" $(printf '%q' "$a")" ;; esac
  done
  # shellcheck disable=SC2086
  ssh $SSH_OPTS -- "$WORKER" "$remote"
}
# wsh <script text>: run a snippet on the worker's bash (for a loop or a sent function definition).
# shellcheck disable=SC2086
wsh() { ssh $SSH_OPTS -- "$WORKER" bash -s <<< "$1"; }
# wfn <function names...> -- <call text>: send those function definitions and run <call text> on the worker.
wfn() {
  local -a names=()
  while (( $# )) && [[ "$1" != -- ]]; do names+=("$1"); shift; done
  shift || true
  wsh "$(declare -f "${names[@]}"); $*"
}
worker_reachable() { wssh true >/dev/null 2>&1; }

# --- the link between the nodes --------------------------------------------------------------------------------------
# Two kinds of fabric carry the ranks:
#   ib   both nodes expose an RDMA device (a Spark pair on ConnectX-7): NCCL over verbs, plus the engine's own RoCE
#        all-gathers.
#   tcp  at least one node has no RDMA device (a Jetson AGX Thor pair: its 100G QSFP cage is plain Ethernet on the
#        SoC's MGBE controllers, no verbs device anywhere): NCCL's socket transport over every directly reached rail.
# The kind is found, not configured; the discovery below never assumes a ConnectX-7.

# rdma_present: this node can open an RDMA device (a verbs HCA under /sys/class/infiniband).
rdma_present() { compgen -G '/sys/class/infiniband/*' >/dev/null 2>&1; }

# node_addrs <min speed Mb/s>: this node's IPv4 addresses on up netdevs at or above that speed, one per line:
# "<netdev> <address> <prefix> <mtu> <speed>". Self-contained: it is sent to the worker as text.
node_addrs() {
  local min=${1:-0} d n a mtu spd
  for d in /sys/class/net/*; do
    n=${d##*/}
    [[ -e "$d/device" ]] || continue                 # a real port: no bridge, veth, bond or loopback
    [[ "$(cat "$d/operstate" 2>/dev/null)" == up ]] || continue
    spd=$(cat "$d/speed" 2>/dev/null || echo 0)
    case "$spd" in ''|*[!0-9]*) spd=0 ;; esac          # bonds, bridges and virtual links report -1 or nothing
    (( spd >= min )) || continue
    mtu=$(cat "$d/mtu" 2>/dev/null || echo 1500)
    while read -r _ _ _ a _; do
      [[ -n "$a" ]] || continue
      printf '%s %s %s %s %s\n' "$n" "${a%%/*}" "${a##*/}" "$mtu" "$spd"
    done < <(ip -o -4 addr show dev "$n" 2>/dev/null)
  done
}

# fabric_pairs: the rails between this node and the worker, one per line: "<netdev> <address> <worker netdev>
# <worker address> <mtu>". A rail is a pair of addresses in one subnet on both nodes at or above FABRIC_MIN_SPEED_MBPS,
# so a 1G/5G management LAN beside the 100G cage is left out. NCCL pairs its own interfaces with the peer's on the
# same subnet, so listing every rail as NCCL_SOCKET_IFNAME stripes the channels across them.
fabric_pairs() {
  { node_addrs "$FABRIC_MIN_SPEED_MBPS"; printf -- '--\n'
    wfn node_addrs -- "node_addrs $FABRIC_MIN_SPEED_MBPS" 2>/dev/null || true
  } | python3 -c '
import ipaddress, sys
mine, theirs, side = [], [], 0
for line in sys.stdin:
    line = line.strip()
    if line == "--":
        side = 1; continue
    p = line.split()
    if len(p) != 5: continue
    (mine if side == 0 else theirs).append(p)
def net(a, pfx):
    try: return ipaddress.ip_network(f"{a}/{pfx}", strict=False)
    except ValueError: return None
def addr(a):
    try: return ipaddress.ip_address(a)
    except ValueError: return None
for dev, ip, pfx, mtu, _ in mine:
    ln, la = net(ip, pfx), addr(ip)
    if ln is None or la is None: continue
    for wdev, wip, wpfx, wmtu, _ in theirs:
        wn, wa = net(wip, wpfx), addr(wip)
        if wn is None or wa is None or la not in wn or wa not in ln: continue
        print(dev, ip, wdev, wip, min(int(mtu), int(wmtu)))
'
}

# link_info <peer address>: this node's side of the link to <peer>: "<address> <netdev> <rdma device> <gid index>".
# The netdev and source address come from the route to the peer; the RDMA device from sysfs ("-" when the port has
# none); the GID index is the RoCE v2 entry that holds this node's own IPv4 address (an all-zero or v1 entry makes
# NCCL fail about a minute in). Self-contained: it is sent to the worker as text.
link_info() {
  local peer=$1 route dev src hca idx type gid want f
  route=$(ip -o -4 route get "$peer" 2>/dev/null) || return 1
  dev=$(sed -n 's/.* dev \([^ ]*\).*/\1/p' <<<"$route")
  src=$(sed -n 's/.* src \([^ ]*\).*/\1/p' <<<"$route")
  [[ -n "$dev" && -n "$src" ]] || return 1
  hca=$(ls /sys/class/net/"$dev"/device/infiniband 2>/dev/null | head -1)
  idx=""
  if [[ -n "$hca" ]]; then
    # shellcheck disable=SC2046
    want=$(printf '0000:0000:0000:0000:0000:ffff:%02x%02x:%02x%02x' $(tr '.' ' ' <<<"$src"))
    for f in /sys/class/infiniband/"$hca"/ports/1/gids/*; do
      gid=$(cat "$f" 2>/dev/null) || continue
      [[ "$gid" == "$want" ]] || continue
      type=$(cat /sys/class/infiniband/"$hca"/ports/1/gid_attrs/types/"${f##*/}" 2>/dev/null)
      [[ "$type" == *v2* ]] && { idx=${f##*/}; break; }
    done
  fi
  echo "$src $dev ${hca:--} ${idx:--}"
}

# rails <netdev> <gid index>: the RDMA devices this node can run the link on, comma separated, the link's own first:
# every other RoCE device that is up, holds a RoCE v2 GID at the same index, and either sits on the link's subnet (a
# second cabled port) or is the same QSFP port's PCIe twin (same fN/npM tail, another PCIe prefix). The engine's RoCE
# path stripes over at most two, and NCCL takes the list as NCCL_IB_HCA. Self-contained.
rails() {
  local dev=$1 gid=$2 net cidr hcas other ip idx n tail other2 ib2
  cidr=$(ip -o -4 addr show dev "$dev" 2>/dev/null | awk '{print $4}' | head -1)
  hcas=$(ls /sys/class/net/"$dev"/device/infiniband 2>/dev/null | head -1)
  [[ -n "$hcas" ]] || { echo ""; return 0; }
  net=${cidr%/*}; net=${net%.*}            # the /24 the Sparks' links use; a wider mask falls back to the twin rule
  for n in /sys/class/net/*; do
    other=${n##*/}
    [[ "$other" == "$dev" || ! -d "$n/device/infiniband" || "$(cat "$n/operstate" 2>/dev/null)" != up ]] && continue
    ip=$(ip -o -4 addr show dev "$other" 2>/dev/null | awk '{print $4}' | head -1)
    [[ -n "$ip" && -n "$net" && "${ip%.*}" == "$net" ]] || continue
    idx=$(ls "$n/device/infiniband" 2>/dev/null | head -1)
    [[ -n "$idx" ]] || continue
    [[ "$(cat /sys/class/infiniband/"$idx"/ports/1/gid_attrs/types/"$gid" 2>/dev/null)" == *v2* ]] || continue
    [[ ",$hcas," == *",$idx,"* ]] || hcas+=",$idx"
  done
  tail=${dev##*f}
  [[ "$dev" =~ f[0-9]+np[0-9]+$ ]] || { echo "$hcas"; return 0; }
  for n in /sys/class/net/*; do
    other2=${n##*/}
    [[ "$other2" == "$dev" || "$other2" != *f"$tail" ]] && continue
    [[ -d "$n/device/infiniband" && "$(cat "$n/operstate" 2>/dev/null)" == up ]] || continue
    ib2=$(ls "$n/device/infiniband" 2>/dev/null | head -1)
    [[ -n "$ib2" && ",$hcas," != *",$ib2,"* ]] || continue
    [[ "$(cat /sys/class/infiniband/"$ib2"/ports/1/gid_attrs/types/"$gid" 2>/dev/null)" == *v2* ]] || continue
    hcas+=",$ib2"
  done
  echo "$hcas"
}

# rail_info <rdma device>: "<netdev> <IPv4 address> <MTU>" of the port a RoCE device sits on ("- - -" when it has none).
# Self-contained.
rail_info() {
  local n
  for n in /sys/class/infiniband/"$1"/device/net/*; do
    [[ -d "$n" ]] || continue
    echo "${n##*/} $(ip -o -4 addr show dev "${n##*/}" 2>/dev/null | awk '{print $4}' | head -1 | cut -d/ -f1) $(cat "$n/mtu" 2>/dev/null || echo 1500)"
    return 0
  done
  echo "- - -"
}

# rail_ping <local netdev> <peer address> <MTU>: a full-size, unfragmentable ping bound to that rail's own port, so a
# twin that is up and has a GID but no path to the peer's twin (no address, another subnet, a smaller MTU, a cable on
# another port) fails here instead of as vendor_err 0x81 in the first all-gather. Self-contained.
rail_ping() { ping -c2 -W2 -I "$1" -M do -s $(( $3 - 28 )) "$2" >/dev/null 2>&1; }

# cx7_peer: the worker's address on one of its link ports that a port of this node reaches directly (same subnet, no
# gateway). Empty when there is none, which means the pair is not cabled to each other. On an RDMA fabric only the
# verbs ports count; on TCP any port at or above FABRIC_MIN_SPEED_MBPS does (see fabric_pairs).
cx7_peer() {
  local a route dev
  for a in $(wsh 'for d in /sys/class/net/*; do [ -d "$d/device/infiniband" ] && ip -o -4 addr show dev "${d##*/}"; done' 2>/dev/null |
             awk '{print $4}' | cut -d/ -f1); do
    route=$(ip -o -4 route get "$a" 2>/dev/null) || continue
    [[ "$route" == *" via "* ]] && continue
    dev=$(sed -n 's/.* dev \([^ ]*\).*/\1/p' <<<"$route")
    [[ -d /sys/class/net/$dev/device/infiniband ]] && { echo "$a"; return 0; }
  done
  return 1
}

# Filled by detect_link: this node's and the worker's side of the link.
LINK_SELF= LINK_PEER= LINK_DEV= LINK_HCA= LINK_GID=
WORKER_DEV= WORKER_HCA= WORKER_GID=
LINK_KIND=''                                     # ib (NCCL over verbs) or tcp (NCCL's socket transport)
# detect_link: FABRIC_PEER / FABRIC_SELF / NCCL_SOCKET_IFNAME / TF_ROCE_HCA / NCCL_IB_GID_INDEX, each kept when it was
# set by hand. The fabric's kind decides what the rest of the recipe asks for: with an RDMA device on both nodes,
# NCCL runs over verbs and the engine's RoCE all-gathers are available; without one, NCCL's socket transport carries
# every exchange over the rails fabric_pairs found. Rails past the first are probed and dropped when they do not reach
# the peer's matching rail.
detect_link() {
  local peer self dev hca gid whca wgid wdev info winfo
  if rdma_present && wsh 'ls -d /sys/class/infiniband/* >/dev/null 2>&1'; then
    LINK_KIND=ib
    peer=$FABRIC_PEER
    if [[ -z "$peer" ]]; then
      peer=$(cx7_peer || true)
      [[ -n "$peer" ]] || fail E_FABRIC "Found no RDMA address of $(worker_host) that a port of this node reaches directly. Cable the two nodes and give them addresses on one subnet, or set FABRIC_PEER to the worker's address on the link."
    fi
    info=$(link_info "$peer") || fail E_FABRIC "No route to $peer: check the link to $(worker_host), or set FABRIC_PEER."
    read -r self dev hca gid <<< "$info"
    [[ "$hca" != - ]] || fail E_FABRIC "The route to $peer leaves over $dev, which is not an RDMA port. Set FABRIC_PEER to the worker's address on the direct link."
    [[ "$gid" != - ]] || fail E_FABRIC "$hca has no RoCE v2 GID holding $self. Give the link's port an IPv4 address on both nodes, or set NCCL_IB_GID_INDEX by hand."
    winfo=$(wfn link_info -- "link_info $self") || fail E_FABRIC "Could not read $(worker_host)'s side of the link to $self."
    read -r _ wdev whca wgid <<< "$winfo"
    [[ "$whca" != - && "$wgid" != - ]] || fail E_FABRIC "$(worker_host)'s route back to $self does not leave over an RDMA port with a RoCE v2 GID ($wdev/$whca, gid $wgid)."
    LINK_SELF=${FABRIC_SELF:-$self}; LINK_PEER=$peer
    LINK_DEV=${NCCL_SOCKET_IFNAME:-$dev}; WORKER_DEV=$wdev
    LINK_GID=${NCCL_IB_GID_INDEX:-$gid}; WORKER_GID=${NCCL_IB_GID_INDEX:-$wgid}
    if [[ -n "$TF_ROCE_HCA" ]]; then
      LINK_HCA=$TF_ROCE_HCA; WORKER_HCA=$TF_ROCE_HCA
    else
      LINK_HCA=$(rails "$LINK_DEV" "$LINK_GID"); [[ -n "$LINK_HCA" ]] || LINK_HCA=$hca
      WORKER_HCA=$(wfn rails -- "rails $(printf '%q' "$wdev") $(printf '%q' "$WORKER_GID")"); [[ -n "$WORKER_HCA" ]] || WORKER_HCA=$whca
      probe_rails
    fi
    [[ "$ROCE" != auto ]] || ROCE=on
    log "[I_FABRIC] Link (RDMA): $LINK_SELF ($LINK_DEV) <-> $LINK_PEER ($WORKER_DEV); RDMA $LINK_HCA <-> $WORKER_HCA, RoCE v2 GID $LINK_GID/$WORKER_GID."
    return 0
  fi
  detect_link_tcp
}

# detect_link_tcp: the link when at least one node has no RDMA device. The rails come from fabric_pairs (addresses in
# one subnet on both nodes, at or above FABRIC_MIN_SPEED_MBPS); each is probed with a full-size, unfragmentable ping
# in both directions before it is used, so a port that is up but cabled elsewhere fails here rather than as a hung
# collective later. NCCL then pairs its interfaces with the peer's on the same subnet and stripes its channels over
# them; NCCL_SOCKET_IFNAME takes the whole list.
detect_link_tcp() {
  local -a keep=() rail=()
  local dev ip wdev wip mtu ok first=1
  local -a pairs=()
  mapfile -t pairs < <(fabric_pairs)
  (( ${#pairs[@]} )) || fail E_FABRIC "Found no port at or above ${FABRIC_MIN_SPEED_MBPS} Mb/s that reaches $(worker_host) directly (no gateway). Cable the two nodes on one subnet and give both sides an address, or lower FABRIC_MIN_SPEED_MBPS."
  for rail in "${pairs[@]}"; do
    read -r dev ip wdev wip mtu <<< "$rail"
    [[ -z "$FABRIC_PEER" || "$wip" == "$FABRIC_PEER" ]] || continue
    ok=1
    rail_ping "$dev" "$wip" "$mtu" || ok=0
    (( ok == 0 )) || wfn rail_ping -- "rail_ping $(printf '%q' "$wdev") $(printf '%q' "$ip") $mtu" || ok=0
    if (( ok )); then
      keep+=("$dev $ip $wdev $wip $mtu")
      (( first == 0 )) || { LINK_SELF=${FABRIC_SELF:-$ip}; LINK_PEER=$wip; first=0; }
    else
      warn W_FABRIC "Rail $dev ($ip) <-> $wdev ($wip) on $(worker_host) does not carry a $mtu-byte ping: leaving it out."
    fi
  done
  (( ${#keep[@]} )) || fail E_FABRIC "No rail between this node and $(worker_host) carries a full-size ping. Check the cables, both sides' addresses and MTU, or set FABRIC_PEER."
  LINK_KIND=tcp
  LINK_HCA='' WORKER_HCA='' LINK_GID='' WORKER_GID=''
  if [[ -n "$NCCL_SOCKET_IFNAME" ]]; then
    LINK_DEV=$NCCL_SOCKET_IFNAME
    WORKER_DEV=$(printf '%s\n' "${keep[@]}" | awk -v d="$NCCL_SOCKET_IFNAME" '$1 == d { print $3 }' | paste -sd, -)
    [[ -n "$WORKER_DEV" ]] || WORKER_DEV=$(printf '%s\n' "${keep[@]}" | awk '{ print $3 }' | paste -sd, -)
  else
    LINK_DEV=$(printf '%s\n' "${keep[@]}" | awk '{ print $1 }' | paste -sd, -)
    WORKER_DEV=$(printf '%s\n' "${keep[@]}" | awk '{ print $3 }' | paste -sd, -)
  fi
  # The engine's RoCE all-gathers need verbs; without them every exchange is NCCL's, so say so instead of trying.
  [[ "$ROCE" != on ]] || warn W_ROCE "ROCE=on but there is no RDMA device on both nodes: the ranks run with ROCE=off (NCCL over TCP)."
  ROCE=off
  log "[I_FABRIC] Link (TCP, no RDMA device): ${LINK_SELF} ($LINK_DEV) <-> ${LINK_PEER} ($WORKER_DEV) over ${#keep[@]} rail(s); NCCL's socket transport carries every exchange, the engine's RoCE all-gathers are off."
}

# probe_rails: rail k of this Spark is checked against rail k of the worker, in both directions, before it is used;
# the pairing is by list position, so a rail that is up with a GID but has no path to its partner would end as
# vendor_err 0x81 in the first all-gather. The first device, the link itself, is never probed: ssh already came over it.
probe_rails() {
  local -a mine=() theirs=() keep_mine=() keep_theirs=()
  local k ok hca wrail mdev mip mmtu wdev2 wip wmtu mtu
  IFS=, read -r -a mine <<< "$LINK_HCA"
  IFS=, read -r -a theirs <<< "$WORKER_HCA"
  keep_mine=("${mine[0]}"); keep_theirs=("${theirs[0]}")
  for (( k = 1; k < ${#mine[@]}; k++ )); do
    hca=${mine[k]}; wrail=${theirs[k]:-}
    if [[ -z "$wrail" ]]; then
      warn W_FABRIC "$(worker_host) offers no rail beside ${theirs[0]} to pair with $hca: leaving $hca out."
      continue
    fi
    read -r mdev mip mmtu <<< "$(rail_info "$hca")"
    read -r wdev2 wip wmtu <<< "$(wfn rail_info -- "rail_info $(printf '%q' "$wrail")")"
    if [[ "$mip" == - || "$wip" == - ]]; then
      warn W_FABRIC "Rail $hca ($mdev) or $wrail ($wdev2) on $(worker_host) has no IPv4 address: leaving the pair out."
      continue
    fi
    mtu=$(( mmtu < wmtu ? mmtu : wmtu ))
    ok=1
    rail_ping "$mdev" "$wip" "$mtu" || ok=0
    (( ok == 0 )) || wfn rail_ping -- "rail_ping $(printf '%q' "$wdev2") $(printf '%q' "$mip") $mtu" || ok=0
    if (( ok )); then
      keep_mine+=("$hca"); keep_theirs+=("$wrail")
    else
      warn W_FABRIC "Rail $hca <-> $wrail does not carry a $mtu-byte ping between $mip and $wip: leaving the pair out (the link keeps ${keep_mine[0]})."
    fi
  done
  LINK_HCA=$(IFS=,; printf '%s' "${keep_mine[*]}")
  WORKER_HCA=$(IFS=,; printf '%s' "${keep_theirs[*]}")
}

# nccl_env_args <netdev> <hcas> <gid>: the NCCL settings for one rank's container.
nccl_env_args() {
  NCCL_ENV_ARGS=(-e "NCCL_SOCKET_IFNAME=$1" -e "NCCL_IB_HCA=$2" -e "NCCL_IB_GID_INDEX=$3"
    -e "NCCL_MIN_NCHANNELS=$NCCL_CHANNELS" -e "NCCL_MAX_NCHANNELS=$NCCL_CHANNELS")
}

# --- the image on both Sparks ----------------------------------------------------------------------------------------
# An image's identity by content (its layers' diffIDs and runtime config), the same under Docker's overlay2 and
# containerd image stores: .Id is the config digest under one and the manifest digest under the other, so it would
# never match across a mixed pair. The template holds no spaces: it passes through ssh, which re-splits arguments.
IMAGE_IDENT='{{.RootFS.Layers}}{{.Config.Env}}{{.Config.Entrypoint}}{{.Config.Cmd}}{{.Config.WorkingDir}}'
image_ident() { local s; s=$(docker image inspect -f "$IMAGE_IDENT" "$1" 2>/dev/null) && sha256sum <<<"$s" | cut -c1-64 || echo missing; }
worker_image_ident() { local s; s=$(wssh docker image inspect -f "$IMAGE_IDENT" "$1" 2>/dev/null) && sha256sum <<<"$s" | cut -c1-64 || echo missing; }

# ship_image: the worker runs the very image this Spark built. When its copy is missing or different, the image is
# streamed over the link (docker save | ssh docker load), which is why the pins are hashed into the tag.
ship_image() {
  local here there
  if [[ "$DRY_RUN" == 1 ]]; then
    log "Compare the image on both Sparks by content; when $(worker_host)'s copy differs or is missing:"
    plan docker save "$IMAGE" '|' ssh "$WORKER" docker load
    return 0
  fi
  here=$(image_ident "$IMAGE")
  [[ "$here" != missing ]] || fail E_IMAGE "The image $IMAGE is not on this Spark. Run scripts/prepare.sh --image first."
  there=$(worker_image_ident "$IMAGE")
  if [[ "$here" == "$there" ]]; then
    log "[I_IMAGE] $(worker_host) already runs this image ($IMAGE, ${here:0:12})."
    return 0
  fi
  log "Sending $IMAGE to $(worker_host) (its copy is ${there:0:12}); this takes a few minutes over the link."
  # shellcheck disable=SC2086
  docker save "$IMAGE" | ssh $SSH_OPTS -- "$WORKER" docker load >/dev/null \
    || fail E_IMAGE "Could not load $IMAGE on $(worker_host). Check its disk space and Docker access."
  there=$(worker_image_ident "$IMAGE")
  [[ "$here" == "$there" ]] || fail E_IMAGE "$(worker_host) still reports another image for $IMAGE (${there:0:12} against ${here:0:12})."
  log "[I_IMAGE] $(worker_host) now runs this image ($IMAGE, ${here:0:12})."
}

# --- the worker's memory, disk and weights ---------------------------------------------------------------------------
# The worker's available memory, counted as the engine counts it (memavail.sh's rule, sent over ssh).
worker_avail_gib() { wfn mem_counting mem_avail_gb -- 'mem_avail_gb' 2>/dev/null | awk '{print $1}'; }
worker_free_gib() { wssh df -Pk -- "$1" 2>/dev/null | awk 'END {printf "%d\n", $4 / 1048576}'; }

# worker_weights_ready <dir> <revision>: the worker holds that checkpoint at that revision (the marker this recipe
# writes), or, for a local checkpoint with no marker, at least a config.json and shards.
worker_weights_ready() {
  wssh test -f "$1/config.json" >/dev/null 2>&1 || return 1
  [[ -n "$2" ]] || return 0
  local marker
  marker=$(wssh cat "$1/.revision" 2>/dev/null || true)
  if [[ -n "$marker" ]]; then
    [[ "$marker" == "$2" ]]
  else
    # No marker at all: a MODEL_SOURCE=local checkpoint prepared outside this recipe. Its config.json and shards
    # stand for it, as the header comment says. MODEL_REVISION defaults with :-, so the argument above is never
    # empty in the recipe's calls and cannot itself say "skip the marker".
    wssh sh -c 'ls "$1"/*.safetensors >/dev/null 2>&1' sh "$1" >/dev/null 2>&1
  fi
}

# --- the worker's weights --------------------------------------------------------------------------------------------
# WORKER_MOUNTS: the worker's -v arguments for /model and /dflash2. Both ranks are given the same container paths, so
# both get the same `serve /model --drafter /dflash2` command line and their settings fingerprints agree.
declare -a WORKER_MOUNTS=()

# send_dir <local dir> <worker dir> <what>: rsync over the link, resumable, never deleting anything on the worker.
send_dir() {
  local src=$1 dst=$2 what=$3
  if [[ "$DRY_RUN" == 1 ]]; then
    plan rsync -a --partial --inplace -e "ssh $SSH_OPTS -o Compression=no" "$src/" "$WORKER:$dst/"
    return 0
  fi
  log "Sending the $what to $(worker_host):$dst over the link (resumable)."
  wssh mkdir -p "$dst" || fail E_WEIGHTS "Could not create $dst on $(worker_host)."
  # shellcheck disable=SC2086
  rsync -a --partial --inplace --exclude '.downloading' \
    -e "ssh $SSH_OPTS -o Compression=no -c aes128-gcm@openssh.com" "$src/" "$WORKER:$dst/" \
    || fail E_WEIGHTS "Could not send the $what to $(worker_host). Check its disk space and the link."
}

# nfs_volume <worker volume> <exported path> <container path>: a read-only NFS volume on the worker for one directory,
# created when it is missing and refused when it points somewhere else.
nfs_volume() {
  local vol=$1 path=$2 at=$3 server=${NFS_SERVER:-$LINK_SELF} have
  [[ -n "$server" ]] || fail E_WEIGHTS 'NFS_SERVER is empty and the link address is unknown: set NFS_SERVER to this Spark'"'"'s address on the link.'
  if [[ "$DRY_RUN" == 1 ]]; then
    plan ssh "$WORKER" docker volume create --driver local --opt type=nfs --opt "device=:$path" \
      --opt "o=addr=$server,$NFS_OPTS" "$vol"
    WORKER_MOUNTS+=(-v "$vol:$at:ro")
    return 0
  fi
  have=$(wssh docker volume inspect -f '{{index .Options "device"}} {{index .Options "o"}}' "$vol" 2>/dev/null || true)
  if [[ -z "$have" ]]; then
    wssh docker volume create --driver local --opt type=nfs --opt "device=:$path" \
      --opt "o=addr=$server,$NFS_OPTS" "$vol" >/dev/null \
      || fail E_WEIGHTS "Could not create the Docker volume $vol on $(worker_host)."
  elif [[ "${have%% *}" != ":$path" || "$have" != *"addr=$server,"* ]]; then
    fail E_WEIGHTS "$(worker_host)'s Docker volume $vol is ${have%% *} from ${have#* }, not :$path from $server. Remove it there (docker volume rm $vol), or set NFS_MODEL_VOLUME / NFS_DFLASH2_VOLUME / NFS_SERVER."
  fi
  # A mount only proves itself when something reads it: an export that does not reach the worker hangs or refuses here,
  # before the ranks are started and a hard mount can wedge a load.
  wssh docker run --rm --network host -v "$vol:/w:ro" --entrypoint /bin/sh "$IMAGE" -c 'test -f /w/config.json' \
    >/dev/null 2>&1 || fail E_WEIGHTS "$(worker_host) cannot read $path over NFS from $server (volume $vol). Export it to $(worker_host)'s link address, read-only, and run: sudo exportfs -ra. On this Spark: 'echo \"$path ${LINK_PEER:-<worker link address>}(ro,no_subtree_check,all_squash,anonuid=$(id -u),anongid=$(id -g))\" | sudo tee -a /etc/exports && sudo systemctl enable --now nfs-server && sudo exportfs -ra'."
  log "[I_WEIGHTS] $(worker_host) reads $at over NFS from $server:$path (volume $vol)."
  WORKER_MOUNTS+=(-v "$vol:$at:ro")
}

# prepare_worker_weights: leaves WORKER_MOUNTS ready for launch, by WORKER_WEIGHTS.
prepare_worker_weights() {
  WORKER_MOUNTS=()
  case "$WORKER_WEIGHTS" in
    nfs)
      nfs_volume "$NFS_MODEL_VOLUME" "$MODEL_DIR" /model
      [[ "$DRAFTER" != dflash2 ]] || nfs_volume "$NFS_DFLASH2_VOLUME" "$DFLASH2_DIR" /dflash2
      ;;
    rsync)
      if [[ "$DRY_RUN" == 1 ]]; then
        log "When $(worker_host)'s checkpoint is missing or another revision:"
        send_dir "$MODEL_DIR" "$WORKER_MODEL_DIR" checkpoint
        [[ "$DRAFTER" != dflash2 ]] || send_dir "$DFLASH2_DIR" "$WORKER_DFLASH2_DIR" 'DFlash2 weights'
      else
        if worker_weights_ready "$WORKER_MODEL_DIR" "$MODEL_REVISION"; then
          log "[I_WEIGHTS] $(worker_host) already holds the checkpoint at ${MODEL_REVISION:0:12} ($WORKER_MODEL_DIR)."
        else
          send_dir "$MODEL_DIR" "$WORKER_MODEL_DIR" checkpoint
        fi
        if [[ "$DRAFTER" == dflash2 ]]; then
          if wssh test -f "$WORKER_DFLASH2_DIR/config.json" >/dev/null 2>&1; then
            log "[I_WEIGHTS] $(worker_host) already holds the DFlash2 weights ($WORKER_DFLASH2_DIR)."
          else
            send_dir "$DFLASH2_DIR" "$WORKER_DFLASH2_DIR" 'DFlash2 weights'
          fi
        fi
      fi
      WORKER_MOUNTS+=(-v "$WORKER_MODEL_DIR:/model:ro")
      [[ "$DRAFTER" != dflash2 ]] || WORKER_MOUNTS+=(-v "$WORKER_DFLASH2_DIR:/dflash2:ro")
      ;;
    check)
      if [[ "$DRY_RUN" != 1 ]]; then
        worker_weights_ready "$WORKER_MODEL_DIR" "$MODEL_REVISION" \
          || fail E_WEIGHTS "$(worker_host) has no checkpoint at ${MODEL_REVISION:0:12} in $WORKER_MODEL_DIR. Set WORKER_WEIGHTS=rsync to send this Spark's copy, WORKER_WEIGHTS=nfs to mount it, or write $MODEL_REVISION to $WORKER_MODEL_DIR/.revision when that tree is this checkpoint."
        [[ "$DRAFTER" != dflash2 ]] || wssh test -f "$WORKER_DFLASH2_DIR/config.json" >/dev/null 2>&1 \
          || fail E_WEIGHTS "$(worker_host) has no DFlash2 weights in $WORKER_DFLASH2_DIR (WORKER_WEIGHTS=check)."
        log "[I_WEIGHTS] $(worker_host) holds the checkpoint and the drafter."
      fi
      WORKER_MOUNTS+=(-v "$WORKER_MODEL_DIR:/model:ro")
      [[ "$DRAFTER" != dflash2 ]] || WORKER_MOUNTS+=(-v "$WORKER_DFLASH2_DIR:/dflash2:ro")
      ;;
  esac
  # The config.json overlay is this Spark's file; the worker gets it over the same path the launch mounts it from.
  if (( ${#OVERLAY_MOUNT[@]} )); then
    [[ "$DRY_RUN" == 1 ]] || { wssh mkdir -p "$(dirname -- "$WORKER_OVERLAY")" && \
      wssh tee "$WORKER_OVERLAY" < "$STATE_DIR/model-config.json" >/dev/null; } \
      || fail E_WEIGHTS "Could not place the config.json overlay on $(worker_host)."
    WORKER_MOUNTS+=(-v "$WORKER_OVERLAY:/model/config.json:ro")
  fi
}
