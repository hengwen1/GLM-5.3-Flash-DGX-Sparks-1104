# Running on two Jetson AGX Thor (`sm_110a` + `/dev/mgbe_zc`)

This recipe was written for two DGX Sparks (GB10, `sm_121`, ConnectX-7 RoCE). On two **Jetson AGX Thor** nodes
(`sm_110a`, 4 × 25 GbE MGBE ports in the 100G QSFP cage, no RDMA verbs device), the launcher detects the platform
automatically and enables both the `sm_110a` kernel optimizations and zero-copy 4-rail `/dev/mgbe_zc` TP=2
communication.

## Hardware & launcher differences

| | DGX Spark (GB10) | Jetson AGX Thor |
| --- | --- | --- |
| GPU compute capability | `12.1` → `SM=121` | `11.0` → `SM=110a` |
| EXL3 prompt kernel | `mma.sync` | `tcgen05` 5th-gen tensor core + TMEM (`sm_110a`, patch `0019`) |
| EXL3 grouped decode load | `AHEAD` (`__CUDA_ARCH__ >= 1200`) | `AHEAD` (`__CUDA_ARCH__ >= 1100`, patch `0017`) |
| Interconnect | ConnectX-7 RoCE (`libibverbs`) | 4 × 25 GbE MGBE Ethernet (`mgbe0_0..mgbe3_0`, MTU 9000) |
| Engine TP=2 all-gather | `roce.zig` (`TF_ROCE=1`) | `/dev/mgbe_zc` 4-rail zero-copy (`TF_MGBE_ZC=1`, patch `0018`) |
| NCCL fallback / bootstrap | IB verbs | Socket (TCP) across all 4 MGBE rails (`NCCL_IB_DISABLE=1`) |
| Docker GPU passthrough | `--gpus all` | `--runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all` |
| CUDA runtime floor | `cuda>=13.3` | Relaxed automatically to host driver CUDA (`13.2` on JetPack 7.2) |

## What the launcher detects automatically

1. **Architecture (`SM=auto`).** `nvidia-smi --query-gpu=compute_cap` returning `11.0` selects `SM=110a` (enabling
   architecture-specific `tcgen05.*` SASS in `exl3_experts.cu`).
2. **Fabric & rails.** Without an RDMA device on both nodes, `scripts/nodes.sh` discovers all point-to-point subnet
   pairs at or above `FABRIC_MIN_SPEED_MBPS` (default `10000`, excluding 1G LAN), validates each rail in both
   directions with unfragmented jumbo pings (`ping -M do -s 8972`), sets `NCCL_SOCKET_IFNAME` to the surviving rails,
   sets `NCCL_IB_DISABLE=1`, and forces `ROCE=off`.
3. **Zero-copy `/dev/mgbe_zc` (`MGBE_ZC=auto`).** When `/dev/mgbe_zc` is present on both nodes (or `MGBE_ZC=on`),
   `scripts/nodes.sh` maps the surviving `mgbe<N>_0` netdevs to `TF_MGBE_ZC_PORTS=0,1,2,3`, detects the host group GID
   of `/dev/mgbe_zc` on each node, passes `--device /dev/mgbe_zc --group-add <gid>` to the non-root container, and sets
   `TF_MGBE_ZC=1`, `TF_MGBE_ZC_MAX_KB=16384`, `TF_MGBE_ZC_EAGER_KB=32`, and `TF_GLM_PREFILL_REDUCE=gather`.
4. **GPU runtime & CUDA floor.** `GPU_RUNTIME=auto` picks `--runtime=nvidia` when the `nvidia` runtime is registered in
   Docker, and `cuda_compat_args` relaxes `NVIDIA_REQUIRE_CUDA` when JetPack's driver reports an older minor CUDA
   version (e.g. `13.2`) than the `13.3.1` base image.

## Host setup for `/dev/mgbe_zc` (both nodes)

The `/dev/mgbe_zc` kernel module ([hengwen1/mgbe_zc](https://github.com/hengwen1/mgbe_zc)) exposes direct userspace
VDMA queues (`VDMA 2` for `0x88B5` control/eager frames and `VDMA 3` with hardware split-header DDP for `0x88B6` bulk
frames) while leaving normal IP/SSH/NFS/NCCL traffic on `VDMA 0/1`.

On **both** Jetson AGX Thor nodes:

```bash
# 1. Build and load mgbe_zc.ko (requires DTB nvidia,dma-chans = <0 1> on mgbe0..mgbe3)
git clone https://github.com/hengwen1/mgbe_zc.git
cd mgbe_zc && make && make tests

# 2. Allow non-root container access via dedicated group + udev rule
sudo groupadd -f mgbe_zc
sudo usermod -aG mgbe_zc "$USER"
echo 'KERNEL=="mgbe_zc", GROUP="mgbe_zc", MODE="0660"' | sudo tee /etc/udev/rules.d/99-mgbe-zc.rules
sudo udevadm control --reload-rules

# 3. Load module and verify device permissions + self-test
sudo insmod ./mgbe_zc.ko
ls -l /dev/mgbe_zc   # crw-rw---- 1 root mgbe_zc
./tests/zc_hw_selftest --ports 0,1,2,3
```

## Engine patches for Thor (`0017`–`0019`)

- **`0017-glm-cuda-sm110a-exl3-ahead-and-TF_GLM_MM_CUBLAS.patch`**:
  - Enables `GROUPED_LOAD = tf_exl3::AHEAD` for `__CUDA_ARCH__ >= 1100` in `exl3_experts.cu`.
  - Adds `-gencode=arch=compute_110,code=sm_110` to `mailbox.cu` fatbin.
  - Ports opt-in `TF_GLM_MM_CUBLAS=1` (off by default, `TF_GLM_MM_CUBLAS_MIN_ROWS=128`) with 1:1 `plain_keys`/`plain_vals`
    tracking (fixing mixed `head_bits=16` / `dense_bits=8` lookup alignment) and per-stream cuBLASLt workspaces.
- **`0018-cuda-mgbe-zc-4-rail-tp2-all-gather.patch`**:
  - Implements `zig/src/cuda/mgbe_zc.zig` + `mgbe_zc_uapi.zig`, wired into `tp_group.Group` and `cuda_forward_pieces.zig`.
  - Allocates the 2-slot staging region via CUDA VMM `CU_MEM_LOCATION_TYPE_HOST_NUMA` (directly accessible by GPU
    `roce_gather` kernels and `/dev/mgbe_zc` `FOLL_LONGTERM` contiguous 48-bit SMMU DMA).
  - Reuses `kernels.roce_gather` on the CUDA stream so decode/verify all-gathers are **100% CUDA-Graph capturable**.
  - Uses 0.5-RTT raw L2 eager frames for small/decode stripes (`<= 32 KiB/rail`) and hardware split-header DDP (`SPH=1`,
    8 KiB payload landing directly into CUDA `HOST_NUMA`) for bulk/prefill stripes (`<= 16 MiB/rank`).
- **`0019-exl3-cuda-sm110a-tcgen05-prompt-kernel.patch`**:
  - Implements `tmain_tcgen05` + `tspill_tcgen05` in `zig/kernels/cuda/exl3_prompt.cuh` and `exl3_experts.cu`, gated on
    `__CUDA_ARCH__ == 1100 && defined(__CUDA_ARCH_FEAT_SM110_ALL)` (`sm_110a` only).
  - Maps EXL3 MCG `decode_tile_p` (`K2=8` 4-bit and `K2=6` 3-bit) directly into canonical 8×8 core matrices with zero
    shuffles and zero shared-memory bank conflicts (`lane == rl * 4 + kp`), stages `X` directly into canonical core
    matrices via `cp.async`, accumulates two `128×64` FP32 tiles in 128 TMEM columns via `tcgen05.mma.cta_group::1.kind::f16`,
    and spills via dual-warpgroup `tcgen05.ld.sync.aligned.32x32b.x16.b32` into the existing `fwht128` + SwiGLU epilogue.
  - Fits inside the existing static shared memory footprint (`43.3 KiB`, 52–58 registers/thread, 0 spill bytes).

## Measured kernel & transport gates on 2× Jetson AGX Thor

### 1. EXL3 prompt kernel (`tf-cuda-test exl3-prompt-k6`, 1 layer, 4096 rows, top-8 of 288 experts, D=4096, N=1024)

| Kernel path | 4-bit / 4-bit (one call) | Gate+Up | Down | 3-bit / 4-bit | 3-bit / 3-bit |
| --- | --- | --- | --- | --- | --- |
| Baseline `mma.sync` (`sm_110`) | ~205 ms | — | — | — | — |
| **`tcgen05` (`sm_110a`, patch `0019`)** | **64.61 ms** (**3.17× faster**) | **42.88 ms** | **21.38 ms** | **65.97 ms** | **66.31 ms** |

Both `tf-cuda-test exl3-prompt-k6` (max 0.0702 of float64 tolerance) and `tf-cuda-test glm-exl3` (max 0.198 of float64
tolerance across 4-bit, 3/4-bit, and 2/3/4-bit with NaN isolation) pass.

### 2. 2-node TP=2 all-gather (`tf-cuda-test mgbe-zc-gather`, 4 × 25 GbE rails, non-root container)

All 14 sizes (`64 B`..`16 MiB`), CUDA Graph capture + 32× replay, and 200 eager + 16 DDP stress exchanges passed
bit-exact against NCCL (`0` mismatches, **`0` new `mmc_rx_fifo_overflow`** across all 8 MGBE ports):

| Shard size / rank | `mgbe_zc` (4 rails) | NCCL Socket (4 rails) | Speedup |
| --- | --- | --- | --- |
| 4 KiB (decode) | **13.0 µs** (2.51 Gbps/dir) | 192.9 µs (0.17 Gbps/dir) | **14.8×** |
| 16 KiB (decode) | **16.7 µs** (7.83 Gbps/dir) | 308.7 µs (0.42 Gbps/dir) | **18.5×** |
| 32 KiB (verify) | **19.1 µs** (13.71 Gbps/dir) | 432.1 µs (0.61 Gbps/dir) | **22.6×** |
| 64 KiB (verify) | **22.9 µs** (22.90 Gbps/dir) | 452.4 µs (1.16 Gbps/dir) | **19.8×** |
| 2 MiB (prefill) | **259.0 µs** (64.79 Gbps/dir) | 1,517.0 µs (11.06 Gbps/dir) | **5.9×** |
| 8 MiB (prefill) | **1,061.0 µs** (63.25 Gbps/dir) | 5,943.8 µs (11.29 Gbps/dir) | **5.6×** |
| 16 MiB (prefill) | **2,112.9 µs** (63.52 Gbps/dir) | 11,090.6 µs (12.10 Gbps/dir) | **5.2×** |

### 3. Live TP=2 serving (`./start.sh`, DFlash2 4-bit, FP8 KV pool, Q8 dense, `sm_110a` + `/dev/mgbe_zc`)

Measured greedy (`temperature=0`, `enable_thinking=false`, fresh prompts, `0` new NIC RX FIFO overflows):

| Workload | Result | Notes |
| --- | --- | --- |
| 1-stream decode (256 tok) | **31.1 tok/s** (8.95 s, TTFT 767 ms) | vs ~17.8 tok/s over NCCL TCP (**1.75× faster**) |
| 2-stream decode (2 × 256 tok) | **35.5 tok/s agg** (20.2 tok/s/stream) | 14.41 s wall |
| 4-stream decode (4 × 256 tok) | **38.9–46.7 tok/s agg** (12.8–13.2 tok/s/stream) | 21.9–26.4 s wall |
| Cold ~4K prefill (3,948 tok) | **488.9 tok/s** (TTFT 8.08 s) | `TF_GLM_MM_CUBLAS=0`, 2048-row passes |
| Cold ~8K prefill (8,429 tok) | **479.2 tok/s** (TTFT 17.59 s) | Exact retrieval, 0 RX FIFO overflows, stays healthy |

## Tuning knobs

| Setting | Default on Thor | Notes |
| --- | --- | --- |
| `SM` | `auto` (`110a`) | Override with `SM=110` to disable `tcgen05` |
| `MGBE_ZC` | `auto` (`on` when `/dev/mgbe_zc` present) | Set `MGBE_ZC=off` to fall back to NCCL Socket for A/B testing |
| `MGBE_ZC_MAX_KB` | `16384` | Max shard size routed over `/dev/mgbe_zc` (16 MiB/rank) |
| `MGBE_ZC_EAGER_KB` | `32` | Per-rail threshold between 0.5-RTT eager and hardware DDP |
| `TF_GLM_MM_CUBLAS` | `0` (off) | Opt-in cuBLASLt dense prompt GEMMs (requires ~7 GiB extra VRAM) |
