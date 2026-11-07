# Two Jetson AGX Thor instead of two DGX Sparks

The recipe was written for two DGX Sparks (GB10, `sm_121`, a ConnectX-7 cable, RoCE). It also runs on two Jetson AGX
Thor (T5000/T4000, `sm_110`, a 100G QSFP cage that is plain Ethernet on the SoC's MGBE controllers, **no verbs device
at all**). Nothing in the recipe has to be told which one it is on: the launcher reads the GPU, the driver and the
link, and takes the path that hardware has. This page says what differs, what is detected, and what to expect.

## What differs

| | DGX Spark (GB10) | Jetson AGX Thor |
| --- | --- | --- |
| GPU compute capability | 12.1 → `SM=121` | 11.0 → `SM=110` |
| Fabric | one ConnectX-7 port, two PCIe twins, two RDMA devices | 100G QSFP cage on 4 MGBE ports (4 × 25G), **no RDMA device** |
| NCCL transport | IB verbs over the cable | socket (TCP) over every rail |
| Engine RoCE all-gathers (`zig/src/cuda/roce.zig`) | available (`ROCE=on`) | not available (`ROCE=off`, set automatically) |
| Docker GPU passthrough | `--gpus all` (toolkit prestart hook, DGX OS) | `--runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all` (JetPack refuses `--gpus`) |
| CUDA floor in the image | `cuda>=13.3` | driver 595.78 reports 13.2: the image's floor is moved to the driver's own version |
| Unified memory | 128 GB | 122 GiB usable |
| SMs / tensor cores | 48 SMs | 20 SMs |

## What the launcher detects

1. **The SASS target.** `SM` defaults to empty/`auto` and is read from the driver
   (`nvidia-smi --query-gpu=compute_cap`): `11.0` → `-Dsm=110`, `12.1` → `-Dsm=121`. The image tag carries it, so a
   Thor image and a Spark image never mix. Override with `SM=110` (or a comma list) when building elsewhere.
2. **The fabric kind.** `rdma_present` on both nodes decides: with an RDMA device on both, NCCL runs over verbs as
   before. Without one (Thor), `scripts/nodes.sh` finds the rails instead — every pair of addresses in one subnet on
   both nodes at or above `FABRIC_MIN_SPEED_MBPS` (default 10000), which keeps a 1G/5G management LAN out — probes each
   with a full-size, unfragmentable ping in both directions, and hands NCCL the surviving list as
   `NCCL_SOCKET_IFNAME`. NCCL pairs its own interfaces with the peer's on the same subnet, so each rail carries its own
   channels. `NCCL_IB_DISABLE=1` is set and no `--device /dev/infiniband` is added.
3. **The GPU passthrough.** `GPU_RUNTIME=auto` asks the daemon: a registered `nvidia` runtime means `--runtime=nvidia`
   (JetPack on Thor), otherwise `--gpus all` (DGX OS on a Spark). Each node of a mixed pair gets its own answer.
4. **The CUDA floor.** The NVIDIA container runtime refuses an image whose `NVIDIA_REQUIRE_CUDA` names a newer CUDA than
   the driver reports. JetPack 7.2 on Thor reports 13.2 against an image built on `nvidia/cuda:13.3.1`, which asks for
   13.3. CUDA 13.x minor version compatibility makes the run supported, so `cuda_compat_args` moves the floor to the
   driver's own version (and keeps the image's other requirements). Nothing is passed when the driver is new enough.

Everything else is unchanged: the same 106 engine patches apply, the same checkpoint, the same `serve` flags, the same
memory budgeting (`/proc/meminfo` + parked pages), the same `CONTEXT=auto`, the same logs and the same `stop.sh`.

## Settings for a Thor pair

Nothing is required. These are the ones worth knowing:

| Setting | Default | On Thor |
| --- | --- | --- |
| `SM` | auto | `110` (detected) |
| `ROCE` | auto | `off` (no verbs device; forced when the fabric is TCP) |
| `GPU_RUNTIME` | auto | `runtime` |
| `FABRIC_MIN_SPEED_MBPS` | `10000` | keeps the 1G/5G LAN out; the QSFP cage's MGBE ports report 25000 |
| `FABRIC_PEER`, `NCCL_SOCKET_IFNAME` | auto | only when the discovery needs help, e.g. `mgbe0_0,mgbe1_0,mgbe2_0,mgbe3_0` |
| `MEMORY_RESERVE_GIB` | `12` | 122 GiB usable here; the model is ~83 GiB a rank, so the default holds |
| `PARALLEL` | `4` | the recipe's default; raise it only after watching rank 0's startup line |

`WORKER` may be either node's address: the fabric is found from the addresses, not from the ssh route, so
`WORKER=test@192.168.100.39` (a rail) and `WORKER=test@192.168.2.39` (the LAN) both work — the rail is faster for the
image and the weights.

## The rails' own all-gathers (TCP_GATHER)

NCCL's socket transport stripes its channels over the four MGBE rails but carries an exchange on one rail's worth of
bandwidth (measured on this pair: a 16 MiB all-gather costs the same with `NCCL_SOCKET_IFNAME=mgbe0_0` as with all
four; a raw TCP stream reaches 1.28 GB/s, which is what NCCL's 2.8 GB/s wire rate is for a two-way exchange). The
engine's own path for a prompt piece's all-gathers (`zig/src/cuda/tcp_gather.zig`, patches 0108-0109) spreads the
exchange over `TCP_GATHER_CONNS` sockets striped `i % rails`, each bound to its own rail's address (rank 0 listens
and checks the peer's address, rank 1 dials from the rail's). The microbenchmark beside the port (`bench/tp_gather/`)
measured 64 MiB a rank at 12.5 ms against NCCL's 48.2, 16 MiB at 4.5 against 12.2.

**The exchange runs on the calling thread, in the Thor EXL3 backend's prefill-transport shape (MGBE-RAILS), rails and all.**
The rail set is the reference's qualified profile: with four rails discovered, rails 0, 2 and 3 serve (`rails = [0, 2, 3]`
in its prefill.py; the fourth is connected and verified but idle), and a smaller topology serves every rail. 32
connections stripe `i % rails`, 8 a rail, each bound to its own rail's address (rank 0 listens and checks the peer's
address, rank 1 dials from the rail's); the stripes tile the 24 serving connections. Patch 0108's first cut staged
gathers asynchronously through device flag words, and that wedged on a cold prompt: the caller's stream owed the
worker a ready word while it also carried the prompt path's NCCL collectives, and the two ranks gated each other into
a stop (reproduced: job 29 of the first prompt, both ranks parked in a copy-stream wait, no socket ever idle, no
timeout ever reached -- the flags, not the wire). Since 0109 the gather is synchronous on the calling CPU between the
pipeline's steps -- stage down ordered after the producer by an event, stripe over the connections, copy back ordered
before the caller's next work by an event -- so no GPU stream waits on a value another thread writes, and an exchange
that stalls is bounded by its poll-tick budget (`TCP_GATHER_TIMEOUT_S`) instead of parking a stream forever: the
failed exchange above stopped both ranks with `TCP all-gather failed: exchange: Setup` after 60 s, loudly. Patterned
round trips at several sizes prove the wire before the first gather (`tcp-gather: round trip ... KiB in ... ms`
under `TF_TCP_GATHER_DEBUG=1`, which also logs every exchange).

Small gathers (decode, verify) always stay on NCCL (`TCP_GATHER_MIN_KB`). Because the path carries all-gathers, the
launcher serves prompts in `PREFILL_REDUCE=gather` when it is on; split's point-to-point sends have no rail path. A
connect failure, a failed round trip or a silent peer turns the path off on both ranks and every later gather falls
back to NCCL. The qualified profile also raises the kernel's socket buffers to 16 MiB on both nodes
(`net.core.rmem_max`/`wmem_max`, as the DS deployment's `prepare_tcp_buffers` does); `start.sh` warns with the command
while they are low.

## Checking it

```sh
DRY_RUN=1 ./start.sh                  # prints the SM, the rails found, each rank's command line
DRY_RUN=1 ./scripts/prepare.sh --check # the link only: rails, addresses, MTU, full-size pings
./start.sh                            # build for sm_110, ship, load, serve
```

A Thor pair's step 4 reads:

```
[I_FABRIC] Link (TCP, no RDMA device): 192.168.100.38 (mgbe0_0,mgbe1_0,mgbe2_0,mgbe3_0) <-> 192.168.100.39 (…) over 4 rail(s);
           NCCL's socket transport carries every exchange, the engine's RoCE all-gathers are off.
[I_ROCE]   No RDMA device on both nodes: NCCL's socket transport carries every exchange over … (the engine's RoCE all-gathers are off).
```

## What to expect

Measured on this pair (two Jetson AGX Thor, TP=2, 1M-token window, FP8 KV on shared pages, q8 dense, DFlash2 drafts,
four 25G rails, greedy, thinking off, 2026-11-04). The Spark pair's numbers in the README do not transfer:

| | two Sparks (GB10) | two Thors |
| --- | ---: | ---: |
| decode, 1 stream | 38.4 tok/s | 17.8 tok/s |
| decode, 2 streams, total | ~58 tok/s | 27.2 tok/s |
| decode, 4 streams, total | 79-81 tok/s | 39.0 tok/s |
| prefill | ~1,600-1,800 tok/s | ~590 tok/s |
| KV pool at 1M context | 4,098 pages (2.1M tokens) | 6,173 pages (3.16M tokens) |
| first start (load) | ~10 min | ~33 min over NFS (see below) |

- **Decode is about half and prefill about a third of the Spark pair's.** Thor has 20 SMs against the GB10's 48, so
  prefill drops furthest; decode is bandwidth-bound and closer, minus what the TCP collectives cost.
- **Collectives cost more.** The engine's RoCE all-gather is a one-shot RDMA write (~10 µs a call on a Spark pair);
  over TCP the same exchange is a kernel socket round trip. Rank 0's log says `p2p gather: off (a rank could not map
  the other's memory); NCCL carries every all-gather`, which is the expected line on two hosts without RDMA.
- **Arch-gated kernels take their fallback.** The patches that gate on `__CUDA_ARCH__ >= 1200` (grouped EXL3
  load-ahead 0027, EXL3 decode 0081) compile their plain path on `sm_110`; the `>= 900` gates (PDL, dependent
  launches) are on. Replies are unaffected: these are scheduling changes, not arithmetic. The GLM and EXL3 kernel
  suites (`tf-cuda-test glm-ref`, `exl3-tiles`, `exl3-prompt-k6`, `act-rows`) and a two-rank NCCL exchange pass on
  `sm_110`.
- **The first start is long, and its cost is the worker's read.** Rank 1 hashes the checkpoint and then loads it, so
  it reads 163.6 GiB twice: 799 s to hash and about as long to load when it comes over NFS v3 (273 MB/s here).
  `WAIT_TIMEOUT` defaults to 2400 s and this pair finished at 2007 s: raise it (`WAIT_TIMEOUT=3600 ./start.sh`) on a
  first start over NFS. A second start reuses the digest cache and is minutes. A worker with room for the checkpoint
  (`WORKER_WEIGHTS=rsync`) loads from local disk instead, which is faster and does not touch the server's page cache.
- **`nconnect` is not available on JetPack's NFSv3 server**, so the recipe's `nfsvers=4.2,nconnect=8` default is
  overridden in `scripts/local.sh` (`NFS_OPTS='vers=3,...'`). NFSv4 is not served by this kernel
  (`/proc/fs/nfsd/versions` is `+3`).
- **`NCCL_SOCKET_NTHREADS` / `NCCL_NSOCKS_PERTHREAD`** are the knobs to try if a rail looks underused; they are not
  set by default. Rank 0's startup line lists every interface NCCL took (`NET/Socket : Using [0]mgbe0_0:… [1]…`).


## Troubleshooting

| Symptom | Cause and fix |
| --- | --- |
| `[E_FABRIC] Found no port at or above 10000 Mb/s …` | The cage is not cabled, or one side has no IPv4 on that subnet. Give both sides an address on one subnet, or lower `FABRIC_MIN_SPEED_MBPS` (0: no floor). |
| `[W_FABRIC] Rail … does not carry a 9000-byte ping` | MTU mismatch or a cable on another port. Both sides must use the same MTU (the cage is 9000 here). |
| `error running prestart hook … invoking the NVIDIA Container Runtime Hook directly (e.g. specifying the docker --gpus flag) is not supported` | `--gpus` on JetPack. `GPU_RUNTIME=auto` avoids it; set `GPU_RUNTIME=runtime` if the daemon's runtime list confuses the probe. |
| `requirements not met: cuda>=13.3 …` | The image's CUDA floor is above the driver's. `cuda_compat_args` handles it; a hand-run container needs `-e NVIDIA_REQUIRE_CUDA=cuda>=13.2`. |
| `NCCL WARN NET/Socket : … ` or a hang at init | Two ranks started differently, or a rail is up without a path. Check `rank1.log`; `NCCL_DEBUG=INFO` for detail. |
| Rank 1 cannot see the weights | On Thor the worker's disk is often smaller than the checkpoint. `WORKER_WEIGHTS=nfs` mounts this node's copy over the rail instead of copying 164 GiB. |

## Tensor cores: tcgen05, and the 228 KiB of shared memory

Thor is not a smaller GB10: `sm_110` is the datacenter Blackwell lineage (10.x) and keeps **tcgen05 / TMEM**, the 5th-gen
tensor core; `sm_121` is the consumer 12.x family and has none. Measured on this pair:

| | Jetson AGX Thor (`sm_110`) | DGX Spark GB10 (`sm_121`) |
| --- | --- | --- |
| `tcgen05.alloc` / `tcgen05.mma` | assembles and runs | rejected by ptxas |
| opt-in dynamic shared memory a block | **227 KiB** (228 KiB an SM) | 99 KiB |
| SMs / CUDA cores | 20 / 2,560 | 48 / 6,144 |

The engine's kernels are all warp-level `mma.sync` (Ampere-style) with `cp.async` staging: nothing in TensorFold at this
pin uses tcgen05, and the recipe's `__CUDA_ARCH__ >= 1200` paths compile out on Thor. So the ranks run the consumer code
path on a datacenter-style SM.

### What that costs, measured

GLM's dense prompt linears, the eight shapes `tf-cuda-test glm-mm-bench` uses, total ms for all eight:

| rows | engine `glm_mm` | cuBLASLt on Thor (dispatches tcgen05) | |
| ---: | ---: | ---: | --- |
| 1 | 1.03 | 0.63 | 1.6x |
| 8 | 1.07 | 0.65 | 1.6x |
| 128 | 1.80 | 0.81 | 2.2x |
| 512 | 6.41 | 1.16 | **5.5x** |

The gap grows with rows, exactly as a tcgen05-versus-`mma.sync` comparison predicts: at prompt row counts cuBLAS is
5.5x the engine's kernel, at decode row counts only 1.6x.

### What was tried here, and what it cost

- **A deeper `cp.async` pipeline on the prompt kernel, paid for with the 228 KiB.** `glm_mm16_rows` and the packed
  `glm_mm{B}_128s2` were instantiated at 2, 3, 4, 6 and 8 stages behind `TF_GLM_MM_PROMPT_STAGES` (8 stages is 193 KiB
  of the block's shared memory). **It lost**: 4 stages was neutral, 8 stages was 8.43 ms against 6.41 at 512 rows and
  2.37 against 1.80 at 128. The kernel is occupancy-bound, not latency-bound: the shared memory buys nothing, and the
  variant was not shipped. (`tf-cuda-test glm-mm-bench`, this pair, 2026-11-04.)
- **The 64 KiB cap on `tf_exl3_group`'s dynamic shared memory** is a host-side attribute the engine sets, not an SM
  limit: a 4,096-row call asks 128 KiB, which exceeds it on every arch. It is a test-harness case, not the serve path
  (the engine chunks at 1,024 rows).

### What actually wins, and its status

Two levers, both of which change the summation order, so replies stop being bit-identical to the `mma.sync` path (the
recipe's gates compare bytes; the float64 suites compare within tolerance and are the right gates for these).

| | state | where |
| --- | --- | --- |
| 1. cuBLASLt for the prompt dense linears | **implemented**, `TF_GLM_MM_CUBLAS=1` (patch 0107), opt-in | this document, below |
| 2. tcgen05 in the EXL3 expert prompt kernels | **not implemented** — plan for a new session | "Item 2" below |

#### Item 1: `TF_GLM_MM_CUBLAS` (implemented, off by default — it did not pay)

Patch 0107 copies every dense linear's rows to a row-major bf16 arena beside its tiles and sends a prompt's matmuls
(>= 128 rows, one K slice) to cuBLASLt. It runs, and it is correct, but measured against the baseline on this pair it
does not earn its memory. Same five fresh 2,547-token prompts each, `PROMPT_CACHE_GIB=2`, `CONTEXT=524288`:

| configuration | model | prefill (median) | decode | |
| --- | ---: | ---: | ---: | --- |
| baseline, no flag | 83.31 GiB | **647-649 tok/s** | 17.8-18.3 tok/s | stable |
| `TF_GLM_MM_CUBLAS=1`, all 282 linears | 90.96 GiB | 674-694 at 3.5K in one run, 548 in another | 17.4 | **hangs on the 2nd request** |
| selective copy, 8 largest linears (1.26 GiB) | 84.57 GiB | 651 tok/s | 18.3 tok/s | stable |

What that says:

- **At 2.5K-token prompts the dense GEMMs are too small a share of a prompt pass for tcgen05 to show.** 651 against
  649 is noise. The +6-8% seen once at 3,502 tokens did not reproduce (the same configuration measured 548-694 across
  runs), so it cannot be claimed.
- **The full arena hangs.** The first request completes, then rank 0 spins at ~8 W in the engine's *next-turn background
  prefill* (`next turn: background prefill of N tokens stepped aside`) and never returns. No environment variable turns
  that path off. It is not the matmul: the same image serves prompts fine with the copy limited to 8 linears.
- Decode is unaffected either way (17.8-18.3 tok/s), as intended: the 128-row gate keeps it on the kernels.

So the switch ships **off**, and the honest summary is that on this model the prompt path is dominated by the EXL3
expert kernels (item 2), not by the dense linears that cuBLAS can take.

#### The zero-copy variant was tried and reverted (2026-11-04)

Instead of copying the dense linears into an arena, the loader can simply *not* tile or pack them — the checkpoint's
bf16 rows are already row-major — and send every matmul to cuBLASLt. It was built and run: 282 linears left plain, model
86.73 GiB (against 90.96 with the arena, 83.58 without the flag), pool 4,100 pages.

It was reverted for two measured reasons:

- **Decode fell from 17.8 to 9.6 tok/s.** A cuBLASLt call per linear per token is host overhead the kernels do not pay;
  the 1.6x it shows at 8 rows in `glm-mm-bench` is lost to ~282 calls a token. (Prefill was unaffected: 595/685 tok/s.)
- **Routing the decode rows to the engine's plain-layout glue tiles instead deadlocked**: both GPUs spun at ~8 W, a
  kernel waiting rather than computing, and requests stopped returning.

So the shipped form keeps the arena and the 128-row gate: prompts take cuBLASLt, decode keeps glm_mm's tiled kernels.

#### Why the arena is not fp8 (probe result)

cuBLASLt on this GPU supports `e4m3 x e4m3` only: an `e4m3` weight matrix with `bf16` activations returns
`CUBLAS_STATUS_NOT_SUPPORTED` from `cublasLtMatmulAlgoGetHeuristic` (`bf16 x e4m3` returns `INVALID_VALUE`). An fp8 arena
therefore forces fp8 activations — W8A8-fp8 — with a per-prompt quantisation and its own quality eval. This recipe
already measured the analogous int8 W8A8 as "+10.5% prefill, one of ten answers wrong" and ships it off; an fp8 arena
would need the same treatment before it could be a default. Probe: `/tmp/fp8probe.cu`-style harness, `sm_110`,
CUDA 13.3.1, 2026-11-04.

### Item 2: tcgen05 in the EXL3 expert prompt kernels (plan for a new session)

This is the other four fifths of a prompt pass, and cuBLAS cannot reach it: 4-bit EXL3 dequantisation is not a BLAS
product. It is a kernel rewrite, not a switch.

**Where the time is.** `tf-cuda-test exl3-prompt-k6` times the 4-bit prompt kernel on this pair at 205 ms for its
largest case (4102 rows) and 46.5 / 29.2 / 19.2 / 9.6 ms for the others; the dense linears (item 1) are the 6.41 ms of
`glm-mm-bench` per 512-row block of eight shapes. That 205 ms is the target.

**What the kernel is today.** `zig/kernels/cuda/exl3_prompt.cuh` and `exl3_experts.cu`: `tf_exl3_grouped_{0,1,2}` run
128 threads a block with 32,832 B of static shared memory, read 4-bit MCG codes from shared memory, dequantise to bf16
with per-group fp16 scales, and accumulate through warp-level `mma.sync.aligned.m16n8k32` in registers, staged with
`cp.async`. No TMEM, no warp specialisation, no `mbarrier`.

**Why tcgen05 is the lever.** `sm_110` keeps the 5th-gen tensor core (tcgen05/TMEM); `mma.sync` cannot reach it, and
cuBLAS's 5.5x at 512 rows on this very GPU is the size of the gap (community per-SM figures: ~5.2 vs ~1.5 TFLOPS/SM,
consistent with it). The recipe's 106 patches gate the `>= 1200` paths out on Thor, so nothing today uses it.

**What the port needs.**

1. **Operand staging.** `tcgen05.mma` reads A and B from shared memory through 64-bit matrix descriptors (base address,
   leading-byte offset, stride-byte offset, swizzle mode). The dequantised bf16 tiles must be written in the descriptor's
   swizzled layout; the dequantise step itself stays, only its destination changes.
2. **TMEM accumulators.** `tcgen05.alloc` a column block per CTA, `tcgen05.mma` against the accumulator address,
   `tcgen05.commit` plus an `mbarrier` to know when it retired, `tcgen05.ld` (32x32b lanes) to move the tile to
   registers for the epilogue, then `tcgen05.dealloc` and `relinquish_alloc_permit`.
3. **Warp specialisation.** One warp issues the MMA (a single thread is enough), the others dequantise and stage the next
   K group; per-stage `mbarrier`s instead of `__syncthreads`.
4. **The 227 KiB.** A 128x64 bf16 tile at N stages fits three times what a GB10 can hold. Do not assume that helps: the
   `mma.sync` prompt kernel was measured occupancy-bound (8 stages: 8.43 ms against 6.41 at 512 rows). Measure.
5. **Numerics.** tcgen05 accumulates in fp32 in its own order: not bit-identical to `mma.sync`. `tf-cuda-test glm-exl3`
   and `exl3-prompt-k6` are the gates (float64 tolerance); greedy replies need one re-baseline.

**How to check it.** `tf-cuda-test exl3-prompt-k6` per-case times (the 205 ms above), then the engine's own prompt path
(cold 4K TTFT: 6.9 s, 590 tok/s today), then `tf-cuda-test glm-exl3` for the numerics.

**Reference points.** This session verified only that `tcgen05.alloc` and `tcgen05.mma` assemble and run on `sm_110a`
and are rejected by `sm_121a` (see the table above); the instruction encodings are the ones CUDA 13's `cuda::ptx`
namespace (`tcgen05_alloc`, `tcgen05_mma`, `tcgen05_commit`, `tcgen05_ld`) or CUTLASS 3.x's SM100 MMA atoms use.
Do **not** start from more `cp.async` stages — that was measured and lost.
