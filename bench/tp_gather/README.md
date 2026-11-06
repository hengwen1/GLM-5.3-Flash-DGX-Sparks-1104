# TP=2 gather microbenchmark: 4-rail TCP vs `ncclAllGather`

Standalone measurement of the exchange this recipe's engine carries on
`tp_group.Group.allGather` (see `zig/src/cuda/tp_group.zig`), at the sizes a
GLM-5.3 prefill piece actually uses. No engine code is changed and the serving
containers are untouched: each rank runs its own short-lived container.

Two paths, same semantics (`bytes` per rank in, `2 * bytes` gathered):

| path | what it is |
| --- | --- |
| `nccl` | `torch.distributed.all_gather_into_tensor`, i.e. `ncclAllGather` — what the engine falls back to on this pair today (`p2p gather: off`, no verbs device) |
| `tcp` | a clone of `deepinfer/backends/thor_exl3/prefill_transport.py`: pinned `cudaHostRegister`'ed staging buffer, `cudaHostGetDevicePointer` zero-copy view, `--conns` TCP connections striped over the 4 rails (`i % 4`), `(seq, len)` header then bytes |

Sizes are per-rank bytes. GLM-5.3 hidden size is 4096 fp32 = **16 KiB a row**, so
`rows = bytes / 16384`:

| rows | bytes/rank | where it comes from |
| ---: | ---: | --- |
| 32 | 512 KiB | `p2p_gather` cap (`TF_GLM_P2P_GATHER_MAX_KB`) |
| 64–256 | 1–4 MiB | 128-row piece; `ROCE_MAX_KB=4096` |
| 1024 | 16 MiB | 1024-row piece |
| 4096 | 64 MiB | one prompt chunk at `TF_GLM_PREFILL_REDUCE=gather` |

## Run

```sh
cd bench/tp_gather
./run.sh                        # both paths, default sweep (512 KiB … 64 MiB)
SIZES=4194304 ./run.sh          # one size
CONNS=16 ./run.sh               # connection count (default 8)
MODE=nccl ./run.sh              # one path
ITERS=20 WARMUP=10 ./run.sh
NCCL_MIN_NCHANNELS=4 NCCL_MAX_NCHANNELS=4 ./run.sh   # the engine's NCCL_CHANNELS=4
```

`run.sh` copies the script to `$WORKER:/tmp/tp_gather_bench`, starts rank 1 there,
runs rank 0 here, and merges both JSON reports into `bench/tp_gather/results/<utc>/`.
Defaults assume this pair: `WORKER=test@192.168.100.39`, rails `.38`/`.39` on
192.168.100–103.0/24, `IMAGE=deepinfer1017-tp2:runtime-0.1.3-arm64` (it has
torch 2.13 + NCCL 2.30; any image with both works).

Timing: `nccl` is `cudaEvent` around one call (plus a back-to-back `chained_us`
loop); `tcp` is `perf_counter` around the blocking exchange plus
`torch.cuda.synchronize()` (the source transport blocks too). Every size is
bitwise-checked on both paths before timing; `stage_us` is the D2H into the
pinned buffer and `net_us` the socket exchange, so a slow copy is not mistaken
for a slow wire.

## Measured on this pair (2026-11-06)

Two Jetson AGX Thor (sm_110), 4 × 25 GbE MGBE rails, MTU 9000, no verbs device,
`CONNS=8`, `ITERS=20`, `WARMUP=10`, NCCL at the engine's `NCCL_CHANNELS=4`.
Wire GB/s = `2 * bytes / time` (both directions counted).

| rows | KiB | NCCL µs | NCCL GB/s | TCP µs | TCP GB/s | TCP stage µs | TCP net µs |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 512 | 1016 | 1.45 | 1307 | 0.80 | 34 | 1178 |
| 64 | 1024 | 1223 | 2.17 | 1331 | 1.58 | 40 | 1153 |
| 128 | 2048 | 1880 | 2.64 | 1553 | 2.70 | 45 | 1490 |
| 256 | 4096 | 3060 | 2.77 | 2129 | 3.94 | 50 | 1974 |
| 512 | 8192 | 6290 | 2.74 | 3007 | 5.58 | 88 | 2738 |
| 1024 | 16384 | 12188 | 2.77 | 4457 | 7.53 | 175 | 3867 |
| 2048 | 32768 | 23567 | 2.85 | 7862 | 8.54 | 328 | 7605 |
| 4096 | 65536 | 48205 | 2.78 | 12536 | 10.71 | 642 | 11254 |

Reference points measured alongside:

- **Raw single-stream TCP** over one rail, 256 MiB: 210 ms = **1.28 GB/s**
  (10.2 Gbps). That is the ceiling a single socket reaches on this MGBE driver.
- **NCCL with one rail vs four**: 16 MiB is 12138 µs with
  `NCCL_SOCKET_IFNAME=mgbe0_0` and 13063 µs with all four. NCCL's socket
  transport is **carrying the exchange on effectively one rail**, despite listing
  all four interfaces (`NET/Socket : Using [0]mgbe0_0 … [3]mgbe3_0`) and creating
  two channels. Its 2.8 GB/s wire is 1.4 GB/s one-way — exactly one rail's TCP.
- **Connection count** at 1 MiB / 4 MiB / 64 MiB (TCP, median µs):
  4 conns 14260 (64 MiB), 8 conns 12647, 16 conns 13985, 32 conns 15506; at 1 MiB,
  8 conns 1219 vs 16 conns 1940 vs 32 conns 2722. The 32 connections of the Python
  source are a cost here, not a benefit.
- `stage_us` never dominates: the pinned copy runs at ~20 GB/s, so the exchange is
  the wire.

## What it says about the port

- The 4-rail TCP exchange is **3.85× NCCL at 64 MiB** (12.5 ms vs 48.2 ms) and
  already 2.7× at 8 MiB. Crossover is ~2 MiB (128 rows); below it the TCP path's
  per-call overhead wins out.
- That overhead is ~1.2 ms of *Python*: 8 threads and 8 `sendall`/`recv_into`
  pairs per call. A Zig implementation with one pinned ring and one event chain
  should remove most of it, so treat the small-size TCP numbers as a floor, not a
  ceiling.
- The rails are used evenly (`rails rx MiB` column: 157–158 MiB each for 64 MiB × 10
  calls), so the striping logic is right; the wire limit is one TCP flow per rail
  (~1.4 GB/s each). More flows per rail is the only way past ~10.7 GB/s wire here.
- NCCL's single-rail behavior explains the engine's prefill cost on this pair. Any
  transport that actually spreads the exchange over four rails starts with a
  ~4× headroom on the collective alone. It says nothing about prefill wall time:
  time the engine's gathers first (they are 4–64 MiB per piece, and a prompt chunk
  issues one per site), because if collectives are a small share of prefill, the
  port buys little.

## Caveats

- `--conns` connections per rank, striped `i % rails`. The source uses 32; this
  pair prefers 8 (measured above).
- The TCP path binds each rank to its own rail addresses; pass
  `--rails`/`--peer-rails` if the pair is numbered differently.
- `/sys/class/net/*/statistics/tx_bytes` is not a usable 64-bit counter on this
  driver (it reads back garbage), so only `rx_bytes` deltas are reported.
- NCCL and TCP both block or sync per call here; the engine enqueues gathers on a
  stream, so real prefill overlaps them. Compare like with like: the `chained_us`
  column is NCCL's back-to-back cost, and it matches the per-call number, i.e. NCCL
  is not hiding anything at these sizes.
- NCCL's small-size numbers vary run to run (4 MiB: 3.1–5.8 ms median across
  runs), so keep `ITERS` at 20 and read the median/p10, not one call. The table
  above is the 20-iteration reference run; `results/` keeps the raw JSON.
