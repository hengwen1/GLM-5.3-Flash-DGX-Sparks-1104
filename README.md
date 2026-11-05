<h1 align="center">GLM-5.3-Flash on DGX Sparks with TensorFold</h1>

<img src=".github/image.png" alt="GLM-5.3-Flash on two DGX Sparks" width="100%">

<p align="center">
  <sub>by <a href="https://x.com/MiaAI_lab">Mia's AI Lab</a></sub>
  <br><br>
  <a href="https://github.com/sponsors/MiaAI-Lab" target="_blank" rel="noopener noreferrer" style="display:inline-block;margin:0 8px;vertical-align:middle;"><img src="https://img.shields.io/badge/Sponsor%20me%20on%20GitHub-181717?style=for-the-badge&logo=githubsponsors&logoColor=white" alt="Sponsor me on GitHub" height="28" style="height:28px;width:auto;vertical-align:middle;border:0;" /></a>
  <a href="https://x.com/MiaAI_lab" target="_blank" rel="noopener noreferrer" style="display:inline-block;margin:0 8px;vertical-align:middle;"><img src="https://img.shields.io/badge/Follow%20me%20on%20X-000000?style=for-the-badge&logo=x&logoColor=white" alt="Follow Mia on X" height="28" style="height:28px;width:auto;vertical-align:middle;border:0;" /></a>
</p>

> [!IMPORTANT]
> **This is a brand-new recipe, version 1104.** It is not an update of
> [GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold)
> (the Python-engine recipe and its `zig-engine` branch, which stay as they are). It descends from the lab's
> [2x RTX PRO 6000 recipe](https://github.com/MiaAI-Lab/GLM-5.3-Flash-2x-RTX-PRO-6000-TensorFold): the same engine pin
> and 93 of its patches unchanged, plus a two-Spark launch over the ConnectX-7 link, unified-memory budgeting and twelve
> GB10 patches, every default measured on the pair. Different engine lineage, different patch stack, different numbers.

This recipe serves GLM-5.3-Flash from an EXL3 checkpoint on **two NVIDIA DGX Sparks** (GB10, 128 GB unified memory
each) at tensor parallel 2, one rank on each Spark, over their ConnectX-7 link. It uses TensorFold's native Zig/CUDA
server (`tensorfold-native`) built in Docker for `sm_121`. The default checkpoint is the pinned 4 bpw EXL3 quant,
downloaded on the first start and mounted read-only. The pair's KV pool holds about 2M tokens in total (fp8 on shared
pages, 12.5 GiB on each rank). Rank 0 serves the OpenAI-compatible API; rank 1 follows its steps.

```sh
./start.sh            # check both Sparks, build the image once, send it to the worker, check the weights, start both ranks, smoke-test
./stop.sh             # stop both ranks, archive the logs (logs/glm53-spark-*.tar.gz)
DRY_RUN=1 ./start.sh  # print the whole plan: the link it finds, the docker build, both ranks' command lines, the health wait
```

The engine is TensorFold at commit `f350cd3e` ([ashhart/TensorFold](https://github.com/ashhart/TensorFold)), plus the
`patches/engine/*.patch` files, compiled with Zig 0.17.0 for `sm_121` by `docker/Dockerfile`.


## Why this engine on a Spark

- `sm_121` is the engine's default `-Dsm` target, GLM at tensor parallel over hosts is an upstream-measured
  configuration, and the engine's **RoCE all-gather** (`zig/src/cuda/roce.zig`) carries the decode and verify
  exchanges as one-shot RDMA writes over the Sparks' cable: same bytes, same bits as NCCL, less time.
- **Unified-memory budgeting** is upstream: the engine budgets from `/proc/meminfo` plus the pages parked on the
  kernel's per-CPU lists, not from a card's free bytes.
- The two single-host optimisations in the stack (peer-store all-gather, copy-engine prompt exchange) step aside across
  hosts instead of failing.

## What is checked

105 patches apply to `f350cd3e` with no rejects; the image builds on aarch64 for `sm_121` (about 6 minutes) and
reaches the worker by content; link discovery finds both PCIe twins of the cabled port and the RoCE v2 GID that
holds each port's own IPv4; NCCL and the engine's RDMA all-gather run over the cable; the full 1,048,576-token window
serves 8 streams beside a 3.6 GiB prompt cache; greedy replies are byte-stable across every change
([CHANGELOG.md](CHANGELOG.md)). Not checked: image input; `W8A8` is stashed on quality.

## Performance

Served configuration: `PARALLEL=8`, `CONTEXT=1048576`, `KV=fp8` on shared pages, `DENSE=q8`, `HEAD=q8`,
`ABSORB_MMA=1`, DFlash2 drafts at 4 bits, `MEMORY_RESERVE_GIB=9.5`, image `8c4ca7bd8c39` (105 patches). The complete
measured record, traps and retractions included, is [docs/speed/RESULTS.md](docs/speed/RESULTS.md).

**Official numbers** (sparkDash, this pair, 2026-11-04). Decode is prose at temperature 0.7 with **thinking on**;
prefill is a cold prompt of the given size:

| Decode, prose, T=0.7, thinking on | 1 stream | 2 streams | 3 streams | 4 streams |
| --- | ---: | ---: | ---: | ---: |
| Total tok/s | **47.8** | **66.5** | **77.0** | **87.3** |
| tok/s per stream | 47.8 | 33.6 | 27.3 | 22.6 |
| TTFT | 200 ms | 324 ms | 437 ms | 557 ms |

| Prefill, prompt tokens | 4K | 8K | 16K | 32K | 64K | 128K | 256K |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| tok/s | 1,601 | 1,688 | 1,767 | 1,801 | 1,812 | 1,778 | 1,708 |
| TTFT | 2.8 s | 5.1 s | 9.5 s | 18.4 s | 36.4 s | 74.0 s | 153.7 s |

**The lab's harness** (`docs/speed/harness/`, greedy, thinking off, fixed prompts, the client on a third Spark), the
shape every engine change is gated on:

| Decode, greedy, thinking off | 1 stream | 4 streams | 6 streams | 8 streams |
| --- | ---: | ---: | ---: | ---: |
| Prose, total tok/s | 38.4 | 79-81 | 94-95 | 105-106 |
| Code, 1 stream | 56.6 | | | |
| Sampled prose, T=0.7, total | 39.1-39.3 | 75-78 | | |

A chat decoding beside a 50K-token prefill keeps 12.9 tok/s with a worst pause of 2.7 s (one 4,096-row chunk). On the
same checkpoint, prompts and day, prose at T=0.7 reads 39.1 / 75-78 tok/s (1 / 4 streams) here against 29.6 / 68-71
on TensorFold's Python engine v0.6.0 and 21.3 / 30-31 on the Zig v1.0.7 branch; under greedy decoding the v1.0.7
branch leads at 4 streams (89-95 against 79-81).

> [!WARNING]
> **Measure cold prefill with fresh prompt content.** The prompt cache resumes a shared prefix and reports a rate
> that is not prefill; `prompt_tokens_details.cached_tokens` is 0 when nothing was resumed.

Prefill was tuned on 2026-11-02 (docs/speed/RESULTS.md): `ABSORB_MMA` is +8.0% with identical answers and is on; the
4,096-row prompt chunk (0094/0095) is +7% with byte-identical replies; `W8A8` reaches +10.5% but rounds activations
(one of ten answers wrong) and stays off; `q4` dense is rejected on quality; the cable is about 3% of a long prompt's
time. At equal dense width the engine is at parity with the v1.0.7 build's prefill.

## Requirements

- **Two DGX Sparks** (GB10, 128 GB, CUDA 13 driver) cabled to each other on their ConnectX-7 ports, with IPv4
  addresses on one subnet. One cable is two netdevs and two RDMA devices (the port's two PCIe links); both are used.
- **Or two Jetson AGX Thor** (sm_110, 122 GiB usable, JetPack 7) cabled on their 100G QSFP cage, with IPv4 addresses on
  one subnet per rail. Thor exposes no RDMA device: NCCL's socket transport carries every exchange over the rails and
  the engine's RoCE all-gathers are off. The launcher reads the GPU, the driver and the link, so nothing has to be set
  by hand; [docs/thor.md](docs/thor.md) is the port's page.
- Key-based ssh from this node to the worker; your login in `docker` on both; Docker with the NVIDIA Container
  Toolkit and BuildKit; `rdma-core` on an RDMA fabric (without it every exchange stays on NCCL, more slowly).
- `curl`, `python3`, `flock`, `tar`, `gzip`, `sha256sum`, `nvidia-smi`, `ssh`, `rsync`, `ip`, `awk`; `hf` for downloads.
- The checkpoint `Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold@76c0b517` (about 164 GiB, downloaded when missing);
  both Sparks read every shard, so the worker gets it over the link (`WORKER_WEIGHTS=rsync`) or by NFS. The default
  drafter `incoai/GLM-5.3-Flash-DFlash2@bf582e4e` is CC BY-NC-ND 4.0: for commercial use set `DRAFTER=mtp` or `none`.
- Disk: about 40 GiB for the image on each Spark, 180 GiB for the checkpoint on each that holds it.

## How it starts

1. **Checks** on both Sparks: Linux aarch64, Docker, the NVIDIA runtime, a GB10, memory as the engine counts it
   (`scripts/memavail.sh`), disk, the ports.
2. **Image**: `docker build` tagged `tensorfold-glm53-spark:<hash>` over the Dockerfile, the patches and the pins;
   skipped when that label exists; the worker gets the same image by content over the link.
3. **Weights**: downloaded when missing, placed on the worker by `WORKER_WEIGHTS`; both ranks mount `/model` and
   `/dflash2`, so their startup fingerprints agree.
4. **The link**: the route to the worker gives the netdev, sysfs the port's RDMA devices and the RoCE v2 GID holding
   each side's IPv4; a second rail is probed with a full-size ping before use.
5. **Launch**: rank 1 on the worker first, then rank 0 here, each PID 1 in its own container with `--network host
   --ipc=host --device /dev/infiniband`; rank 0 serves the API and watches the follower (a rank that dies stops the
   other).
6. **Ready**: `/health`, then `/v1/models`, then a greedy smoke test.

Every rank carries the recipe's `TF_*`/`TENSORFOLD_*`/`NCCL_*` settings; any other such variable you export reaches
both ranks by name (`TENSORFOLD_API_KEY` included, never printed).

### Prefill beside decoding

A long prompt is read in atomic 4,096-row chunks (about 2.5 s each); other streams decode between chunks.
`PREFILL_SHARE=p` makes each chunk owe the decoders `(1 - p) / p` of its own time in decode rounds first. At the
default `0.68` a chat beside a long prefill keeps about 12-13 tok/s with a worst pause of one chunk, and the long
prompt's first token comes about 47% later than at `1`. Replies are byte-identical at any value.

## Memory on a 128 GB unified Spark

- VRAM and system RAM are one pool. The budget is `MemAvailable` plus parked pages, minus `MEMORY_RESERVE_GIB`; per
  Spark on the served pair: model 83.3 GiB, KV pool 12.5 GiB (4,098 pages, 2.1M tokens), prompt cache 3.6 GiB, 9.5 GiB
  reserve kept free.
- **The page pool takes whole 1M windows before the prompt cache**, so a larger room can silently leave the cache at
  zero and every chat turn re-reads its history. The 9.5 GiB reserve holds the pool at two windows; after a start,
  rank 0's startup line must show `4098 pages`, `prompt cache: N GiB` and no `cannot keep one state` warning.
- `HOST_CACHE_GIB=0` on GB10 (a pinned host tier takes the pool's own memory). The memory guard stops the pair in
  order when a Spark stays under 2.5 GiB available. To free memory: lower `PARALLEL`, `DENSE`/`HEAD` bits or
  `VISION=0`; raise `MEMORY_RESERVE_GIB` on out-of-memory errors.

## CONTEXT=auto

The engine refuses a window that does not fit after loading, naming bytes. `CONTEXT=auto` asks for `CONTEXT_MAX`
(1,048,576) and, on a refusal with figures, restarts both ranks at the window those figures allow (times
`CONTEXT_FIT_MARGIN`, the tighter rank winning), up to `CONTEXT_RETRIES` (3) times; a window that served is kept in
`.state/context-fit.tsv` for the same settings. `CONTEXT_RETRY=0` turns it off.

## Settings

Layered: `scripts/config.sh` defaults, then `.env`, then `scripts/local.sh` (trusted Bash, gitignored), then exported
variables. **The served pair's `local.sh`** sets `PORT=8888`, `PARALLEL=8`, `CONTEXT=auto`, `KV=fp8`, `DENSE=q8`,
`HEAD=q8`, `ABSORB_MMA=1`, `HOST_CACHE_GIB=0`, `MEMORY_RESERVE_GIB=9.5`, `MODEL_SOURCE=local`, `WORKER_WEIGHTS=check`.

### The pair

| Setting | Default | Meaning |
| --- | --- | --- |
| `WORKER` | **required** | The second Spark's ssh target on the direct link, e.g. `zurih@10.0.34.4` |
| `TP` | `2` | Two Sparks with `./start.sh`; four cabled as a ring with `./start-tp4.sh` (below) |
| `FABRIC_PEER`, `FABRIC_SELF`, `NCCL_SOCKET_IFNAME`, `TF_ROCE_HCA`, `NCCL_IB_GID_INDEX` | auto | Found from the route to `WORKER`, the port's RDMA devices and its RoCE v2 GID table; on a fabric without RDMA, from the rails' subnets and a full-size ping |
| `NCCL_CHANNELS` | `4` | `NCCL_MIN_NCHANNELS` and `NCCL_MAX_NCHANNELS` |
| `ROCE`, `ROCE_MAX_KB` | `auto`, `4096` | The engine's one-shot RDMA all-gathers, and the largest shard they carry; `off` puts everything on NCCL. Auto: on when both nodes expose an RDMA device |
| `PORT`, `HOST`, `NCCL_PORT` | `8020`, `0.0.0.0`, `29583` | The API address; rank 0's bootstrap port on the link (the watch line uses the next one) |
| `WORKER_WEIGHTS` | `rsync` | `rsync` the checkpoint over the link, `nfs` (the worker mounts it; `NFS_*` settings), or `check` (refuse unless present) |
| `WORKER_MODEL_DIR`, `WORKER_DFLASH2_DIR`, `WORKER_CACHE_DIR` | the same paths | Where the worker keeps them; the home-based defaults re-root on the worker's own home when the two logins differ |
| `SSH_OPTS` | `-o BatchMode=yes -o ConnectTimeout=10` | For every call to the worker |

### Weights and serving

| Setting | Default | Meaning |
| --- | --- | --- |
| `MODEL_SOURCE`, `MODEL_DIR` | `download`, `DATA_DIR/model` | Fetch the pinned 4 bpw checkpoint when missing, or `local` to serve `MODEL_DIR` as it is (a tree with no `.revision` marker is checked by its config.json and shards) |
| `ABLIT` | `0` | `1`: the gated [Ablit weights](#ablit-weights); needs `HF_TOKEN` |
| `DRAFTER`, `DRAFTER_BITS` | `dflash2`, `4` | `dflash2`, `mtp` (the checkpoint's own head) or `none` |
| `PARALLEL` | `4` | `--parallel`; admitted only if it fits, `auto` serves up to 8. The pair runs 8: free at 1-4 streams, 94-95 tok/s at 6, 105-106 at 8 |
| `PREFILL_SHARE`, `PREFILL_BUDGET` | `0.68`, empty | See [Prefill beside decoding](#prefill-beside-decoding) |
| `CONTEXT`, `CONTEXT_MAX` | `auto`, `1048576` | See [CONTEXT=auto](#contextauto) |
| `KV`, `KV_PAGES` | `fp8`, `on` | `--kv-cache`, one shared pool of 512-row pages with `CONTEXT` as one request's cap |
| `DENSE`, `HEAD` | `q8`, `q8` | `bf16`, `q8`, `q6` or `q4` for the dense linears and the LM head |
| `MEMORY_RESERVE_GIB`, `HOST_CACHE_GIB`, `PROMPT_CACHE_GIB` | `12`, `0`, empty | See [Memory](#memory-on-a-128-gb-unified-spark); the pair runs the reserve at 9.5 |
| `VISION`, `VISION_MAX_IMAGES`, `CHAT_TEMPLATE` | `1`, `50`, `auto` | Image input on rank 0's tower, with `templates/glm-5.3-flash-vision.jinja` |
| `THINKING`, `REASONING_EFFORT`, `MAX_TOKENS`, `CLEAR_THINKING` | `0`, `high`, `32768`, `1` | Request defaults; earlier turns' reasoning is dropped from the prompt |
| `ABSORB_MMA`, `W8A8` | `1`, `0` | Prompt numerics: `ABSORB_MMA` is free (+8.0%, identical answers); `W8A8` rounds activations and stays off |
| `CE_EXCHANGE` | `0` | Needs CUDA IPC between ranks, which two Sparks do not have |
| `SERVED_NAME`, `SERVE_EXTRA_ARGS`, `TENSORFOLD_API_KEY` | `GLM-5.3-Flash`, empty, unset | The model id clients send; more `serve` flags; a bearer key for every route but `/health` |

### Build, checks and timeouts

| Setting | Default | Meaning |
| --- | --- | --- |
| `SM`, `ZIG_CPU` | auto, `native` | The GPU's compute capability: `121` on a GB10, `110` on a Thor (read from the driver; set it to build elsewhere). `baseline` builds an image that runs anywhere |
| `GPU_RUNTIME` | `auto` | `runtime` (`--runtime=nvidia`, JetPack on Thor) or `gpus` (`--gpus all`, DGX OS on a Spark); auto picks per node |
| `FABRIC_MIN_SPEED_MBPS` | `10000` | On a TCP fabric (no RDMA device), the speed floor a rail must reach; keeps a 1G/5G LAN out |
| `TF_REPO`, `TF_REF`, `ZIG_VERSION`, `ZIG_SHA256`, `NCCL_WHEEL` | pinned | All hashed into the image tag |
| `IMAGE`, `IMAGE_REPO`, `DOCKER_BUILD_FLAGS`, `CONTAINER_NAME` | `tensorfold-glm53-spark:<hash>`, `glm53-spark` | |
| `MEMGUARD`, `MEMGUARD_FLOOR_GIB`, `MEMGUARD_HOLD_S`, `MEMGUARD_INTERVAL_S`, `MIN_AVAIL_GIB` | `1`, `2.5`, `4`, `2`, `100` | See [Memory](#memory-on-a-128-gb-unified-spark) |
| `DATA_DIR`, `CACHE_DIR`, `LOG_DIR`, `LOG_KEEP` | `~/.cache/aevonix-glm53`, `DATA_DIR/zig-home`, `logs`, `10` | Downloads, the ranks' `HOME`, run logs |
| `WAIT_TIMEOUT`, `SMOKE_TIMEOUT`, `STOP_TIMEOUT` | `2400`, `180`, `45` | Seconds |
| `DRY_RUN` | `0` | `1` prints the plan and changes nothing |

## Engine patches

`patches/engine/` holds 105 patches on TensorFold `f350cd3e`, applied in filename order at image build: 0001-0093 are
the [RTX PRO 6000 recipe](https://github.com/MiaAI-Lab/GLM-5.3-Flash-2x-RTX-PRO-6000-TensorFold)'s, unchanged;
0094-0105 are this pair's own or ported from that recipe. None is GB10-specific: the arch-gated kernels test
`__CUDA_ARCH__ >= 1200`, which `sm_121` satisfies.

| Patches | What they do |
| --- | --- |
| 0001–0012 | EXL3 experts at mixed bit widths, 3-bit prompt kernels; FP8 attention loads, the shared expert on a side stream; one expert call a prompt pass, no MTP pass beside DFlash2 |
| 0013–0016 | DFlash2 reads complete prompt taps at TP>1 (upstream #1623); no target MTP layer beside DFlash2; a pinned host prompt-state tier (off here) |
| 0017–0033 | FP8 KV on shared pages; EXL3 load-ahead, fused MoE glue, KDA in one launch, one broadcast a follower step |
| 0034–0035 | Peer-store all-gather and copy-engine prompt exchange, both stepping aside across hosts |
| 0036–0048 | DFlash2 copy rounds, `ABSORB_MMA` (opt-in), "fits:" refusals, the device prompt-cache share |
| 0049–0066 | Paired prompt chunks (upstream #1423) extended to DFlash2; `W8A8` (opt-in) |
| 0067–0091 | DSA select on many CTAs, audited programmatic dependent launches along the decode chain, glue fusions, EXL3 prompt and decode kernels; each keeps an off switch (`TF_GLM_DSEL=0`, `TF_GLM_PDL_CHAIN=0`, `TF_GLM_MM_TWO=0`, `TF_GLM_TOPK_SUM=0`, `TF_GLM_SHARED_SEGS=0`, `TF_EXL3_SHFL=0`, `TF_EXL3_PDL=0`, `TF_GLM_KDA_STEP_WARPS=8`, `TF_GLM_HC_SPLIT=0`, `TF_GLM_SHARED_WRITES=0`, `TF_GLM_HC_PART=0`, `TF_GLM_DSCORES=0`). Never ship 0086 or 0089 without 0090 |
| 0092–0093 | A JSON answer stays JSON beside offered tools on chat, Responses and Messages; a reply grammar is freed when its stream cannot start |
| 0094–0097 | **This pair's.** A 4,096-row prompt chunk with DFlash2's `MAX_CTX` to match (+7% prefill, byte-identical replies); the GB10's measured cost tables and draft-rule seed |
| 0098–0102 | **This pair's,** env-gated, defaults behaviour-identical: `TF_GLM_DFLASH2_EDGE`, `TF_GLM_PROF` on shared windows, `TF_GLM_MM_STAGES`, `TF_GLM_DFLASH2_TCONF`, `TF_GLM_DFLASH2_Q32` (the decode campaign's exact knobs, each measured neutral) |
| 0103–0104 | From the RTX recipe: a tool call after prose that quotes a tool-call opener is the call (with a guard so a call quoted inside a call's value is never invoked); GLM warm turns, so a user message after an agent loop resumes past the loop (8,192 of 9.3K tokens cached, was 4,096) |
| 0105 | An untyped tool parameter takes the JSON object or array the model wrote; text stays text (sibling recipe issue #180) |

Every default-on patch was A/B tested against the image before it: greedy replies identical plus a measured gain.

## 4 Sparks (TP=4 on a ring, experimental)

`./start-tp4.sh` runs the same image and settings as tensor parallel over four Sparks cabled to each other without a
switch: each Spark's two ConnectX-7 ports to its two neighbours, one subnet per cable (a port and its PCIe twin on it),
the two Sparks across the ring from each other reachable only through a neighbour (an address routed over the cables,
or the LAN). Rank 0 is this Spark (the API); `WORKER`, `WORKER2` and `WORKER3` are the other three **in the ring's
order** from here (`WORKER` and `WORKER3` cabled to this Spark, `WORKER2` across). `./stop-tp4.sh` stops the four.
`DRY_RUN=1 ./start-tp4.sh` prints the ring it finds and every rank's command, and changes nothing.

What differs from the pair (`scripts/tp4.sh`):

- **The ring.** Each Spark's RoCE addresses are read over ssh and paired by subnet; the launcher refuses a switch, a
  ring that is not in `WORKER` order, or a missing cable. NCCL gets `NCCL_ALGO=Ring` and a graph file per rank whose
  channels sit on the devices toward the previous rank, with subnet-aware routing (`NCCL_IB_SUBNET_AWARE_ROUTING=1`,
  `NCCL_IB_SUBNET_PREFIX_LEN` set so a port and its twin both carry traffic) moving each send onto the next rank's
  cable; NIC fusion, P2P and SHM transports are off. NCCL connects channels at first use
  (`NCCL_RUNTIME_CONNECT=1`): its tree for four ranks joins rank 0 to rank 2, which no cable reaches, so connecting
  everything at init, as the engine asks by default, times out in `ibv_modify_qp`. The engine's own exchanges run as
  all-gathers between neighbours (`PREFILL_REDUCE=gather`; `ring` passes the split prefill's rows along the ring and
  is the next thing to measure); the two-rank RoCE path does not apply and `ROCE` is off. Engine patch 0106 makes the
  start's peer warm-up follow that mode instead of posting to every rank.
- **Bootstrap.** `NCCL_COMM_ID` and the ranks' watch lines go to `MASTER_ADDR` (what this Spark's name resolves to; an
  address every worker reaches, through a neighbour for the Spark across); NCCL's bootstrap socket on each Spark's
  default-route netdev (`SOCKET_IFNAME`).
- **Weights.** Rank 0 reads its own `MODEL_DIR`; the workers mount it over NFS from this Spark's address on each
  worker's route (`WORKER_WEIGHTS=nfs`, the default here; `check` for workers with their own copy). When the NFS
  server exports another view of the path (a container exporting `/huggingface` for `~/.cache/huggingface`), set
  `NFS_MODEL_PATH` and `NFS_DFLASH2_PATH` to the exported paths; `NFS_SERVER1..3` override the per-worker address.
  Every rank still reads every shard: three workers over NFS makes the first load slower than the pair's.
- **Memory.** Each rank holds a quarter of the routed experts, the KV pool is replicated on every rank, so the page
  planner has far more room: eight 1M-token windows and a prompt cache usually fit at the recipe's default reserve.
  Read rank 0's startup line after every start (the pages, the prompt cache, no "cannot keep one state").

Measured 2026-11-04 on this kit (spark1 rank 0, image `74e19a63a9a5`, 8 streams, 1M window, FP8 KV on pages, q8
dense and head, DFlash2; greedy, the same harness as the pair's numbers under [Performance](#performance)):

| | two Sparks (TP=2) | four Sparks (TP=4, a ring) |
| --- | ---: | ---: |
| 1 stream, prose / code | 38.4 / 56.6 tok/s | 56.1 / 85.0 tok/s |
| 4 streams, prose, total | 79-81 tok/s | 115-120 tok/s |
| 8 streams, prose, total | 105-106 tok/s | 163-166 tok/s |
| prefill 16K / 64K | ~1,600-1,800 tok/s | 1,752 / 1,889 tok/s |
| KV pool | 4,098 pages (2.1M tokens) | 16,392 pages (8.4M tokens), 3.6 GiB prompt cache |

Rank 0 loads 45.6 GiB; each rank's NCCL connects only to its two neighbours. Greedy replies are repeatable run to
run and the same with drafts off; they are not the pair's tokens (a different partition rounds differently).

Settings beside the pair's: `WORKER2`, `WORKER3`, `MASTER_ADDR`, `SOCKET_IFNAME`, `PREFILL_REDUCE` (`gather`),
`RING_NCCL_ALGO` (`Ring`), `NFS_MODEL_PATH`, `NFS_DFLASH2_PATH`, `NFS_SERVER1..3`. Logs: `logs/current/rank0..3.log`.

## Logs and stopping

`logs/current` holds `rank0.log` and `rank1.log`, collected from each Spark's Docker output. `./stop.sh` stops rank 0
first (it drains for `TF_DRAIN_S`, 20 s), then rank 1, archives both logs as `logs/glm53-spark-<time>.tar.gz`, and
only ever stops containers labelled `tf.recipe=glm53-spark`.

## Troubleshooting

| Symptom | Cause and fix |
| --- | --- |
| `[E_WORKER]` | ssh, Docker or the NVIDIA runtime is not ready on the worker; the message names which |
| `[E_FABRIC]` | No directly reached address, or no RoCE v2 GID holding an IPv4 on the link's port: cable the Sparks on one subnet, or set `FABRIC_PEER` / `NCCL_IB_GID_INDEX` |
| `[W_FABRIC]` a rail left out | The second rail does not carry a full-size ping; check the cable, the twin's address and MTU |
| `[E_MEMORY]`, `[E_PORT]` | Less than `MIN_AVAIL_GIB` available (another rank, usually), or a port in use |
| `RankMismatch`, `RankNeverCame`, NCCL timeouts | The ranks were started differently, or rank 1 did not reach rank 0 within 300 s: check `rank1.log` and the ports; `NCCL_DEBUG=INFO` for detail |
| `vendor_err 0x81` | A rail without a path to its partner was set by hand through `TF_ROCE_HCA` |
| A long prompt dies in a RoCE gather | The gather has no wait setting at this pin; `ROCE=off` puts every exchange on NCCL |
| `--parallel N needs … GiB`, `[memguard]` | The window or stream count does not fit, or a Spark went under the floor: lower `CONTEXT`/`PARALLEL` or raise `MEMORY_RESERVE_GIB` |
| No prompt cache in the startup line | The page pool took a third window: raise `MEMORY_RESERVE_GIB` (see [Memory](#memory-on-a-128-gb-unified-spark)) |
| The first start takes long | Each rank hashes all 164 GiB once (cached in `CACHE_DIR`); raise `WAIT_TIMEOUT` |

## Ablit weights

`ABLIT=1` serves
[Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold-Ablit](https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold-Ablit),
the same checkpoint abliterated, at the revision pinned in `scripts/config.sh`. The repository is gated: `start.sh`
checks that `HF_TOKEN` reaches it before building or downloading. Thinking off, the recipe's default, is also how
these weights give their best results.

## Credits

TensorFold and its native engine are by Ash Hart and contributors
([ashhart/TensorFold](https://github.com/ashhart/TensorFold)). The GLM-5.3-Flash EXL3 checkpoints are by Mia's AI Lab,
from zai-org's GLM-5.3-Flash. DFlash2 is by incoai. The two-host launch, the link discovery and the unified-memory
accounting are adapted from Mia's AI Lab's
[GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold)
(branch `zig-engine`), and the engine's RoCE transport is the GPU side of b12x's "RoCEnante" exchange
([local-inference-lab/b12x](https://github.com/local-inference-lab/b12x), Apache 2.0).

Patches 0103-0104 are ported from the lab's own [2x RTX PRO 6000 recipe](https://github.com/MiaAI-Lab/GLM-5.3-Flash-2x-RTX-PRO-6000-TensorFold).
Full credits, including the authors of the carried upstream patches (Adam Hillson, ajensenwaud, Bizuayeu,
SvangenStudios, with their names in each patch's `From:` line), are in [CREDITS.md](CREDITS.md). Required notices are
in [NOTICE.md](NOTICE.md).

## License

Apache License 2.0: see [LICENSE](LICENSE). Downloaded software and weights keep their own terms; DFlash2's weights are
CC BY-NC-ND 4.0.
