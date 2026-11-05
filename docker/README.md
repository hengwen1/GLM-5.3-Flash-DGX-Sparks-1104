# docker/

`docker/Dockerfile` builds TensorFold's native (Zig) CUDA server from source: upstream at a pinned commit, plus
`patches/engine/*.patch`, with kernels for one GPU architecture (`SM`, empty by default in the launcher, which reads
the GPU: 121 for a DGX Spark's GB10, 110 for a Jetson AGX Thor). The build context is the repository root. The root
`.dockerignore` sends only `patches/engine/` (a few bytes), so `logs/`, `.state/`, `.git` and models stay out.

```sh
docker build -f docker/Dockerfile \
  --build-arg TF_REPO=https://github.com/ashhart/TensorFold.git \
  --build-arg TF_REF=f350cd3ea253a042e205746928271f9d8f71a82d \
  --build-arg ZIG_VERSION=0.17.0 \
  --build-arg ZIG_SHA256=9e8d11661d4ae3bd57702a3832781e23ad151dde5798e16a5ccd503f65234ff8 \
  --build-arg SM=121 --build-arg PATCHES_HASH=dev \
  -t tensorfold-glm53-zig:dev .
docker run --rm --gpus device=1 tensorfold-glm53-zig:dev tensorfold-native --version
# On JetPack (Jetson AGX Thor) `--gpus` is refused; use the registered runtime and, when the driver's CUDA is older
# than the image's floor, relax that floor to the driver's version:
docker run --rm --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all -e 'NVIDIA_REQUIRE_CUDA=cuda>=13.2' \
  tensorfold-glm53-zig:dev tensorfold-native --version
```

## Build args

| Arg | Default | Meaning |
| --- | --- | --- |
| `TF_REPO` | upstream GitHub URL | Repository to fetch. |
| `TF_REF` | required | Commit fetched at depth 1. A 40-hex ref must equal the fetched `HEAD`, or the build fails. |
| `ZIG_VERSION` | `0.17.0` | Must equal the checkout's `.zig-version`. `build.zig` refuses any other Zig. |
| `ZIG_SHA256` | required | sha256 of `zig-${ZIG_ARCH}-linux-${ZIG_VERSION}.tar.xz` (`ZIG_ARCH` is `aarch64` here), checked before unpacking. `ZIG_URL` overrides the download URL. |
| `SM` | `121` | SASS targets for nvcc (`-Dsm`), comma separated, for example `120,121`. RTX PRO 6000 Blackwell is 120, a DGX Spark's GB10 is 121, a Jetson AGX Thor is 110. The launcher passes the GPU's own value. |
| `PATCHES_HASH` | `dev` | Goes into the `tf.patches` label. The scripts compare it to decide whether to rebuild. |
| `TOOLS` | `1` | `1` also runs `zig build install`: `tf-cuda-test`, `tf-tp-test`, `tf-tp-watch-check` and the CUDA dev CLI. `0` builds only the server and grammar library, about 4 minutes faster. |
| `ZIG_CPU` | `native` | `-Dcpu` for host code. `native` targets the build machine's CPU, so build the image on a DGX Spark. Set `baseline` if the image will run on another aarch64 machine. |
| `ZIG_JOBS` | all cores | Passed as `-jN`. Each big Zig compile takes 2 to 6 GB RSS, and a few run at once. |
| `STRIP` | `false` | `-Dstrip`. `false` keeps debug info for panic traces. |
| `CUDA_VERSION` | `13.3.1` | Tag of both `nvidia/cuda` stages (devel for the build, runtime for the image). 13.3.1 matches the host's nvcc 13.3.73. |

Patches go in `patches/engine/`. Only `*.patch` files are used, applied in byte order of filename (`0001-…`, `0002-…`).
Each one gets `git apply --check` and then `git apply`. If any patch does not apply, the build fails. With no patches,
upstream builds as it is. `.gitkeep` keeps the directory in git so the `COPY` always has a source.

## Stages

1. `toolchain`: `nvidia/cuda:13.3.1-devel-ubuntu24.04` (nvcc 13.3, g++ 13), git, cmake and nasm, and Zig with its
   sha256 checked.
2. `codecs`: builds libjpeg-turbo 3.1.4.1 and libwebp 1.6.0 from tarballs with pinned sha256. The server refuses
   JPEG and WebP images (`--vision`) unless it finds exactly these releases (it checks their version strings).
   Ubuntu's packages are other releases. PNG needs no library.
3. `build`: xgrammar v0.2.8 (commit `97787376…`, with submodules), then TensorFold fetched and patched, then
   `zig build [install] native grammar -Doptimize=fast -Dsm=$SM -Dnvcc=/usr/local/cuda/bin/nvcc -Dxgrammar=…
   --prefix /opt/tensorfold`. Every build compiles all kernels and executables again, so expect the full time after
   any change to `TF_REF` or the patches. The reason is upstream's build: it runs `nvcc --version` as a side-effect
   step, and that output feeds every fatbin's cache key, so Zig's cache never hits across invocations. This is the
   same on the host. Only Zig's global cache (build runner, compiler_rt, libc++) is kept, in the BuildKit cache mount
   `tensorfold-zig-global`.
4. `runtime`: `nvidia/cuda:13.3.1-runtime-ubuntu24.04` plus `/opt/tensorfold`. Labels: `tf.patches`, `tf.engine=zig`,
   `tf.ref`, `tf.repo`, `tf.sm`, `tf.zig`. `WORKDIR /workspace`.

## What is in the image

```
/opt/tensorfold/native/bin/tensorfold-native   the server (serve, pull, models, info, capabilities, benchmark)
/opt/tensorfold/native/lib/libtfgrammar.so     structured output and required tool calls (xgrammar 0.2.8)
/opt/tensorfold/native/lib/libturbojpeg.so.0   JPEG inputs (libjpeg-turbo 3.1.4.1)
/opt/tensorfold/native/lib/libwebpdecoder.so.3 WebP inputs (libwebp 1.6.0)
/opt/tensorfold/bin/{tensorfold,tf-cuda-test,tf-tp-test,tf-tp-watch-check}   with TOOLS=1
/opt/tensorfold/BUILD-INFO                     repo, ref, resolved commit, Zig, nvcc, SM, patch sha256s
/opt/tensorfold/LICENSES/                      TensorFold, xgrammar, dlpack, libjpeg-turbo, libwebp notices
```

On `PATH` (`/usr/local/bin`): `tensorfold-native`, and `tensorfold` as a link to the same server, following
upstream's packaging, whose docs run `tensorfold serve`. The tools are also linked: `tf-cuda-test`, `tf-tp-test`,
`tf-tp-watch-check`, and `tensorfold-cuda`, which is the CUDA dev CLI (`run`, `digest`, `kl`, …) that a source install
names `bin/tensorfold`.

## How kernels and libraries are found

- Kernels: every CUDA fatbin (sm_120 SASS only) is embedded in the executables at build time. No kernel files are
  needed at run time, and none are JIT-compiled. `TENSORFOLD_CUDA_KERNELS` and `share/tensorfold/cuda/sm<cap>/`
  only apply to captured Triton sets for other families. GLM always runs on its own kernels, so leave the variable
  unset. A GPU other than `SM` has no image to load: rebuild with its SM.
- The executable's only link-time dependency is glibc. Everything else is loaded with `dlopen` when first used:
  - `libcuda.so.1` and `libnvidia-ml.so.1` (NVML, used for the P2P report) come from the host driver through the
    NVIDIA container runtime (`--gpus …`; the base image sets `NVIDIA_DRIVER_CAPABILITIES=compute,utility`).
  - `libcublasLt.so.13` is used by `--vision` and the cuBLASLt projections. It comes from cuBLAS 13.6 in the CUDA
    runtime image.
  - `libnccl.so.2` is needed by every rank of a GLM pair. The runtime image ships NCCL 2.30.7+cuda13.3. The engine
    requires at least 2.30.0.
  - `libtfgrammar.so`, `libturbojpeg.so.0` and `libwebpdecoder.so.3` are looked up in `<exe>/../lib`, which is
    `/opt/tensorfold/native/lib`. The symlink in `/usr/local/bin` resolves to that path.
  - The native engine uses no cuDNN.
- `tensorfold-native capabilities --json` lists each optional library with `found`. With a GPU it also shows `chip`
  (`nvidia-sm120`) and the NVML P2P pairs.

## Running it (what the scripts pass)

- GPUs: give each rank both GPUs (`--gpus '"device=0,1"'`) and pick its own with `--device N`. If `CUDA_VISIBLE_DEVICES`
  hides the peer GPU, NCCL cannot use P2P between the processes. Use `--ipc=host` (NCCL's shared memory and CUDA IPC
  between rank processes) and `--ulimit memlock=-1`. If the ranks run in separate containers, add `--network host` so
  `NCCL_COMM_ID` reaches 127.0.0.1.
- Ranks: `TF_GLM_RANK=<r> TF_GLM_WORLD=2 NCCL_COMM_ID=127.0.0.1:<port>`, and the same `serve` arguments on every rank.
  Rank 0 serves HTTP. Each rank also holds a TCP line to rank 0 on `<port>+1`.
- Memory: `TENSORFOLD_MEMORY_RESERVE_GIB` sets the floor (at least 2 GiB). `TENSORFOLD_CUDA_MEMORY_LIMIT_GB` caps
  the grant.
- No `LD_LIBRARY_PATH` or kernel path is needed. Leave `TENSORFOLD_CUDA_KERNELS` unset. Mount the model directory,
  or `HF_HOME`, and pass `--no-update-check` (or `TENSORFOLD_NO_UPDATE_CHECK=1`) to taste.
- Check: `tensorfold-native capabilities --json`, then `tf-cuda-test info|smoke|cublaslt|nccl|glm-exl3` (all small:
  under 1 GiB of GPU memory and under 20 s each).

## Build time and size (this host: Ryzen 9 9900X, 24 threads; other builds were running at the same time)

- Image build: 431 to 465 s over three builds from cached base layers. The `zig build install native grammar` step
  took 423 to 458 s of that.
  nvcc compiles about 100 sm_120 fatbins in about 1.5 min. The rest is Zig: `tf-cuda-test` takes about 4 to 5 min
  (6 GB RSS) and `tensorfold-native` about 2.5 min (4 GB), in parallel. With `TOOLS=0`, expect about 4 min.
- A first build also pulls the base images (devel 12 GB, runtime 4.4 GB). The codecs stage takes about 40 s once.
- Image size: 4.75 GB. That is `nvidia/cuda:13.3.1-runtime-ubuntu24.04` at 4.39 GB plus a 291 MB `/opt/tensorfold`
  layer: tensorfold-native 87 MB, tf-cuda-test 150 MB, dev CLI 38 MB, the libraries 5 MB.
