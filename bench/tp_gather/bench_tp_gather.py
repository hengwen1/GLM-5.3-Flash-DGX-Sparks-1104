#!/usr/bin/env python3
"""TP=2 all-gather microbenchmark: 4-rail TCP staged gather vs ncclAllGather.

Standalone; no engine code. Run once per rank (rank 0 first, it listens).

The TCP path is a faithful clone of the Thor EXL3 backend's
`deepinfer/backends/thor_exl3/prefill_transport.py` (MultiChannelMapped):
  * one pinned, cudaHostRegister'ed staging buffer of 2*n floats,
  * its device alias by cudaHostGetDevicePointer (zero-copy on UMA),
  * `--conns` TCP connections striped over the rails (`i % rail_count`),
  * per-stripe (seq, len) header then raw bytes, send and receive concurrently.

Sizes are per-rank bytes; GLM-5.3 hidden size is 4096 fp32 = 16 KiB a row, so
`bytes = rows * 16384`:
    512 KiB = 32 rows    (p2p_gather cap)
      4 MiB = 256 rows   (ROCE_MAX_KB)
     64 MiB = 4096 rows  (one prompt chunk, TF_GLM_PREFILL_REDUCE=gather)

Nothing here changes the serving container: this runs its own short-lived
container on each node.
"""
import argparse
import ctypes
import glob
import json
import math
import os
import socket
import struct
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.distributed as dist

HEADER = struct.Struct('!QQ')

DEFAULT_SIZES = [
    512 << 10,   # 32 rows
    1 << 20,
    2 << 20,     # 128 rows
    4 << 20,     # 256 rows
    8 << 20,     # 512 rows
    16 << 20,    # 1024 rows
    32 << 20,    # 2048 rows
    64 << 20,    # 4096 rows: one prompt chunk, gather mode
]


# ----------------------------------------------------------------------------- cudart
def load_cudart():
    candidates = sorted(glob.glob('/usr/local/lib/python3*/dist-packages/nvidia/cu*/lib/libcudart.so*'))
    candidates += sorted(glob.glob(os.path.join(os.path.dirname(torch.__file__), 'lib', 'libcudart*')))
    for path in candidates:
        try:
            return ctypes.CDLL(path), path
        except OSError:
            pass
    raise SystemExit('libcudart not found (looked in nvidia/cu*/lib and torch/lib)')


def read_into(sock, buf):
    offset = 0
    while offset < len(buf):
        n = sock.recv_into(buf[offset:])
        if n == 0:
            raise ConnectionError('peer closed the connection')
        offset += n


# ----------------------------------------------------------------------------- TCP rail
class MultiChannelMapped:
    """Same protocol as the source transport, minus profiling and the mailbox."""

    def __init__(self, rank, local, peer, conns, port, cudart):
        self.rank, self.cudart = rank, cudart
        self.local, self.peer = local, peer
        self.conns = conns
        self.seq = 0
        self.host = None
        self.registered = None
        self.interface = None
        self.cuda_view = None
        self.sockets = []
        self.stage_us = 0.0
        self.net_us = 0.0
        self.pool = ThreadPoolExecutor(conns * 2, thread_name_prefix='rail')
        self._connect(port)

    # -- buffers ------------------------------------------------------------
    def allocate(self, n):
        if self.host is not None and self.host.numel() >= 2 * n:
            return
        self._release()
        size = ((2 * n * 4 + 65535) // 65536) * 65536
        owner = torch.empty(size + 65536, dtype=torch.uint8)
        offset = (-owner.data_ptr()) % 65536
        self.owner = owner
        self.host = owner[offset:offset + size].view(torch.float32)
        result = self.cudart.cudaHostRegister(ctypes.c_void_p(self.host.data_ptr()),
                                              ctypes.c_size_t(size), ctypes.c_uint(0))
        if result != 0:
            raise SystemExit(f'cudaHostRegister: {result}')
        self.registered = self.host.data_ptr()

        pointer = ctypes.c_void_p()
        status = self.cudart.cudaHostGetDevicePointer(ctypes.byref(pointer),
                                                      ctypes.c_void_p(self.host.data_ptr()),
                                                      ctypes.c_uint(0))
        if status != 0:
            raise SystemExit(f'cudaHostGetDevicePointer: {status}')

        class Interface:
            pass
        obj = Interface()
        obj.owner = self.host
        obj.__cuda_array_interface__ = {
            'shape': tuple(self.host.shape), 'strides': None, 'typestr': '<f4',
            'data': (pointer.value, False), 'version': 3,
        }
        self.interface = obj
        self.cuda_view = torch.as_tensor(obj, device='cuda')
        if self.cuda_view.data_ptr() != pointer.value:
            raise SystemExit('mapped view and cudaHostGetDevicePointer disagree')

    def _release(self):
        if self.registered is not None:
            torch.cuda.synchronize()
            self.cudart.cudaHostUnregister(ctypes.c_void_p(self.registered))
            self.registered = None
        self.host = None
        self.interface = None
        self.cuda_view = None

    # -- sockets ------------------------------------------------------------
    def _connect(self, port):
        for i in range(self.conns):
            rail = i % len(self.local)
            address = (self.local[rail] if self.rank == 0 else self.peer[rail], port + i)
            if self.rank == 0:
                listener = socket.socket()
                listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                listener.settimeout(120)
                listener.bind(address)
                listener.listen(1)
                sock, accepted = listener.accept()
                listener.close()
                if accepted[0] != self.peer[rail]:
                    raise SystemExit(f'connection on {address} from {accepted[0]}, wanted {self.peer[rail]}')
            else:
                end = time.monotonic() + 120
                while True:
                    sock = socket.socket()
                    sock.settimeout(5)
                    sock.bind((self.local[rail], 0))
                    try:
                        sock.connect(address)
                        break
                    except OSError:
                        sock.close()
                        if time.monotonic() > end:
                            raise
                        time.sleep(0.1)
            sock.settimeout(120)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self.sockets.append(sock)

    # -- the collective -----------------------------------------------------
    def gather(self, source):
        n = source.numel()
        self.allocate(n)
        host = self.host[:2 * n].view(2, n)
        t_stage = time.perf_counter()
        if source.data_ptr() == self.cuda_view.data_ptr() + self.rank * n * 4:
            torch.cuda.current_stream().synchronize()
        else:
            host[self.rank].copy_(source.view(-1), non_blocking=False)
        self.stage_us += (time.perf_counter() - t_stage) * 1e6
        source_bytes = memoryview(host[self.rank].numpy()).cast('B')
        target_bytes = memoryview(host[1 - self.rank].numpy()).cast('B')
        self.seq += 1
        sequence = self.seq
        conns = len(self.sockets)

        def send(i, lo, hi):
            s = self.sockets[i]
            s.sendall(HEADER.pack(sequence, hi - lo))
            s.sendall(source_bytes[lo:hi])

        def recv(i, lo, hi):
            s = self.sockets[i]
            header = bytearray(HEADER.size)
            read_into(s, memoryview(header))
            if HEADER.unpack(header) != (sequence, hi - lo):
                raise SystemExit('sequence/size mismatch on the wire')
            read_into(s, target_bytes[lo:hi])

        t_net = time.perf_counter()
        futures = []
        for stripe, i in enumerate(range(conns)):
            lo = 4 * (n * stripe // conns)
            hi = 4 * (n * (stripe + 1) // conns)
            futures += [self.pool.submit(recv, i, lo, hi), self.pool.submit(send, i, lo, hi)]
        for f in futures:
            f.result()
        self.net_us += (time.perf_counter() - t_net) * 1e6
        return self.cuda_view[:2 * n].view(2, n)

    def close(self):
        self._release()
        for s in self.sockets:
            s.close()
        self.pool.shutdown(wait=False)


# ----------------------------------------------------------------------------- helpers
def rail_counters(devs):
    out = {}
    for dev in devs:
        base = f'/sys/class/net/{dev}/statistics'
        try:
            out[dev] = int(open(os.path.join(base, 'rx_bytes')).read())
        except OSError:
            out[dev] = None
    return out


def counter_delta(before, after):
    # rx only: this driver's tx_bytes is not a usable 64-bit counter.
    out = {}
    for dev, a in after.items():
        b = before.get(dev)
        out[dev] = None if (a is None or b is None) else {'rx_bytes': a - b}
    return out


def stats(samples_us):
    s = sorted(samples_us)
    n = len(s)
    mean = sum(s) / n
    return {
        'mean_us': mean,
        'median_us': s[n // 2],
        'p10_us': s[max(0, int(0.10 * n) - 1)],
        'p90_us': s[min(n - 1, int(math.ceil(0.90 * n)) - 1)],
        'min_us': s[0],
        'max_us': s[-1],
    }


def check(name, got, expected):
    if not torch.equal(got, expected):
        bad = (got != expected).sum().item()
        raise SystemExit(f'{name}: {bad} of {got.numel()} values differ')
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rank', type=int, required=True)
    ap.add_argument('--world', type=int, default=2)
    ap.add_argument('--master', default='192.168.100.38')
    ap.add_argument('--port', type=int, default=29600)
    ap.add_argument('--sizes', default=','.join(str(s) for s in DEFAULT_SIZES),
                    help='per-rank bytes, comma separated')
    ap.add_argument('--iters', type=int, default=20)
    ap.add_argument('--warmup', type=int, default=5)
    ap.add_argument('--conns', type=int, default=8)
    ap.add_argument('--rails', default='192.168.100.38,192.168.101.38,192.168.102.38,192.168.103.38',
                    help='this node\'s four rail addresses')
    ap.add_argument('--peer-rails', default='192.168.100.39,192.168.101.39,192.168.102.39,192.168.103.39')
    ap.add_argument('--mode', choices=('both', 'nccl', 'tcp'), default='both')
    ap.add_argument('--json', default='')
    ap.add_argument('--label', default='')
    args = ap.parse_args()

    sizes = [int(x) for x in args.sizes.split(',') if x]
    local = [x for x in args.rails.split(',') if x]
    peer = [x for x in args.peer_rails.split(',') if x]
    if len(local) != len(peer) or not local:
        raise SystemExit('--rails and --peer-rails need the same, non-zero number of addresses')
    devs = [f'mgbe{i}_0' for i in range(len(local))]
    cudart, cudart_path = load_cudart()

    os.environ.setdefault('MASTER_ADDR', args.master)
    os.environ.setdefault('MASTER_PORT', str(args.port))
    os.environ.setdefault('RANK', str(args.rank))
    os.environ.setdefault('WORLD_SIZE', str(args.world))
    os.environ.setdefault('NCCL_SOCKET_IFNAME', ','.join(devs))
    os.environ.setdefault('NCCL_IB_DISABLE', '1')
    os.environ.setdefault('NCCL_DEBUG', 'WARN')
    torch.cuda.set_device(0)

    report = {
        'label': args.label,
        'rank': args.rank,
        'world': args.world,
        'torch': torch.__version__,
        'cuda': torch.version.cuda,
        'nccl': ''.join(str(x) for x in torch.cuda.nccl.version()),
        'gpu': torch.cuda.get_device_name(0),
        'cudart': cudart_path,
        'nccl_socket_ifname': os.environ['NCCL_SOCKET_IFNAME'],
        'nccl_socket_nthreads': os.environ.get('NCCL_SOCKET_NTHREADS'),
        'nccl_nsocks_perthread': os.environ.get('NCCL_NSOCKS_PERTHREAD'),
        'nccl_channels': (os.environ.get('NCCL_MIN_NCHANNELS'), os.environ.get('NCCL_MAX_NCHANNELS')),
        'conns': args.conns,
        'rails': local,
        'rows': {s // 16384: s for s in sizes},
        'results': [],
    }

    try:
        dist.init_process_group('nccl', device_id=torch.device('cuda', 0))
    except TypeError:  # older torch: no device_id
        dist.init_process_group('nccl')
    transport = None
    try:
        max_bytes = max(sizes)
        max_n = max_bytes // 4
        device_in = torch.empty(max_n, dtype=torch.float32, device='cuda')
        device_out = torch.empty(2 * max_n, dtype=torch.float32, device='cuda')

        if args.mode in ('both', 'tcp'):
            transport = MultiChannelMapped(args.rank, local, peer, args.conns, args.port + 100, cudart)
            transport.allocate(max_n)
        dist.barrier()
        report['ready'] = True

        for n_bytes in sizes:
            n = n_bytes // 4
            rows = n_bytes // 16384
            base = torch.arange(n, dtype=torch.float32, device='cuda') + 1_000_000.0 * args.rank
            device_in[:n].copy_(base)
            expected = torch.stack([torch.arange(n, dtype=torch.float32, device='cuda'),
                                    torch.arange(n, dtype=torch.float32, device='cuda') + 1_000_000.0])

            entry = {'bytes_per_rank': n_bytes, 'rows': rows, 'paths': {}}

            if args.mode in ('both', 'nccl'):
                for _ in range(args.warmup):
                    dist.all_gather_into_tensor(device_out[:2 * n], device_in[:n])
                torch.cuda.synchronize()
                ok = check('nccl', device_out[:2 * n].view(2, n), expected)
                per_call = []
                for _ in range(args.iters):
                    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
                    torch.cuda.synchronize()
                    start.record()
                    dist.all_gather_into_tensor(device_out[:2 * n], device_in[:n])
                    end.record()
                    torch.cuda.synchronize()
                    per_call.append(start.elapsed_time(end) * 1e3)
                chained_start, chained_end = torch.cuda.Event(True), torch.cuda.Event(True)
                torch.cuda.synchronize()
                chained_start.record()
                for _ in range(args.iters):
                    dist.all_gather_into_tensor(device_out[:2 * n], device_in[:n])
                chained_end.record()
                torch.cuda.synchronize()
                chained = chained_start.elapsed_time(chained_end) * 1e3 / args.iters
                entry['paths']['nccl'] = {'bitwise': ok, **stats(per_call), 'chained_us': chained,
                                          'wire_GBps': 2 * n_bytes / chained / 1e3}
                dist.barrier()

            if args.mode in ('both', 'tcp'):
                for _ in range(args.warmup):
                    transport.gather(device_in[:n])
                torch.cuda.synchronize()
                got = transport.gather(device_in[:n]).clone()
                ok = check('tcp', got, expected)
                before = rail_counters(devs)
                transport.stage_us = transport.net_us = 0.0
                per_call = []
                for _ in range(args.iters):
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    transport.gather(device_in[:n])
                    torch.cuda.synchronize()
                    per_call.append((time.perf_counter() - t0) * 1e6)
                after = rail_counters(devs)
                st = stats(per_call)
                entry['paths']['tcp'] = {'bitwise': ok, **st,
                                         'wire_GBps': 2 * n_bytes / st['median_us'] / 1e3,
                                         'stage_us': transport.stage_us / args.iters,
                                         'net_us': transport.net_us / args.iters,
                                         'rails': counter_delta(before, after)}
                dist.barrier()

            report['results'].append(entry)
            if args.rank == 0:
                cells = [f'{rows:>5} rows', f'{n_bytes >> 10:>7} KiB']
                for path in ('nccl', 'tcp'):
                    p = entry['paths'].get(path)
                    if not p:
                        cells.append(f'{path}: -')
                    elif path == 'tcp':
                        cells.append(f'tcp: {p["median_us"]:>9.1f} us {p["wire_GBps"]:>6.2f} GB/s '
                                     f'(stage {p["stage_us"]:>6.1f} net {p["net_us"]:>7.1f})')
                    else:
                        cells.append(f'nccl: {p["median_us"]:>8.1f} us {p["wire_GBps"]:>6.2f} GB/s')
                print('  '.join(cells), flush=True)
    finally:
        if transport is not None:
            transport.close()
        dist.barrier()
        dist.destroy_process_group()
        report['done'] = True
        if args.rank == 0 or args.json:
            text = json.dumps(report)
            if args.json:
                os.makedirs(os.path.dirname(args.json) or '.', exist_ok=True)
                with open(args.json, 'w') as f:
                    f.write(text + '\n')
            print('JSON ' + text, flush=True)


if __name__ == '__main__':
    sys.exit(main())
