"""bus-fill v0: batched quant-weight GEMM for sm_70.

The point: weights for ALL batch rows stream from DRAM once per step; the
single memory trip computes B conversations. If aggregate tokens/sec scales
with B on Volta, the bus was idle — the kernel just needs to keep it busy.

Weight layout (Q4-quad, defined by this harness; v1 will eat llama.cpp's
native Q4_K blocks — same math, their 144-byte block format):
  Wq: [N, K/2] uint8 — per output row, packed nibbles; byte j covers inputs
      (2j, 2j+1), low nibble first; value = (nibble - 8) * scale (Q4_0-style)
  Ws: [N, K/32] float16 — one scale per 32-value sub-block

Triton 2.2's LLVM backend on Volta hard-crashes on tl.trans (and on
reshaping loaded values), so every operand is loaded already in its
consumption orientation via pointer arithmetic — the B operand arrives as
[K/2, BN] directly. Activations are pre-split even/odd on the host (free:
they're MBs, weights are the GBs), giving two pure dot(A_even,W_even) +
dot(A_odd,W_odd) halves with no value shuffles anywhere.

On Volta Triton lowers tl.dot through the m16n8k16 fp16 mma path and
software-pipelines plain loads (no cp.async) — exactly the machinery this
kernel exists to measure.
"""
import torch, time, json, sys
import triton, triton.language as tl
from triton.testing import do_bench


@triton.jit
def q4q_gemm(Ae, Ao, Wq, Ws, C, Bsz, K: tl.constexpr, N,
             BM: tl.constexpr, BN: tl.constexpr, BK2: tl.constexpr):
    # BK2 = bytes (= nibble-pairs) per K-chunk; K/2 total across the loop
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)

    rows = pid_m * BM + tl.arange(0, BM)
    rm = rows < Bsz
    outs = pid_n * BN + tl.arange(0, BN)

    jjc = tl.arange(0, BK2)[None, :]                         # along-K tile for A operands
    jjr = tl.arange(0, BK2)[:, None]                         # along-K tile for W operands (transposed)
    a_e_ptrs = Ae + rows[:, None] * (K // 2) + jjc           # [BM, BK2]
    a_o_ptrs = Ao + rows[:, None] * (K // 2) + jjc
    w_ptrs = Wq + outs[None, :] * (K // 2) + jjr             # [BK2, BN]
    s_ptrs = Ws + outs[None, :] * (K // 2) + jjr             # [BK2, BN]

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K // 2, BK2):
        b = tl.load(w_ptrs)                                                  # [BK2, BN] u8
        sc = tl.load(s_ptrs)                                                 # [BK2, BN] fp16
        we = ((b & 0xF).to(tl.float16) - 8.0) * sc
        wo = ((b >> 4).to(tl.float16) - 8.0) * sc
        acc += tl.dot(tl.load(a_e_ptrs), we)
        acc += tl.dot(tl.load(a_o_ptrs), wo)
        a_e_ptrs += BK2
        a_o_ptrs += BK2
        w_ptrs += BK2
        s_ptrs += BK2
    c_ptrs = C + rows[:, None] * N + outs[None, :]
    tl.store(c_ptrs, acc, mask=rm[:, None])


def quantize_q4q(Wf, device):
    """fp32 [N, K] -> Q4-quad layout, for the harness + reference."""
    N, K = Wf.shape
    w = Wf.to(device).reshape(N, K // 32, 32)
    amax = w.abs().amax(dim=2).clamp(min=1e-6)
    scale = (amax / 8.0)
    q = (w / scale[:, :, None] + 8.0).round().clamp(0, 15).to(torch.uint8)
    lo = q[:, :, 0::2]; hi = q[:, :, 1::2]
    bytes_ = (lo | (hi << 4)).reshape(N, K // 2).contiguous()
    s16 = scale.half().contiguous()
    ref = ((q.float() - 8.0) * s16.float()[:, :, None]).reshape(N, K)
    # Triton 2.2 on Volta wants every tile pointer affine: replicate each
    # sub-block scale across the 16 bytes that consume it.
    srep = s16.repeat_interleave(16, dim=1).contiguous()      # [N, K//2] fp16
    return bytes_, srep, ref


def launch(Ae, Ao, Wq, Ws, C, B, K, N):
    grid = (N // 128, 1)
    q4q_gemm[grid](Ae, Ao, Wq, Ws, C, B, K=K, N=N, BM=32, BN=128, BK2=64)


def main():
    dev = "cuda"
    K, N = 5120, 5120
    torch.manual_seed(0)
    Wf = torch.randn(N, K, device=dev) * 0.02
    Aall = torch.randn(32, K, device=dev, dtype=torch.float16)
    Ae = Aall[:, 0::2].contiguous()
    Ao = Aall[:, 1::2].contiguous()
    Wq, Ws, ref = quantize_q4q(Wf, dev)

    # correctness at B=4 — quantized-kernel output vs fp32 dequantized matmul
    C = torch.zeros(4, N, device=dev)
    launch(Ae[:4], Ao[:4], Wq, Ws, C, 4, K, N)
    referr = (C - (Aall[:4].float() @ ref.T)).abs().max().item()
    print(f"max abs err vs dequantized matmul: {referr:.4f}")
    if referr > 0.6:
        print("!! correctness gate failed — timings would be meaningless")
        sys.exit(1)

    results = {"max_abs_err": referr}
    for B in (1, 8, 32):
        C = torch.zeros(B, N, device=dev)
        ms = do_bench(lambda: launch(Ae[:B], Ao[:B], Wq, Ws, C, B, K, N), warmup=25, rep=100)
        results[B] = round(B / (ms / 1000.0), 1)
        bh = ref.half().contiguous()
        ms_cub = do_bench(lambda: torch.matmul(Aall[:B], bh.T), warmup=25, rep=100)
        results[f"cublas_fp16_{B}"] = round(B / (ms_cub / 1000.0), 1)
        print(f"B={B:2d}: bus-fill {results[B]:9.1f} t/s ({ms*1000:.1f} us)   cublas-fp16 {results[f'cublas_fp16_{B}']:9.1f} t/s ({ms_cub*1000:.1f} us)   ratio {results[B]/results[f'cublas_fp16_{B}']:.3f}")

    json.dump(results, open("busfill_v0.json", "w"), indent=1)


if __name__ == "__main__":
    main()
