// bus-fill v1 (MEASURED: levers do NOT stack — see RESULTS.md; kept as the
// combination experiment): v0 + three squad levers combined:
//   - v1b's depth-2 register-chain staging (two global loads in flight per
//     warp while mma chews the current block; depth-1 was free — ptxas
//     already pipelines one stage)
//   - v1c's wide shared reads (one LDS.U64 feeds 4 mmas)
//   - v1a's prmt/lop3 magic-bias dequant (8 ALU/FMA instrs per uint16 pair,
//     bitwise-identical results to v0's SIMT chain — verified: both halves of
//     the magic bias are exact in fp16, sc/16 is a power of two)
// v1d (strip/A-frag reuse) tested NEGATIVE — those loads were already
// latency-hidden — so it is not merged. Layout facts unchanged (probed:
// s = (lane/4) mod 4, B col slot = q + 4h, C scatter as below).
//
// Build (on the sm_70 box): /usr/bin/nvcc -O3 -arch=sm_70 busfill_v1.cu -o v1

#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

__device__ __forceinline__ void mma_m8n8k4(float* c, const uint32_t* a, const uint32_t* b) {
    asm volatile(
        "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "
        "{%0,%1,%2,%3,%4,%5,%6,%7};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]),
          "+f"(c[4]), "+f"(c[5]), "+f"(c[6]), "+f"(c[7])
        : "r"(a[0]), "r"(a[1]), "r"(b[0]), "r"(b[1]));
}

extern "C" __global__ void __launch_bounds__(128)
busfill_q4(const __half* __restrict__ A,      // [Bsz, K] row-major
           const uint8_t* __restrict__ Wq,    // [N, K/2] packed nibbles
           const __half* __restrict__ Ws,     // [N, K/32] fp16 scales
           float* __restrict__ C,             // [Bsz, N]
           int Bsz, int K, int N) {
    __shared__ __align__(16) uint8_t Wsh[2][4][8][16];  // double-buffered per-warp staging
    __shared__ __align__(4) __half Ssh[2][4][8];

    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    int q = lane & 3, h = lane >> 4, s = (lane >> 2) & 3;
    int trow = q + 4 * h;                                    // tile row in this warp's strip
    long wbase = (long)(blockIdx.x * 32 + warp * 8 + trow) * (K / 2);
    long sbase = (long)(blockIdx.x * 32 + warp * 8 + trow) * (K / 32);

    int arow = 8 * s + q + 4 * h;                            // batch row this lane feeds
    bool live = arow < Bsz;

    float acc[8] = {0,0,0,0,0,0,0,0};
    __align__(8) uint32_t aR[2], bR[2];
    const __half* aptr = A + (long)arow * K;                 // valid only when live
    int nb = K / 32;

    // ---- prologue: two register chains (A=even blocks, B=odd), both ahead ----
    uint4 wregA = make_uint4(0,0,0,0), wregB = make_uint4(0,0,0,0);
    __half sregA = __float2half(0.f), sregB = __float2half(0.f);
    if (lane < 8) {
        wregA = *(const uint4*)(Wq + wbase);            sregA = Ws[sbase];
        if (nb > 1) { wregB = *(const uint4*)(Wq + wbase + 16); sregB = Ws[sbase + 1]; }
    }

    const uint32_t z = 0;                              // zero source for PRMT
    const __half2 SUBC = __halves2half2(__float2half(1032.f), __float2half(1152.f)); // exact

    for (int blk = 0; blk < nb; blk++) {
        int cur = blk & 1;
        // STS the chain this half uses, then re-fill it with block blk+2 —
        // up to two global loads in flight while mma chews block blk.
        if (lane < 8) {
            if (cur == 0) {
                *(uint4*)&Wsh[0][warp][lane][0] = wregA; Ssh[0][warp][lane] = sregA;
                if (blk + 2 < nb) { wregA = *(const uint4*)(Wq + wbase + (blk + 2) * 16); sregA = Ws[sbase + blk + 2]; }
            } else {
                *(uint4*)&Wsh[1][warp][lane][0] = wregB; Ssh[1][warp][lane] = sregB;
                if (blk + 2 < nb) { wregB = *(const uint4*)(Wq + wbase + (blk + 2) * 16); sregB = Ws[sbase + blk + 2]; }
            }
        }
        __syncwarp();                                  // swap barrier
        float sc = __half2float(Ssh[cur][warp][trow]);
        __half2 SC2 = __halves2half2(__float2half(sc), __float2half(sc * 0.0625f)); // exact
        // two wide LDS reads feed 4 mmas each; prmt/lop3 dequant per uint16
        __align__(8) uint32_t W2[2][2];
        *(uint64_t*)W2[0] = *(const uint64_t*)&Wsh[cur][warp][trow][0];
        *(uint64_t*)W2[1] = *(const uint64_t*)&Wsh[cur][warp][trow][8];
        #pragma unroll
        for (int hf = 0; hf < 2; hf++) {
            #pragma unroll
            for (int j = 0; j < 4; j++) {              // four uint16 slots per wide reg -> 4 mmas
                uint32_t w = W2[hf][j >> 1];
                uint32_t p = (j & 1) ? (w >> 16) : (w & 0xFFFFu);
                uint32_t qa = __byte_perm(p, z, 0x4040);   // [p.b0,0,p.b0,0]
                uint32_t qb = __byte_perm(p, z, 0x4141);   // [p.b1,0,p.b1,0]
                uint32_t ma, mb;
                asm("lop3.b32 %0, %1, 0x00F0000F, 0x64006400, 0xEA;" : "=r"(ma) : "r"(qa));
                asm("lop3.b32 %0, %1, 0x00F0000F, 0x64006400, 0xEA;" : "=r"(mb) : "r"(qb));
                ((__half2*)bR)[0] = __hmul2(__hsub2(*(__half2*)&ma, SUBC), SC2);
                ((__half2*)bR)[1] = __hmul2(__hsub2(*(__half2*)&mb, SUBC), SC2);
                if (live) *(uint64_t*)aR = *(const uint64_t*)(aptr + blk * 32 + hf * 16 + j * 4);
                else      { aR[0] = aR[1] = 0; }
                mma_m8n8k4(acc, aR, bR);
            }
        }
    }

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int crow = 8 * s + (q & 1) + (i & 2) + 4 * h;
        if (crow < Bsz) {
            int ccol = (i & 4) + (q & 2) + (i & 1);
            C[(long)crow * N + blockIdx.x * 32 + warp * 8 + ccol] = acc[i];
        }
    }
}

// ---------------------- standalone harness (identical to v0) ----------------------
#include <cuda_runtime.h>

static float frand() { return (rand() / (float)RAND_MAX - 0.5f) * 2.f; }

int main(int argc, char** argv) {
    int K = 5120, N = 5120;
    srand(42);
    std::vector<float> fW((size_t)N * K);
    for (auto& v : fW) v = frand() * 0.4f;

    std::vector<uint8_t> qnib((size_t)N * K / 2);
    std::vector<__half> qsc((size_t)N * K / 32);
    std::vector<float> refW((size_t)N * K);
    for (long n = 0; n < N; n++) for (int blk = 0; blk < K / 32; blk++) {
        float amax = 1e-6f;
        for (int j = 0; j < 32; j++) amax = fmaxf(amax, fabsf(fW[n * K + blk * 32 + j]));
        float sc = amax / 8.f;
        qsc[n * K / 32 + blk] = __float2half(sc);
        for (int j = 0; j < 16; j++) {
            int lo = (int)lrintf(fW[n * K + blk * 32 + 2 * j] / sc) + 8;
            int hi = (int)lrintf(fW[n * K + blk * 32 + 2 * j + 1] / sc) + 8;
            lo = std::min(std::max(lo, 0), 15); hi = std::min(std::max(hi, 0), 15);
            qnib[n * K / 2 + blk * 16 + j] = lo | (hi << 4);
            refW[n * K + blk * 32 + 2 * j] = (lo - 8) * __half2float(qsc[n * K / 32 + blk]);
            refW[n * K + blk * 32 + 2 * j + 1] = (hi - 8) * __half2float(qsc[n * K / 32 + blk]);
        }
    }

    int Bs[3] = {1, 8, 32};
    std::vector<float> fA(32 * K);
    for (auto& v : fA) v = frand();

    uint8_t* dW; __half *dS, *dA; float* dC;
    cudaMalloc(&dW, (size_t)N * K / 2); cudaMalloc(&dS, (size_t)N * K / 2);
    cudaMalloc(&dA, 32 * K * 2); cudaMalloc(&dC, 32 * (size_t)N * 4);
    cudaMemcpy(dW, qnib.data(), (size_t)N * K / 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dS, qsc.data(), (size_t)N * K / 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dA, fA.data(), 32 * K * 2, cudaMemcpyHostToDevice);

    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    printf("shape K=N=%d, fp16-equiv cublas target would stream %zu MB/tok-stream\n", K, (size_t)N*K*2>>20);
    for (int bi = 0; bi < 3; bi++) {
        int B = Bs[bi];
        dim3 grid(N / 32);
        cudaMemset(dC, 0, 32 * (size_t)N * 4);
        busfill_q4<<<grid, 128>>>(dA, dW, dS, dC, B, K, N);
        cudaError_t le = cudaGetLastError();
        cudaDeviceSynchronize();
        cudaError_t se = cudaDeviceSynchronize();
        if (bi == 0) fprintf(stderr, "launch: %s / sync: %s\n", cudaGetErrorString(le), cudaGetErrorString(se));
        double maxerr = 0;
        std::vector<float> hC((size_t)B * N);
        cudaMemcpy(hC.data(), dC, (size_t)B * N * 4, cudaMemcpyDeviceToHost);
        for (int b = 0; b < B; b++) for (int n = 0; n < N; n += 97) {
            double s = 0; for (int k = 0; k < K; k++) s += (double)fA[b * K + k] * refW[(size_t)n * K + k];
            double e = fabs(s - hC[(size_t)b * N + n]);
            if (e > maxerr) maxerr = e;
        }
        cudaEventRecord(e0);
        for (int rep = 0; rep < 50; rep++) busfill_q4<<<grid, 128>>>(dA, dW, dS, dC, B, K, N);
        cudaEventRecord(e1); cudaEventSynchronize(e1);
        float ms; cudaEventElapsedTime(&ms, e0, e1); ms /= 50;
        double tput = B / (ms / 1000.0);
        double mbs = (double)(N * K / 2 + N * K / 2 / 8) / (ms / 1000.0) / 1e6;
        printf("B=%2d: %8.1f tok/s aggregated  (%.1f us/step, %.0f MB/s bus)  maxerr %.4f\n",
               B, tput, ms * 1000, mbs, maxerr);
    }
    return 0;
}
