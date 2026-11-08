// bus-fill v1c = v0 + ONE lever: batched (wider) shared loads.
// v0 did 8 LDS.U16 per lane per Q4_0 block (one uint16 = 4 nibbles = one mma).
// The per-lane slots for consecutive mmas are contiguous bytes of the lane's 16B
// block, so here ONE LDS.U64 covers 4 mmas (2 per block, 8 bytes each). Nibble
// extraction on the wide register is plain shifts; tiling, global load structure,
// dequant math, and the layout facts are untouched.
//
// Build: /usr/bin/nvcc -O3 -arch=sm_70 busfill_v1c.cu -o v1c

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
    __shared__ uint8_t Wsh[4][8][16];         // per-warp staging: 8 outputs x one Q4_0 block
    __shared__ __half  Ssh[4][8];

    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    int q = lane & 3, h = lane >> 4, s = (lane >> 2) & 3;
    int trow = q + 4 * h;                              // tile row (this warp's 8-output strip)
    long wrow = (long)(blockIdx.x * 32 + warp * 8 + trow);   // global W row

    int arow = 8 * s + q + 4 * h;                      // batch row this lane feeds
    bool live = arow < Bsz;

    float acc[8] = {0,0,0,0,0,0,0,0};
    __align__(8) uint32_t aR[2], bR[2];
    const __half* aptr = A + (long)arow * K;           // valid only when live

    // ---- prefetch first block, then roll ----
    uint4 wreg = make_uint4(0,0,0,0); __half sreg = __float2half(0.f);
    if (lane < 8) { wreg = *(const uint4*)(Wq + wrow * (K / 2)); sreg = Ws[wrow * (K / 32)]; }

    for (int blk = 0; blk < K / 32; blk++) {
        uint4 wcur = wreg; __half scur = sreg;
        if (blk + 1 < K / 32 && lane < 8) {
            wreg = *(const uint4*)(Wq + wrow * (K / 2) + (blk + 1) * 16);
            sreg = Ws[wrow * (K / 32) + blk + 1];
        }
        if (lane < 8) { *(uint4*)&Wsh[warp][lane][0] = wcur; Ssh[warp][lane] = scur; }
        __syncwarp();
        float sc = __half2float(Ssh[warp][trow]);
        // v1c lever: two wide LDS reads feed 4 mmas each (was 8x LDS.U16)
        __align__(8) uint32_t W2[2][2];
        *(uint64_t*)W2[0] = *(const uint64_t*)&Wsh[warp][trow][0];
        *(uint64_t*)W2[1] = *(const uint64_t*)&Wsh[warp][trow][8];
        #pragma unroll
        for (int half = 0; half < 2; half++) {         // one LDS.U64's worth of bytes = 4 mmas
            #pragma unroll
            for (int j = 0; j < 4; j++) {              // uint16 slot within the wide register -> 4 inputs
                uint32_t w = W2[half][j >> 1];
                uint32_t p = (j & 1) ? (w >> 16) : (w & 0xFFFFu);
                ((__half*)bR)[0] = __float2half((float)(((p      ) & 0xF) - 8) * sc);
                ((__half*)bR)[1] = __float2half((float)(((p >>  4) & 0xF) - 8) * sc);
                ((__half*)bR)[2] = __float2half((float)(((p >>  8) & 0xF) - 8) * sc);
                ((__half*)bR)[3] = __float2half((float)(((p >> 12) & 0xF) - 8) * sc);
                if (live) *(uint64_t*)aR = *(const uint64_t*)(aptr + blk * 32 + j * 4 + half * 16);
                else      { aR[0] = aR[1] = 0; }
                mma_m8n8k4(acc, aR, bR);
            }
        }
        __syncwarp();
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

// ---------------------- standalone harness ----------------------
#include <cuda_runtime.h>

static float frand() { return (rand() / (float)RAND_MAX - 0.5f) * 2.f; }

int main(int argc, char** argv) {
    int K = 5120, N = 5120;
    srand(42);
    std::vector<float> fW((size_t)N * K);
    for (auto& v : fW) v = frand() * 0.4f;

    // quantize to Q4_0 planes + keep fp16-dequantized reference
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
        // warmup + verify (first block only would suffice; run full, it's cheap)
        cudaMemset(dC, 0, 32 * (size_t)N * 4);
        busfill_q4<<<grid, 128>>>(dA, dW, dS, dC, B, K, N);
        cudaError_t le = cudaGetLastError();
        cudaDeviceSynchronize();
        cudaError_t se = cudaDeviceSynchronize();
        if (bi == 0) fprintf(stderr, "launch: %s / sync: %s\n", cudaGetErrorString(le), cudaGetErrorString(se));
        // CPU verify rows x few cols
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
        double mbs = (double)(N * K / 2 + N * K / 2 / 8) / (ms / 1000.0) / 1e6;  // nibble+scaled bytes streamed
        printf("B=%2d: %8.1f tok/s aggregated  (%.1f us/step, %.0f MB/s bus)  maxerr %.4f\n",
               B, tput, ms * 1000, mbs, maxerr);
    }
    return 0;
}
