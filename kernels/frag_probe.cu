// Layout deduction, component-wise. Assumed C layout (from PTX f32-accum
// table): slot i of lane l -> C[row][col], row = 8s + (l&1)+(i&2) + 4h,
// col = (i&4)+(l&2)+(i&1), s=l/4 q=l%4 h=l/16. Probe P1: unique A, B=1
// (reveals A frag permutation). P2: A=1, unique B (reveals B frag perm).
// Python harness inverts. Prints: lane,model_slot,got_value.
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdint>

__global__ void probe(const __half* A, const __half* B, float* C, int mode) {
    int l = threadIdx.x;
    int q = l & 3, h = l >> 4, s = l >> 2;
    uint32_t a[2], b[2];
    int arow = 8 * s + q + 4 * h;
    { __half2 t0 = __halves2half2(A[arow*4+0], A[arow*4+1]); a[0] = *(uint32_t*)&t0;
      __half2 t1 = __halves2half2(A[arow*4+2], A[arow*4+3]); a[1] = *(uint32_t*)&t1; }
    int bcol = q + 4 * h;
    { __half2 t0 = __halves2half2(B[0*8+bcol], B[1*8+bcol]); b[0] = *(uint32_t*)&t0;
      __half2 t1 = __halves2half2(B[2*8+bcol], B[3*8+bcol]); b[1] = *(uint32_t*)&t1; }
    float c[8] = {0,0,0,0,0,0,0,0};
    asm volatile(
        "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "
        "{%0,%1,%2,%3,%4,%5,%6,%7};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]),
          "+f"(c[4]), "+f"(c[5]), "+f"(c[6]), "+f"(c[7])
        : "r"(a[0]), "r"(a[1]), "r"(b[0]), "r"(b[1]));
    for (int i = 0; i < 8; i++) {
        int row = 8 * s + (l & 1) + (i & 2) + 4 * h;
        int col = (i & 4) + (l & 2) + (i & 1);
        C[row * 8 + col] = c[i];
    }
}
int main() {
    __half *dA, *dB; float *dC;
    cudaMallocManaged(&dA, 128 * 2); cudaMallocManaged(&dB, 32 * 2); cudaMallocManaged(&dC, 256 * 4);
    // P1: unique A (A[m][k] = (m*4+k+1)*64; B=1 -> C = sum of gathered A slots)
    for (int i = 0; i < 128; i++) dA[i] = __float2half((float)((i + 1) * 64));
    for (int i = 0; i < 32; i++) dB[i] = __float2half(1.f);
    probe<<<1, 32>>>(dA, dB, dC, 1); cudaDeviceSynchronize();
    printf("P1\n"); for (int i = 0; i < 256; i++) printf("%d,%.0f\n", i, dC[i]);
    // P2: A=1, unique B (B[k][n] = (k*8+n+1) -> C = sum of gathered B slots)
    for (int i = 0; i < 128; i++) dA[i] = __float2half(1.f);
    for (int i = 0; i < 32; i++) dB[i] = __float2half((float)(i + 1));
    probe<<<1, 32>>>(dA, dB, dC, 2); cudaDeviceSynchronize();
    printf("P2\n"); for (int i = 0; i < 256; i++) printf("%d,%.0f\n", i, dC[i]);
    return 0;
}
