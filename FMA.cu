#include <stdio.h>
#include <cuda_runtime.h>

// FMA.f32 Microbenchmark - Single-precision floating point FMA
__global__ void FMA_F32_Kernel(int iterations) {
    float result = 0.0f;
    float a = 1.23f, b = 2.34f, c = 3.45f;
    
    for (int i = 0; i < iterations; ++i) {
        asm volatile(
            "fma.rn.f32 %0, %1, %2, %3;\n\t"
            : "=f"(result) 
            : "f"(a), "f"(b), "f"(c)
        );
        // To ensure the result is used and prevent dead code elimination
        if (i == iterations-1) {
            // Force a global memory write using atomics to prevent optimization
            float *ptr = (float*)0x1; // This address is never used
            asm("// Prevent optimization\n");
        }
    }
}

// MAD.lo.u32 Microbenchmark - 32-bit integer multiply-add
// TODO: Perform a few iterations of the kernel itself and take that as a trace
// TODO: Try changing the grid and the block size 
__global__ void MAD_U32_Kernel(int iterations) {
    unsigned int r1 = 123;
    unsigned int r2 = 234;
    unsigned int r3 = 345;
    unsigned int r4;
    
    for (int i = 0; i < iterations; ++i) {
        asm volatile("mad.lo.u32 %0, %1, %2, %3;" : "=r"(r4) : "r"(r1), "r"(r2), "r"(r3));

        // To ensure the result is used and prevent dead code elimination
        if (i == iterations-1) {
            // Force a global memory write using atomics to prevent optimization
            float *ptr = (float*)0x1; // This address is never used
            asm("// Prevent optimization\n");
        }
    }
}

// FMA.f64 Microbenchmark - Double-precision floating point FMA
__global__ void FMA_F64_Kernel(int iterations) {
    double d1 = 1.23;
    double d2 = 2.34;
    double d3 = 3.45;
    double d4;
    
    for (int i = 0; i < iterations; ++i) {
        asm volatile("fma.rn.f64 %0, %1, %2, %3;" : "=d"(d4) : "d"(d1), "d"(d2), "d"(d3));
        // To ensure the result is used and prevent dead code elimination
        if (i == iterations-1) {
            // Force a global memory write using atomics to prevent optimization
            float *ptr = (float*)0x1; // This address is never used
            asm("// Prevent optimization\n");
        }
    }
}

// FMA.f16x2 Microbenchmark - Half-precision vector FMA (using int for packing)
__global__ void FMA_F16x2_Kernel(int iterations) {
    // Packed half2 values as integers (each int contains two FP16 values)
    unsigned int f1 = 0x3C003C00; // Two packed FP16 values (1.0, 1.0)
    unsigned int f2 = 0x3C003C00; // Two packed FP16 values (1.0, 1.0)
    unsigned int f3 = 0x3C003C00; // Two packed FP16 values (1.0, 1.0)
    unsigned int f4;
    
    for (int i = 0; i < iterations; ++i) {
        asm volatile("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(f4) : "r"(f1), "r"(f2), "r"(f3));
        // To ensure the result is used and prevent dead code elimination
        if (i == iterations-1) {
            // Force a global memory write using atomics to prevent optimization
            float *ptr = (float*)0x1; // This address is never used
            asm("// Prevent optimization\n");
        }
    }
}

// MAD.cc.s32 Microbenchmark - 32-bit integer multiply-add with carry
__global__ void MAD_CC_S32_Kernel(int iterations) {
    int r1 = 123;
    int r2 = 234;
    int r3 = 345;
    int r4, r6;
    int r5 = 456;
    
    for (int i = 0; i < iterations; ++i) {
        asm volatile("mad.lo.cc.s32 %0, %1, %2, %3;\n\t"
                     "addc.s32 %4, %5, 0;" : 
                     "=r"(r4), "=r"(r6) : 
                     "r"(r1), "r"(r2), "r"(r3), "r"(r5));
        // To ensure the result is used and prevent dead code elimination
        if (i == iterations-1) {
            // Force a global memory write using atomics to prevent optimization
            float *ptr = (float*)0x1; // This address is never used
            asm("// Prevent optimization\n");
        }
    }
}

int main() {
    int iterations = 10000000;
    int blockSize = 1024;
    int gridSize = 640;
    
    // printf("Running FMA.f32 microbenchmark...\n");
    // FMA_F32_Kernel<<<gridSize, blockSize>>>(iterations);
    // cudaDeviceSynchronize();
    
    printf("Running MAD.lo.u32 microbenchmark...\n");
    MAD_U32_Kernel<<<gridSize, blockSize>>>(iterations);
    cudaDeviceSynchronize();
    
    // printf("Running FMA.f64 microbenchmark...\n");
    // FMA_F64_Kernel<<<gridSize, blockSize>>>(iterations);
    // cudaDeviceSynchronize();
    
    // printf("Running FMA.f16x2 microbenchmark...\n");
    // FMA_F16x2_Kernel<<<gridSize, blockSize>>>(iterations);
    // cudaDeviceSynchronize();
    
    // printf("Running MAD.cc.s32 microbenchmark...\n");
    // MAD_CC_S32_Kernel<<<gridSize, blockSize>>>(iterations);
    // cudaDeviceSynchronize();
    
    printf("All microbenchmarks completed.\n");
    
    return 0;
}