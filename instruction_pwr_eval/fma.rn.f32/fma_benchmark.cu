#include <stdio.h>
#include <cuda_runtime.h>

// Kernel function to execute fma.rn.f32 repeatedly
__global__ void fmaRnF32Kernel(int iterations) {
    // Initialize multiple registers with different values
    __asm volatile (
        ".reg .f32 %fa<8>, %fb<8>, %fc<8>, %fd<8>;\n\t"
        
        // Initialize a set of registers with different values
        "mov.f32 %fa0, 1.23;\n\t"
        "mov.f32 %fa1, 2.34;\n\t"
        "mov.f32 %fa2, 3.45;\n\t"
        "mov.f32 %fa3, 4.56;\n\t"
        "mov.f32 %fa4, 5.67;\n\t"
        "mov.f32 %fa5, 6.78;\n\t"
        "mov.f32 %fa6, 7.89;\n\t"
        "mov.f32 %fa7, 8.90;\n\t"
        
        "mov.f32 %fb0, 0.01;\n\t"
        "mov.f32 %fb1, 0.12;\n\t"
        "mov.f32 %fb2, 0.23;\n\t"
        "mov.f32 %fb3, 0.34;\n\t"
        "mov.f32 %fb4, 0.45;\n\t"
        "mov.f32 %fb5, 0.56;\n\t"
        "mov.f32 %fb6, 0.67;\n\t"
        "mov.f32 %fb7, 0.78;\n\t"
        
        "mov.f32 %fc0, 0.11;\n\t"
        "mov.f32 %fc1, 0.22;\n\t"
        "mov.f32 %fc2, 0.33;\n\t"
        "mov.f32 %fc3, 0.44;\n\t"
        "mov.f32 %fc4, 0.55;\n\t"
        "mov.f32 %fc5, 0.66;\n\t"
        "mov.f32 %fc6, 0.77;\n\t"
        "mov.f32 %fc7, 0.88;\n\t"
    );

    // Execute independent fma.rn.f32 operations repeatedly
    for (int i = 0; i < iterations; ++i) {
        __asm volatile (
            // Perform independent FMAs - no data dependencies between operations
            "fma.rn.f32 %fd0, %fa0, %fb0, %fc0;\n\t"
            "fma.rn.f32 %fd1, %fa1, %fb1, %fc1;\n\t"
            "fma.rn.f32 %fd2, %fa2, %fb2, %fc2;\n\t"
            "fma.rn.f32 %fd3, %fa3, %fb3, %fc3;\n\t"
            "fma.rn.f32 %fd4, %fa4, %fb4, %fc4;\n\t"
            "fma.rn.f32 %fd5, %fa5, %fb5, %fc5;\n\t"
            "fma.rn.f32 %fd6, %fa6, %fb6, %fc6;\n\t"
            "fma.rn.f32 %fd7, %fa7, %fb7, %fc7;\n\t"
        );
    }
}

// Function to measure kernel execution time
float measureKernelTime(int grid_dim, int block_dim, int iterations) {
    dim3 grid(grid_dim);
    dim3 block(block_dim);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    fmaRnF32Kernel<<<grid, block>>>(iterations);
    cudaEventRecord(stop);
    
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(error));
    }
    
    return milliseconds;
}

int main(int argc, char *argv[]) {
    // Default values
    int grid_dim = 640;
    int block_dim = 1024;
    int iterations = 1000000;
    
    // Parse command line arguments if provided
    if (argc > 1) grid_dim = atoi(argv[1]);
    if (argc > 2) block_dim = atoi(argv[2]);
    if (argc > 3) iterations = atoi(argv[3]);
    
    // Ensure valid dimensions
    if (grid_dim <= 0) grid_dim = 640;
    if (block_dim <= 0 || block_dim > 1024) block_dim = 1024;
    if (iterations <= 0) iterations = 1000000;
    
    printf("Running fma.rn.f32 micro-benchmark\n");
    printf("Grid Dimension: %d\n", grid_dim);
    printf("Block Dimension: %d\n", block_dim);
    printf("Iterations per thread: %d\n", iterations);
    printf("Total FMA operations: %llu\n", 
           (unsigned long long)grid_dim * block_dim * iterations * 8);
    
    // Run the kernel and measure execution time
    float elapsed_time = measureKernelTime(grid_dim, block_dim, iterations);
    
    printf("Kernel execution time: %.3f ms\n", elapsed_time);
    
    return 0;
}