#include <iostream>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <cuda_runtime.h>

// Helper macro for CUDA error checking.
#define cudaCheckError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true){
    if (code != cudaSuccess) {
        std::cerr << "GPUassert: " << cudaGetErrorString(code)
                  << " " << file << " " << line << std::endl;
        if (abort) exit(code);
    }
}

// --------------------------
// Serial (CPU) version
// --------------------------
// This version follows the original Rcpp logic:
// 1. For each feature (column), count the number of zeros.
// 2. For each unique pair (i,j) with i > j, compute result[counter] = zeroes[i] + zeroes[j].
void serialCtz(const float* X, int nsubjs, int nfeats, int* result) {
    int llt = nfeats * (nfeats - 1) / 2;

    // Count zero frequency per feature.
    int* zeroes = new int[nfeats];
    for (int i = 0; i < nfeats; i++) {
        zeroes[i] = 0;
        for (int j = 0; j < nsubjs; j++) {
            if (X[j * nfeats + i] == 0.0f)
                zeroes[i]++;
        }
    }
    // Count joint zero frequency for each feature pair.
    int counter = 0;
    for (int i = 1; i < nfeats; i++) {
        for (int j = 0; j < i; j++) {
            result[counter] = zeroes[i] + zeroes[j];
            counter++;
        }
    }
    delete[] zeroes;
}

// --------------------------
// GPU version
// --------------------------
__global__ void jointZeroCountKernel(const float* __restrict__ X, int nsubjs, int nfeats, int* result) {
    // Total number of pairs = nfeats*(nfeats-1)/2.
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalPairs = (nfeats * (nfeats - 1)) / 2;
    if (idx >= totalPairs) return;    
    float temp = sqrtf(8.0f * idx + 1.0f);
    int i = static_cast<int>(floorf((temp + 1.0f) * 0.5f));
    int j = idx - (i * (i - 1)) / 2;
    
    int sum = 0;
    for (int s = 0; s < nsubjs; s++) {
        // Use __ldg to load values from read-only global memory.
        float val_i = __ldg(&X[s * nfeats + i]);
        float val_j = __ldg(&X[s * nfeats + j]);
        sum += (int)(val_i == 0.0f) + (int)(val_j == 0.0f);
    }
    result[idx] = sum;
}

int main() {
    // Matrix dimensions.
    int nsubjs = 100000;  // Number of subjects (rows)
    int nfeats = 80;   // Number of features (columns)
    int totalPairs = nfeats * (nfeats - 1) / 2;

    // Allocate and initialize matrix X on the host (row-major order).
    float* h_X = new float[nsubjs * nfeats];
    float p = 0.1f;  // Probability that an entry is zero.
    for (int i = 0; i < nsubjs * nfeats; i++) {
        float r = static_cast<float>(rand()) / RAND_MAX;
        h_X[i] = (r < p) ? 0.0f : (r * 10.0f + 1.0f);
    }

    // Allocate result arrays for the serial and GPU versions.
    int* h_result_serial = new int[totalPairs];
    int* h_result_gpu    = new int[totalPairs];

    // --------------------------
    // Run the serial (CPU) version.
    // --------------------------
    auto start_cpu = std::chrono::high_resolution_clock::now();
    serialCtz(h_X, nsubjs, nfeats, h_result_serial);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    double cpu_time = std::chrono::duration<double, std::milli>(end_cpu - start_cpu).count();
    std::cout << "Serial CPU time: " << cpu_time << " ms" << std::endl;

    // --------------------------
    // Run the GPU version.
    // --------------------------
    // Allocate device memory for X and result array.
    float* d_X;
    int* d_result;
    cudaCheckError( cudaMalloc((void**)&d_X, nsubjs * nfeats * sizeof(float)) );
    cudaCheckError( cudaMalloc((void**)&d_result, totalPairs * sizeof(int)) );

    // Copy matrix X to device memory
    cudaCheckError( cudaMemcpy(d_X, h_X, nsubjs * nfeats * sizeof(float), cudaMemcpyHostToDevice) );

    // Configure kernel launch parameters.
    int threadsPerBlock = 128;
    int blocks = (totalPairs + threadsPerBlock - 1) / threadsPerBlock;

    // Create CUDA events for timing the kernel.
    cudaEvent_t start, stop;
    cudaCheckError( cudaEventCreate(&start) );
    cudaCheckError( cudaEventCreate(&stop) );
    cudaCheckError( cudaEventRecord(start) );

    // Launch the kernel.
    jointZeroCountKernel<<<blocks, threadsPerBlock>>>(d_X, nsubjs, nfeats, d_result);
    cudaCheckError( cudaEventRecord(stop) );
    cudaCheckError( cudaEventSynchronize(stop) );

    float gpu_time = 0;
    cudaCheckError( cudaEventElapsedTime(&gpu_time, start, stop) );
    std::cout << "CUDA kernel time: " << gpu_time << " ms" << std::endl;
    std::cout << "Speedup : " << cpu_time/gpu_time << " x" << std::endl;



    // Copy the GPU result back to host.
    cudaCheckError( cudaMemcpy(h_result_gpu, d_result, totalPairs * sizeof(int), cudaMemcpyDeviceToHost) );

    // --- Compare Results ---
    bool match = true;
    for (int i = 0; i < totalPairs; i++) {
        if (h_result_serial[i] != h_result_gpu[i]) {
            match = false;
            std::cout << "Mismatch at index " << i << ": CPU = " << h_result_serial[i]
                      << ", GPU = " << h_result_gpu[i] << std::endl;
            break;
        }else {
          //std::cout <<  h_result_serial[i] << " " << h_result_gpu[i]  << std::endl;
        }
    }
    if(match) {
        std::cout << "Results match between serial and CUDA implementations." << std::endl;
    }
    
    delete[] h_X;
    delete[] h_result_serial;
    delete[] h_result_gpu;
    cudaFree(d_X);
    cudaFree(d_result);

    return 0;
}
