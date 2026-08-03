#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <cuda_runtime.h>

#define FH_K1 0x9E3779B97F4A7C15ULL
#define FH_K2 0xBF58476D1CE4E5B9ULL
#define FH_K3 0x94D049BB133111EBULL

// Macro for simple CUDA error checking
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

/* 
 * Device-side functions for GPU execution
 */
static inline __device__ uint64_t fh_load64(const uint8_t *p) {
    uint64_t v;
    // memcpy works on modern CUDA for small constant sizes, 
    // but explicit casting or Type Punning is standard on GPU.
    v = *(const uint64_t*)p; 
    return v;
}

static inline __device__ uint64_t fh_mix(uint64_t h) {
    h ^= h >> 33;
    h *= FH_K2;
    h ^= h >> 29;
    h *= FH_K3;
    h ^= h >> 32;
    return h;
}

// Device-side core hashing logic
static inline __device__ uint64_t fast_hash_device(const void *data, size_t len) {
    const uint8_t *p = (const uint8_t *)data;
    size_t words = len / 8;

    uint64_t h0 = FH_K1, h1 = FH_K2, h2 = FH_K3, h3 = FH_K1 ^ FH_K2;

    for (size_t i = 0; i < words; i += 4) {
        h0 = (h0 ^ fh_load64(p + (i + 0) * 8)) * FH_K1;
        h1 = (h1 ^ fh_load64(p + (i + 1) * 8)) * FH_K2;
        h2 = (h2 ^ fh_load64(p + (i + 2) * 8)) * FH_K3;
        h3 = (h3 ^ fh_load64(p + (i + 3) * 8)) * FH_K1;
    }

    uint64_t h = h0;
    h = (h * FH_K1) ^ h1;
    h = (h * FH_K2) ^ h2;
    h = (h * FH_K3) ^ h3;

    return fh_mix(h ^ (uint64_t)len);
}

/* 
 * CUDA Global Kernel
 * Each thread computes the hash of one independent data block.
 */
__global__ void fast_hash_kernel(const uint8_t *d_in, uint64_t *d_out, size_t block_len, int num_blocks) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < num_blocks) {
        // Find the specific memory offset for this thread's block
        const uint8_t *my_block = d_in + (idx * block_len);
        d_out[idx] = fast_hash_device(my_block, block_len);
    }
}

/* 
 * Host Helper function for patterns
 */
static void fill_pattern(uint8_t *buf, size_t n, uint8_t seed) {
    for (size_t i = 0; i < n; i++) buf[i] = (uint8_t)(seed + i * 31);
}

/* 
 * Main Execution 
 */
int main(void) {
    // We will process 3 items on the GPU to match the original demo
    const int num_blocks = 3;
    uint64_t h_results[num_blocks] = {0};

    // Buffer dimensions
    size_t len128 = 128;
    size_t len4k  = 4096;

    // Allocate host memory buffers
    uint8_t *h_block128 = (uint8_t*)malloc(len128);
    uint8_t *h_block4k  = (uint8_t*)malloc(len4k);
    uint8_t *h_block4k_b = (uint8_t*)malloc(len4k);

    fill_pattern(h_block128, len128, 0xA5);
    fill_pattern(h_block4k,  len4k,  0x11);
    fill_pattern(h_block4k_b, len4k, 0x11);
    h_block4k_b[2000] ^= 0x01; // flip ONE bit

    // To batch them into standard kernels efficiently, we run two separate GPU launches:
    // Launch 1: The 128-byte block
    // Launch 2: The two 4096-byte blocks (consecutively packed)

    // Device allocations
    uint8_t *d_in128, *d_in4k;
    uint64_t *d_out128, *d_out4k;

    CUDA_CHECK(cudaMalloc((void**)&d_in128, len128));
    CUDA_CHECK(cudaMalloc((void**)&d_out128, 1 * sizeof(uint64_t)));

    CUDA_CHECK(cudaMalloc((void**)&d_in4k, len4k * 2));
    CUDA_CHECK(cudaMalloc((void**)&d_out4k, 2 * sizeof(uint64_t)));

    // Copy data from Host to Device
    CUDA_CHECK(cudaMemcpy(d_in128, h_block128, len128, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_in4k, h_block4k, len4k, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_in4k + len4k, h_block4k_b, len4k, cudaMemcpyHostToDevice));

    // Execution configurations (1 block, sufficient threads)
    int threadsPerBlock = 256;
    int blocksPerGrid1 = (1 + threadsPerBlock - 1) / threadsPerBlock;
    int blocksPerGrid2 = (2 + threadsPerBlock - 1) / threadsPerBlock;

    // Launch CUDA Kernels
    fast_hash_kernel<<<blocksPerGrid1, threadsPerBlock>>>(d_in128, d_out128, len128, 1);
    fast_hash_kernel<<<blocksPerGrid2, threadsPerBlock>>>(d_in4k, d_out4k, len4k, 2);

    // Synchronize and check for runtime launch errors
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy results back to Host
    CUDA_CHECK(cudaMemcpy(&h_results[0], d_out128, 1 * sizeof(uint64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_results[1], d_out4k, 2 * sizeof(uint64_t), cudaMemcpyDeviceToHost));

    // Display identical verification patterns to original code
    printf("128B hash : %016llx\n", (unsigned long long)h_results[0]);
    printf("4KB-a hash : %016llx\n", (unsigned long long)h_results[1]);
    printf("4KB-b hash : %016llx (one bit flipped at byte 2000)\n", (unsigned long long)h_results[2]);

    printf("\none-bit-different buffers match? %s (should be no)\n", 
           (h_results[1] == h_results[2]) ? "yes" : "no");

    // Free all allocated spaces
    free(h_block128); free(h_block4k); free(h_block4k_b);
    cudaFree(d_in128); cudaFree(d_out128);
    cudaFree(d_in4k); cudaFree(d_out4k);

    return 0;
}
