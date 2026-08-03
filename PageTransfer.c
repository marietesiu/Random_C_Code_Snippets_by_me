#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define PAGE_SIZE 4096                              // Standard 4KB x86 system page size
#define NUM_PAGES 256                               // Number of pages to transfer
#define N ((NUM_PAGES * PAGE_SIZE) / sizeof(float)) // Total float elements

// Macro to handle and print CUDA errors cleanly
#define CUDA_CHECK(cmd) do { \
    cudaError_t err = cmd; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA page transfer error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

int main() {
    size_t bytes = N * sizeof(float);
    float *h_data = NULL;
    float *d_data = NULL;
    cudaStream_t stream;

    printf("Allocating %d pages (%zu bytes)...\n", NUM_PAGES, bytes);

    // 1. Allocate native CPU memory aligned exactly to hardware page boundaries (4KB)
    if (posix_memalign((void**)&h_data, PAGE_SIZE, bytes) != 0) {
        fprintf(stderr, "Fatal: Page alignment failed.\n");
        return EXIT_FAILURE;
    }

    // 2. Initialize the data inside the physical CPU memory pages
    for (size_t i = 0; i < N; i++) {
        h_data[i] = 1.0f;
    }

    // 3. Pin the actual OS pages directly so the GPU DMA engine can read them
    // cudaHostRegisterPortable: allows all CUDA contexts to use this pinned space
    CUDA_CHECK(cudaHostRegister(h_data, bytes, cudaHostRegisterPortable)); // higher bandwith

    // 4. Allocate equivalent storage space in GPU VRAM
    CUDA_CHECK(cudaMalloc((void**)&d_data, bytes));

    // 5. Create an isolated execution stream for non-blocking operations
    CUDA_CHECK(cudaStreamCreate(&stream));

    // 6. Asynchronously push the physical memory pages over the PCIe bus
    //printf("Streaming pages asynchronously to GPU VRAM...\n");
    CUDA_CHECK(cudaMemcpyAsync(d_data, h_data, bytes, cudaMemcpyHostToDevice, stream));

    // 7. Block the CPU thread until this specific stream's page transfer finishes
    CUDA_CHECK(cudaStreamSynchronize(stream));
    //printf("Transfer complete!\n");

    // 8. Clean up and unpin resources
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaHostUnregister(h_data));
    free(h_data);

    return EXIT_SUCCESS;
}
