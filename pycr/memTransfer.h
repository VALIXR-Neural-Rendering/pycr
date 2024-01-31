#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

void memTransfer(cudaArray_t srcArray, void* dst, size_t width, size_t height);