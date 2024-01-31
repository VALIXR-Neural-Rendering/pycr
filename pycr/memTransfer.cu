#include <iostream>

#include "GL\glew.h"
#include "GLFW\glfw3.h"

#include <cuda_gl_interop.h>

#include <helper_cuda.h>
//#include <helper_cuda_gl.h>

#include "memTransfer.h"

using namespace std;

__global__ void memTransferCuda(cudaTextureObject_t src, void *dst, unsigned int wid, unsigned int ht) {
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

	// Get texture value and write to global memory
	((GLubyte*)dst)[y * wid + x] = tex2D<GLubyte>(src, x, y);
}

void memTransfer(cudaArray_t srcArray, void* dst, size_t width, size_t height) {
	// Allocate pitched linear memory
	int fmtSize = sizeof(GLubyte);
	size_t pitch;
	cout << "Format size: " << fmtSize << endl;
	checkCudaErrors(cudaMalloc(&dst, width * fmtSize * 4 * height));
	//checkCudaErrors(cudaMallocPitch(&dst, &pitch, width * fmtSize * 4, height));
	//cout << "Alloc size: " << pitch * extent.height << endl;
	//cout << "Pitch: " << pitch << endl;


	// Check if the pointers are on device or host
	cudaPointerAttributes attr;
	checkCudaErrors(cudaPointerGetAttributes(&attr, dst));
	cout << "Dst attributes: " << attr.devicePointer << " " << attr.hostPointer << " " << attr.type << endl;
	checkCudaErrors(cudaPointerGetAttributes(&attr, srcArray));
	cout << "Source attributes: " << attr.devicePointer << " " << attr.hostPointer << " " << attr.type << endl;

	// Memcpy or Memset
	checkCudaErrors(cudaMemcpy2DFromArray(dst, width * fmtSize * 4, srcArray, 0, 0, width * fmtSize * 4, height, cudaMemcpyDefault));
	//cout << "Completed memory transfer" << endl;
	//checkCudaErrors(cudaMemset2D(pycuda_tex, pitch, 100, extent.width * fmtSize * 4, extent.height));
	//checkCudaErrors(cudaFree(pycuda_tex));


	//// Create texture object
	//cudaTextureObject_t texObj = 0;

	//cudaResourceDesc resDesc;
	//memset(&resDesc, 0, sizeof(resDesc));
	//resDesc.resType = cudaResourceTypeArray;
	//resDesc.res.array.array = srcArray;

	//cudaTextureDesc texDesc;
	//memset(&texDesc, 0, sizeof(texDesc));
	//texDesc.addressMode[0] = cudaAddressModeClamp;
	//texDesc.addressMode[1] = cudaAddressModeClamp;
	//texDesc.filterMode = cudaFilterModePoint;
	//texDesc.readMode = cudaReadModeElementType;
	//texDesc.normalizedCoords = false;

	//cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);
	//cout << "Created texture object" << endl;
 //   
	//dim3 dimBlock(16, 16);
	//dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x, (height + dimBlock.y - 1) / dimBlock.y);
 //   memTransferCuda <<< dimGrid, dimBlock >>> (texObj, dst, width, height);
}