#include <cuda_runtime.h>

// NOTE: if you include stdio.h, you can use printf inside your kernel

#include "common.h"
#include "matrix.h"
#include "mul_gpu.h"
#include "stdio.h"

// TODO (Task 4): Implement matrix multiplication CUDA kernel

__global__ void MatrixMulKernel(float* m, float* n, float* p, int width){
	int row = threadIdx.y * blockDim.y + threadIdx.y;
	int col = threadIdx.x * blockDim.x + threadIdx.x;


	float value = 0.0;
	if( row < width && col<width){
	for(int k = 0; k < width; k++){
		value += m[row * width +k] * n[k * width + col];
		}
	}
	p[row * width + col] = value;

}

void matrix_mul_gpu(const GPUMatrix &m, const GPUMatrix &n, GPUMatrix &p)
{
	// TODO (Task 4): Determine execution configuration and call CUDA kernel
	dim3 dimBlock(p.width, p.height);
	dim3 dimGrid(1,1);
	MatrixMulKernel<<<dimGrid,dimBlock>>>(m.elements,n.elements,p.elements,p.width);
	CUDA_CHECK_ERROR;


}

