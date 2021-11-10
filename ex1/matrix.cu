#include <cstddef>
#include <iostream>
#include <iomanip>
#include <cstdlib>

#include <cuda_runtime.h>

#include "common.h"
#include "matrix.h"


CPUMatrix matrix_alloc_cpu(int width, int height)
{
	CPUMatrix m;
	m.width = width;
	m.height = height;
	m.elements = new float[m.width * m.height];
	return m;
}
void matrix_free_cpu(CPUMatrix &m)
{
	delete[] m.elements;
}

GPUMatrix matrix_alloc_gpu(int width, int height)
{
	// TODO (Task 4): Allocate memory at the GPU
	GPUMatrix gm;
	gm.width = width;
	gm.height = height;
	gm.elements = (float *)malloc(width*height*sizeof(float)) ;
	cudaMallocPitch(&gm.elements, &gm.pitch, width, height);

	CUDA_CHECK_ERROR;
	return gm;

}
void matrix_free_gpu(GPUMatrix &m)
{
	// TODO (Task 4): Free the memory
	cudaFree(m.elements);
	CUDA_CHECK_ERROR;

}

void matrix_upload(const CPUMatrix &src, GPUMatrix &dst)
{
	// TODO (Task 4): Upload CPU matrix to the GPU
	int size = src.width*src.height*sizeof(float);
	cudaMemcpy2D(dst.elements, dst.pitch, src.elements, size, src.width, src.height, cudaMemcpyHostToDevice);
	CUDA_CHECK_ERROR;


}
void matrix_download(const GPUMatrix &src, CPUMatrix &dst)
{
	// TODO (Task 4): Download matrix from the GPU
	int size = dst.width*dst.height*sizeof(float);
	cudaMemcpy2D(dst.elements, size, src.elements, src.pitch, dst.width, dst.height, cudaMemcpyDeviceToHost);
	CUDA_CHECK_ERROR;

}

void matrix_compare_cpu(const CPUMatrix &a, const CPUMatrix &b)
{
	// TODO (Task 4): compare both matrices a and b and print differences to the console
	for(int i = 0; i <a.width*a.height ;i++){
		if(a.elements[i]!= b.elements[i])
			std::cout<< "Difference detected. "<<i<< " is different. " << a.elements[i] << "and " << b.elements[i] << " are different" <<std::endl;
		else
			continue;
	}
}
		
