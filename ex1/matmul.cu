#include <iostream>

//#ifdef __CUDACC__
#include <cuda_runtime.h>
//#endif

#include "matmul.h"
#include "test.h"
#include "common.h"
#include "mul_cpu.h"
#include "mul_gpu.h"
#include "timer.h"

void print_cuda_devices()
{
	// TODO: Task 2
	cudaDeviceProp prop;
	int count;

	cudaGetDeviceCount(&count);

	for(int i =0; i<count;i++){
		cudaGetDeviceProperties(&prop, i);
		std::cout<< "Information for device: " << i <<std::endl;
		std::cout<< "Compute Capability: " << prop.major << "." << prop.minor <<std::endl;
		std::cout<< "Multiprocessor count: " << prop.multiProcessorCount <<std::endl;
		std::cout<< "GPU clock rate: " << prop.clockRate << " GHz" <<std::endl;
		std::cout<< "Total global memory: " << prop.totalGlobalMem << " MiB" <<std::endl;
		std::cout<< "L2 Cache size: " << prop.l2CacheSize << " KiB" <<std::endl;
		std::cout<< "----------------------------------" <<std::endl;

	}

}

void matmul()
{
	// === Task 3 ===
	// TODO: Allocate CPU matrices (see matrix.cc)
	//       Matrix sizes:
	//       Input matrices:
	//       Matrix M: pmpp::M_WIDTH, pmpp::M_HEIGHT
	//       Matrix N: pmpp::N_WIDTH, pmpp::N_HEIGHT
	//       Output matrices:
	//       Matrix P: pmpp::P_WIDTH, pmpp::P_HEIGHT
	CPUMatrix cm = matrix_alloc_cpu(pmpp::M_WIDTH, pmpp::M_HEIGHT);
	CPUMatrix cn = matrix_alloc_cpu(pmpp::N_WIDTH, pmpp::N_HEIGHT);
	CPUMatrix cp = matrix_alloc_cpu(pmpp::P_WIDTH, pmpp::P_HEIGHT);
	
	// TODO: Fill the CPU input matrices with the provided test values (pmpp::fill(CPUMatrix &m, CPUMatrix &n))
	pmpp::fill(cm,cn);

	// TODO (Task 5): Start CPU timing here!
	timer_tp start = timer_now();

	// TODO: Run your implementation on the CPU (see mul_cpu.cc)
	matrix_mul_cpu(cm,cn,cp);

	// TODO (Task 5): Stop CPU timing here!
	timer_tp end = timer_now();
	float elapsed_time = timer_elapsed(start, end);
	printf("CPU proceesing took: %f ms\n", elapsed_time);

	// TODO: Check your matrix for correctness (pmpp::test_cpu(const CPUMatrix &p))
	pmpp::test_cpu(cp);

	// === Task 4 ===
	// TODO: Set CUDA device
	cudaSetDevice(0);
	CUDA_CHECK_ERROR;
	std::cout<<"CUDA device setting successful"<<std::endl;
	
	// TODO: Allocate GPU matrices (see matrix.cc)

	CPUMatrix cp2 = matrix_alloc_cpu(pmpp::P_WIDTH, pmpp::P_HEIGHT);
	GPUMatrix gm = matrix_alloc_gpu(pmpp::M_WIDTH, pmpp::M_HEIGHT);
	GPUMatrix gn = matrix_alloc_gpu(pmpp::N_WIDTH, pmpp::N_HEIGHT);
	GPUMatrix gp = matrix_alloc_gpu(pmpp::P_WIDTH, pmpp::P_HEIGHT);

	// TODO: Upload the CPU input matrices to the GPU (see matrix.cc)
	matrix_upload(cm,gm);
	matrix_upload(cn,gn);

	// TODO (Task 5): Start GPU timing here!
	cudaEvent_t evStart, evStop;
	cudaEventCreate(&evStart);
	cudaEventCreate(&evStop);
	cudaEventRecord(evStart,0);
	CUDA_CHECK_ERROR;

	// TODO: Run your implementation on the GPU (see mul_gpu.cu)
	matrix_mul_gpu(gm,gn,gp);


	// TODO (Task 5): Stop GPU timing here!
	cudaEventRecord(evStop, 0);
	cudaEventSynchronize(evStop);
	CUDA_CHECK_ERROR;

	float elapsedTime_ms;
	cudaEventElapsedTime(&elapsedTime_ms, evStart, evStop);
	CUDA_CHECK_ERROR;

	printf("CUDA proceesing took: %f ms\n", elapsedTime_ms);
	cudaEventDestroy(evStart);
	cudaEventDestroy(evStop);
	CUDA_CHECK_ERROR;

	
	// TODO: Download the GPU output matrix to the CPU (see matrix.cc)
	matrix_download(gp,cp2);

	// TODO: Check your downloaded matrix for correctness (pmpp::test_gpu(const CPUMatrix &p))
	pmpp::test_gpu(cp2);

	// TODO: Compare CPU result with GPU result (see matrix.cc)
	matrix_compare_cpu(cp2,cp);

	// TODO (Task3/4/5): Cleanup ALL matrices and and events
	matrix_free_cpu(cm);
	matrix_free_cpu(cn);
	matrix_free_cpu(cp);
	matrix_free_cpu(cp2);
	matrix_free_gpu(gm);
	matrix_free_gpu(gn);
	matrix_free_gpu(gp);
}


/************************************************************
 * 
 * TODO: Write your text answers here!
 * 
 * (Task 4) 6. Where do the differences come from?
 * 
 * Answer: TODO
 * 
 * 
 ************************************************************/
