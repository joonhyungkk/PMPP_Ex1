project(MatMul CXX CUDA)
cmake_minimum_required(VERSION 3.8)

# Set the default build type
if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel." FORCE)
endif()

# Specify the files that need to be compiled and linked.
# This will create an executable named 'matmul'.
add_executable(matmul
	main.cc
	matmul.cu
	matrix.cu
	test.cc
	mul_cpu.cc
	mul_gpu.cu
)

set_target_properties(matmul PROPERTIES CUDA_ARCHITECTURES "75")
