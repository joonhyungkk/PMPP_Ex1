#include "matrix.h"
#include "mul_cpu.h"
#include "common.h"

void matrix_mul_cpu(const CPUMatrix &m, const CPUMatrix &n, CPUMatrix &p)
{
	// TODO: Task 3
	// Write a cpu implementation of the matrix multiplication in this file.
	//Then use it and the functions from matrix.cc to compute the result matrix on the CPU matmul.cc
	for(int i= 0; i<m.width;i++)
		for(int j=0; j<n.height;j++){
			float sum = 0;
			for(int k = 0; k<m.width; k++){
				float a = m.elements[i*m.width +k];
				float b = n.elements[k*m.width +j];
				sum+= a*b;
			}
			p.elements[i*m.width +j] =sum;
			}
}

