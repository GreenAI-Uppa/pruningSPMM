/*
Author: Yanis Chaigneau
Organization: GreenAI UPPA
Date: 05/04/2022
Description: This code is inspired by the cuSPARSE/spmm_blockedell repo described
by Takuma Yamaguchi & Federico Busato in "Accelerating Matrix Multiplication with 
Block Sparse Format and NVIDIA Tensor Cores" (https://developer.nvidia.com/blog/accelerating-matrix-multiplication-with-block-sparse-format-and-nvidia-tensor-cores/)
This code computes the Sparse-matrix dense-matrix multiplication (SpMM) C = A * B where A is a sparse matrix
represented in the Blocked-Ellpack format and B a dense matrix.

The matrix computation with cuBlas is taken from https://github.com/hma02/cublasHgemm-P100 with slight modifications.

Outputs:

The computation times with cuSparse and cuBlas.

Usage:
First compile the code with:
	g++ -Wall cusparse_blocked_ell_spmm.c -o ccusparse_blocked_ell_spmm.o -I $path_CUDA_include -L $path_CUDA_lib -lcudart -lcuda -lcusparse -lcublas -fopenmp -g -O2 -std=c++11
	
	$path_CUDA_include the path to the include directory of your CUDA version (minimum 11.4) (traditionaly /usr/local/cuda-11.4/targets/x86_64-linux/include/ )
	$path_CUDA_lib the path to the lib directory of your CUDA version (minimum 11.4) (traditionaly /usr/local/cuda-11.4/targets/x86_64-linux/lib )
	
Then run the .exe:

./cusparse_blocked_ell_spmm.o $size_A $nb_rows_B $blocksize $number_blocks_per_row $number_repetitions: 

Arguments:
	size_A: number of rows and columns of A
	nb_rows_B: Number of rows of the dense matrix
	blocksize: The size of the individual blocks of the sparse matrix (square blocks)
	number_blocks_per_row: Number blocks per row of the sparse matrix A. If 1, a block diagonal matrix is built.
	number_repetitions: Number of time to compute the matrix multiplication. It can be useful for energy consumption measures.
*/

#include "cusparse_blocked_ell_spmm.h"

// Fill the array A(nr_rows_A, nr_cols_A) with random numbers on CPU
void CPU_fill_rand(__half *A, int nr_rows_A, int nr_cols_A) {

    for(int i = 0; i < nr_rows_A * nr_cols_A; i++){
		A[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);	
	}
}

void CPU_fill_rand_sparse(__half *A, int nr_rows_A, int nr_cols_A, int A_ell_blocksize, __half *A_values) {
	int step_values = 0;
	//print_matrix(A, nr_rows_A, nr_cols_A);
    for(int i = 0; i < nr_rows_A; i++){
		for(int j = 0; j< nr_cols_A; j++)
		{
			int step = ((int) i / A_ell_blocksize);
			if(j < A_ell_blocksize * (step + 1) && j>= step * A_ell_blocksize)
			{
				//std::cout<<A_values[step_values]<<std::endl;
				//A[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
				
				A[i+nr_rows_A*j] = A_values[step_values];
				step_values = step_values + 1;
			}
			else
			{
				A[i+nr_rows_A*j] = 0.0f;
			}	
		}		
	}
	
	//print_matrix(A, nr_rows_A, nr_cols_A);
}


// Main function to calculate a matrix multplication with two methods: cuSparse and cuBlas
int main(int argc, char** argv) {

	// Initialize the random numbers with a seed
    srand(0);
    
	if(argc != 6){
		throw std::invalid_argument( "You must specify all the required arguments: ./cusparse_blocked_ell_spmm.o $size_A $nb_rows_B $blocksize $number_blocks_per_row $number_repetitions" );
	}
	std::cout<<"Executing cuSparse to compare the results"<<std::endl;
    // Host problem definition
    int   A_num_rows      = atoi(argv[1]);
    int   A_num_cols      = atoi(argv[1]);
    int   A_ell_blocksize = atoi(argv[3]);
    int   A_ell_cols      = atoi(argv[3])*atoi(argv[4]);
    int   A_num_blocks    = A_ell_cols * A_num_rows /
                           (A_ell_blocksize * A_ell_blocksize);
    int   B_num_rows      = A_num_cols;
    int   B_num_cols      = atoi(argv[2]);
    int   ldb             = B_num_rows;
    int   ldc             = A_num_rows;
    int   B_size          = ldb * B_num_cols;
    int   C_size          = ldc * B_num_cols;
	
    int   hA_columns [A_num_blocks]; 

	int num_repetitions = atoi(argv[5]);
         
	int verbose = 0;
	int save_results = 1;
	std::string pathToFile = "results.txt";
	
	float total_params = A_num_blocks*A_ell_blocksize*A_ell_blocksize;
	float total_elems = (A_num_rows*A_num_cols);
	float sparsity = (total_elems - total_params)*100/total_elems;
	
	
	
    for(int i=0; i<A_num_blocks;i++)
    {
		hA_columns[i] = i; // The blocks on each rows are located diagonally.
		
		//hA_columns[i] = rand() % ((int) A_num_rows / A_ell_blocksize); // Uncomment this line to place the blocks randomly
    }

	if(verbose == 1){
		std::cout <<"A_num_rows "<<A_num_rows<<std::endl;
		std::cout <<"A_num_cols "<<A_num_cols<<std::endl;
		std::cout <<"B_num_rows "<<B_num_rows<<std::endl;
		std::cout <<"B_num_cols "<<B_num_cols<<std::endl;
		std::cout <<"A_ell_blocksize "<<A_ell_blocksize<<std::endl;
		std::cout <<"A_ell_cols "<<A_ell_cols<<std::endl;
		std::cout <<"A_num_blocks "<<A_num_blocks<<std::endl<<std::endl;
		std::cout <<"Sparsity of A: "<<sparsity<<std::endl;
    }

	// Allocate memory for A, B and C matrices
	
    __half* hA_values = (__half*)malloc(A_ell_cols*A_num_rows*sizeof(__half));
    __half* hB = (__half*)malloc(B_num_cols*B_num_rows*sizeof(__half));
    __half* hC = (__half*)malloc(B_num_cols*A_num_rows*sizeof(__half));    
	
	// Generate the random values of A and B
    for(size_t x=0;x<static_cast <size_t>(A_ell_cols*A_num_rows);x++)
    {
		hA_values[static_cast <size_t>(x)] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
    }

  
    for(size_t i=0; i<static_cast <size_t>(B_num_rows);i++)
    {
		for(size_t j=0; j<static_cast <size_t>(B_num_cols);j++)
		{
			hB[static_cast <size_t>(i+B_num_rows*j)] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
		}
    }
	
    //Initialize C with zeros
     for(size_t i=0; i<static_cast <size_t>(A_num_rows);i++)
    {
        for(size_t j=0; j<static_cast <size_t>(B_num_cols);j++)
        {
			hC[static_cast <size_t>(i+A_num_rows*j)] = 0.0f;
		}
    }
    
	
    float alpha           = 1.0f;
    float beta            = 0.0f;
	
    //--------------------------------------------------------------------------
    // Check compute capability
    cudaDeviceProp props;
    CHECK_CUDA( cudaGetDeviceProperties(&props, 0) )
    if (props.major < 7) {
      std::printf("cusparseSpMM with blocked ELL format is supported only "
                  "with compute capability at least 7.0\n");
      return EXIT_UNSUPPORTED;
    }
	
    //--------------------------------------------------------------------------
    // Device memory management
	
	int    *dA_columns;
	__half *dA_values, *dB, *dC;
	CHECK_CUDA( cudaMalloc((void**) &dA_columns, A_num_blocks * sizeof(int)) )
	CHECK_CUDA( cudaMalloc((void**) &dA_values,
									A_ell_cols * A_num_rows * sizeof(__half)) )
	CHECK_CUDA( cudaMalloc((void**) &dB, B_size * sizeof(__half)) )
	CHECK_CUDA( cudaMalloc((void**) &dC, C_size * sizeof(__half)) )


	// Copy to device
	CHECK_CUDA( cudaMemcpy(dA_columns, hA_columns,
						   A_num_blocks * sizeof(int),
						   cudaMemcpyHostToDevice) )
	CHECK_CUDA( cudaMemcpy(dA_values, hA_values,
						   A_ell_cols * A_num_rows * sizeof(__half),
						   cudaMemcpyHostToDevice) )
	CHECK_CUDA( cudaMemcpy(dB, hB, B_size * sizeof(__half),
						   cudaMemcpyHostToDevice) )
	CHECK_CUDA( cudaMemcpy(dC, hC, C_size * sizeof(__half),
						   cudaMemcpyHostToDevice) )
	//--------------------------------------------------------------------------
	// CUSPARSE APIs
	cusparseHandle_t     handle = NULL;
	cusparseSpMatDescr_t matA;
	cusparseDnMatDescr_t matB, matC;
	void*                dBuffer    = NULL;
	size_t               bufferSize = 0;
	
	CHECK_CUSPARSE( cusparseCreate(&handle) )
	
	// Create sparse matrix A in blocked ELL format
	CHECK_CUSPARSE( cusparseCreateBlockedEll(
									  &matA,
									  A_num_rows, A_num_cols, A_ell_blocksize,
									  A_ell_cols, dA_columns, dA_values,
									  CUSPARSE_INDEX_32I,
									  CUSPARSE_INDEX_BASE_ZERO, CUDA_R_16F) )
	// Create dense matrix B
	CHECK_CUSPARSE( cusparseCreateDnMat(&matB, A_num_cols, B_num_cols, ldb, dB,
										CUDA_R_16F, CUSPARSE_ORDER_COL) )
	// Create dense matrix C
	CHECK_CUSPARSE( cusparseCreateDnMat(&matC, A_num_rows, B_num_cols, ldc, dC,
										CUDA_R_16F, CUSPARSE_ORDER_COL) )
	// allocate an external buffer if needed
	CHECK_CUSPARSE( cusparseSpMM_bufferSize(
								 handle,
								 CUSPARSE_OPERATION_NON_TRANSPOSE,
								 CUSPARSE_OPERATION_NON_TRANSPOSE,
								 &alpha, matA, matB, &beta, matC, CUDA_R_32F,
								 CUSPARSE_SPMM_ALG_DEFAULT, &bufferSize) )
	CHECK_CUDA( cudaMalloc(&dBuffer, bufferSize) )

	// execute SpMM
	const clock_t begin_time = clock();
	for(int j=0;j<num_repetitions;j++)
	{
	CHECK_CUSPARSE( cusparseSpMM(handle,
								 CUSPARSE_OPERATION_NON_TRANSPOSE,
								 CUSPARSE_OPERATION_NON_TRANSPOSE,
								 &alpha, matA, matB, &beta, matC, CUDA_R_32F,
								 CUSPARSE_SPMM_ALG_DEFAULT, dBuffer) )
	}
	
	float time_taken = float( clock () - begin_time ) /  CLOCKS_PER_SEC;
	std::cout << "CUSPARSE: Execution done with "<<num_repetitions<<" iterations. Total time: "<<time_taken << std::endl;
	
	if(save_results == 1){
		std::ofstream myfile;
		myfile.open(pathToFile);
		myfile << time_taken << "\n";
		myfile.close();
	}
	
	// destroy matrix/vector descriptors
	CHECK_CUSPARSE( cusparseDestroySpMat(matA) )
	CHECK_CUSPARSE( cusparseDestroyDnMat(matB) )
	CHECK_CUSPARSE( cusparseDestroyDnMat(matC) )
	CHECK_CUSPARSE( cusparseDestroy(handle) )
	
	// CHECK_CUDA( cudaMemcpy(hC, dC, C_size * sizeof(__half), cudaMemcpyDeviceToHost) ) // Uncomment to get the result back to the cpu
	//print_matrix(hC, B_num_cols, A_num_rows); // Uncomment to get print the result
	//--------------------------------------------------------------------------
	
	//--------------------------------------------------------------------------
	// device memory deallocation
	
	
	CHECK_CUDA( cudaFree(dBuffer) )
	CHECK_CUDA( cudaFree(dA_columns) )
	CHECK_CUDA( cudaFree(dA_values) )
	CHECK_CUDA( cudaFree(dB) )
	CHECK_CUDA( cudaFree(dC) )

	
	std::cout<<std::endl;
	
	std::cout<<"Now executing cuBlas to compare the results"<<std::endl;
	calculateCublas(hB, hC, A_num_rows, A_num_cols, B_num_rows, B_num_cols, num_repetitions, A_ell_blocksize, hA_values, save_results, pathToFile);
	
	std::cout<<"Program termination"<<std::endl;
	
	
	
    return EXIT_SUCCESS;
}




//CUBLAS

const char* cublasGetErrorString(cublasStatus_t status)
{
    switch(status)
    {
        case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE"; 
        case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH"; 
        case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED"; 
        case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR"; 
    }
    return "unknown error";
}

// Convenience function for checking CUDA runtime API results
// can be wrapped around any runtime API call. No-op in release builds.
inline
cudaError_t checkCuda(cudaError_t result)
{
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
  return result;
}

inline
cublasStatus_t checkCublas(cublasStatus_t result)
{
  if (result != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cublasGetErrorString(result));
    assert(result == CUBLAS_STATUS_SUCCESS);
  }
  return result;
}


void print_matrix(__half *A, int A_num_rows, int A_num_cols)
{
	std::cout << "Values of A" << std::endl;
    for(size_t i=0; i<static_cast <size_t>(A_num_rows);i++)
    {
        for(size_t j=0; j<static_cast <size_t>(A_num_cols);j++)
        {
			std::cout << A[i+A_num_rows*j] << " ";
		}
		std::cout << std::endl;
    }
}
 


void calculateCublas(__half *hB, __half *hC, int A_num_rows, int A_num_cols, int B_num_rows, int B_num_cols, int num_repetitions, int A_ell_blocksize, __half *A_values, int save_results, std::string pathToFile)
{
	
    cublasStatus_t stat;
    cublasHandle_t handle;
    checkCublas(cublasCreate(&handle));
  
  
    // Allocate an array for the dense matrix A (still sparse)
    __half *hA = (__half *)malloc(A_num_rows * A_num_cols * sizeof(__half));
  
    // Fill A with the same values as cuSparse
    CPU_fill_rand_sparse(hA, A_num_rows, A_num_cols, A_ell_blocksize, A_values);

    
	// Memory management
  	__half *d_A, *d_B, *d_C;
    checkCuda(cudaMallocManaged(&d_A, A_num_rows * A_num_cols * sizeof(__half)));
    checkCuda(cudaMallocManaged(&d_B, B_num_rows * B_num_cols * sizeof(__half)));
    checkCuda(cudaMallocManaged(&d_C, B_num_cols * A_num_rows * sizeof(__half)));
    
    for (int i = 0; i < A_num_rows * A_num_cols; i++) {
      d_A[i] = approx_float_to_half(hA[i]);
  	  d_B[i] = approx_float_to_half(hB[i]);
  	  d_C[i] = approx_float_to_half(hC[i]);
    }
    
    
    const __half alf = approx_float_to_half(1.0);
    const __half bet = approx_float_to_half(0.0);
    const __half *alpha = &alf;
    const __half *beta = &bet;
	
	// Define constants
	int   m 			  = A_num_cols;
	int   n			      = B_num_cols;
	int   k 			  = B_num_cols;
	int   lda   		  = A_num_rows;
	int   ldb             = B_num_rows;
    int   ldc             = A_num_rows;
	
	
	// Matrix calculation
	const clock_t begin_time = clock();
	for(int j=0;j<num_repetitions;j++)
	{
		stat = cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, d_A, lda, d_B, ldb, beta, d_C, ldc); 
	}
	
	float time_taken = float( clock () - begin_time ) /  CLOCKS_PER_SEC;
	std::cout << "CUBLAS: Execution done with "<<num_repetitions<<" iterations. Total time: "<<time_taken << std::endl;
	
	if(save_results == 1){
		std::ofstream myfile;
		myfile.open(pathToFile, std::fstream::app);
		myfile << time_taken << "\n";
		myfile.close();
	}
	//cudaMemcpy(hC, d_C, B_num_cols* A_num_rows * sizeof(__half),
	//						   cudaMemcpyDeviceToHost);
	//Free GPU memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
	
    //print_matrix(hC, B_num_cols, A_num_rows);

    // Free CPU memory
    free(hA);
    free(hB);
    free(hC);
}
