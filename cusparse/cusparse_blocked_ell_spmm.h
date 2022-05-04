/*
Author: Yanis Chaigneau
Organization: GreenAI UPPA
Date: 05/04/2022
*/

#include <cuda_fp16.h>        
#include <cuda_runtime_api.h> 
#include <cusparse.h>         
#include <cstdio>            
#include <cstdlib>           
#include <random>
#include <iostream>
#include <ostream>
#include <unistd.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "fp16_conversion.h"
#include <ctime>
#include <fstream>
#include <stdexcept>


#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        std::printf("CUDA API failed at line %d with error: %s (%d)\n",        \
               __LINE__, cudaGetErrorString(status), status);                  \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        std::printf("CUSPARSE API failed at line %d with error: %s (%d)\n",    \
               __LINE__, cusparseGetErrorString(status), status);              \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

void calculateCublas(__half *hB, __half *hC, int A_num_rows, int A_num_cols, int B_num_rows, int B_num_cols, int num_repetitions, int A_ell_blocksize, __half *A_values, int save_results, std::string pathToFile);
void print_matrix(__half *A, int A_num_rows, int A_num_cols);
const int EXIT_UNSUPPORTED = 2;