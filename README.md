# Pruning in Neural Network 

## Improving the computation performances using sparse matrix multiplications

This repository contains all you need to repeat the experiments in the article : https://medium.com/@yanis.chaigneau/pruning-in-neural-networks-541af4f9a899

It is separated in three parts :

# The matrices multiplication with pytorch.sparse

To repeat this experiment, all you need to do is to change the code in the two files gpu_torch.py and plot_results.txt where you want to store the logs, you need as well to install AIPowerMeter with :

`git clone https://github.com/GreenAI-Uppa/AIPowerMeter`

in the same repository as pruningSPMM.

Then you can install the requirements.txt with pip in a virtual env :

`pip install -r requirements.txt`

and then 

`python gpu_torch.py` to start the experiment

`python plot_results.py` to save the results aftewards as a plot on a file


# cuSPARSE :

This experiment compares the execution time of a sparse matrix multiplication between cuSparse with blocked-ELL format and cuBlas hGEMM.

To launch the experiment with Python, simply run the following script:

`python cusparse_test.py`

To compile the Cpp script, run: 

g++ -Wall cusparse_blocked_ell_spmm.c -o ccusparse_blocked_ell_spmm.o -I $path_CUDA_include -L $path_CUDA_lib -lcudart -lcuda -lcusparse -lcublas -fopenmp -g -O2 -std=c++11

with
$path_CUDA_include the path to the include directory of your CUDA version (minimum 11.4) (traditionaly /usr/local/cuda-11.4/targets/x86_64-linux/include/ )
$path_CUDA_lib the path to the lib directory of your CUDA version (minimum 11.4) (traditionaly /usr/local/cuda-11.4/targets/x86_64-linux/lib )

Then run it with: 

`./cusparse_blocked_ell_spmm.o $size_A $nb_rows_B $blocksize $number_blocks_per_row $number_repetitions`

Arguments:
	size_A: number of rows and columns of A
	nb_rows_B: Number of rows of the dense matrix
	blocksize: The size of the individual blocks of the sparse matrix (square blocks)
	number_blocks_per_row: Number blocks per row of the sparse matrix A. If 1, a block diagonal matrix is built.
	number_repetitions: Number of time to compute the matrix multiplication. It can be useful for energy consumption measures.
	
# SparseLinear

Launch:

`python test_snip_sparse.py`

Analyze the results with:

`python test_results.py`
