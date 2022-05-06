"""
Author: Nicolas Tirel
Organization: GreenAI UPPA
Date: 05/09/2022
Description: This code is used to compute matrix multiplications with different sizes and density using pytorch.sparse. It also uses AIPowerMeter, a library
developed by GreenAI UPPA to monitor the energy consumption.
There are three main functions which correspond to a matrix multiplication with two dense matrices, a matrix multiplication between a dense and a sparse one with a variation in size and density
and finally a multiplication between a dense matrix and a sparse matrix with a density fixed of 1/100
The last functions called print_results is here to print on the terminal the energy consumption of the experiments in Joules whenever it's possible (GPU, CPU, RAM)

For the AIPowerMeter library, you need to install with `git clone https://github.com/GreenAI-Uppa/AIPowerMeter` and keep the AIPowerMeter on the same folder
To make sure torch has access to your gpu, you may need to install it with the specific cuda version, for example :
`pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113` (with pip)
or
`conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch` (with conda)
"""

import torch
from AIPowerMeter.deep_learning_power_measure.power_measure import experiment, parsers
import time

# How much matrices multiplication you want
iters = int(1000000)

# Dense matrix multiplication (GEMM) with a varying size
def dense_varying(sizes = [2**i for i in range(2,13)]):
    for i,size in enumerate(sizes):
        # If there is no GPU, or the GPU is unreachable when executed, then the programm will still run but using cpu
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        densemat = torch.randn(size,size).to(device)
        multmat = torch.randn(size,size).to(device)

        # The directory where you want to store the logs
        dir_name = f"logs/dense_varying/{size}"
        driver = parsers.JsonParser(dir_name)
        exp = experiment.Experiment(driver)

        p, q = exp.measure_yourself(period=2) # measure every 2 seconds
        start_xp = time.time()
        for _ in range(iters):
            torch.mm(densemat,multmat)
        print(round(time.time()-start_xp,3))
        
        q.put(experiment.STOP_MESSAGE)

# Sparse matrix multiplication (SpMM) with a varying size n and a varying density 1/n
def sparse_varying(sizes = [2**i for i in range(2,13)]):
    for i,size in enumerate(sizes):
        print('Multiplying matrices of size = {0}x{1}'.format(size,size))
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        spmat = torch.sparse_coo_tensor(indices = torch.tensor([torch.arange(size).tolist(),torch.arange(size).tolist()]), values = torch.randn(size), size=[size,size], dtype=torch.float32, device=device)
        multmat = torch.randn(size,size).to(device)

        # Track energy and time consumption and store the result in a file 
        dir_name = f"logs/sparse_varying/{size}"
        driver = parsers.JsonParser(dir_name)
        exp = experiment.Experiment(driver)

        p, q = exp.measure_yourself(period=2) # measure every 2 seconds
        start_xp = time.time()
        for _ in range(iters):
            torch.sparse.mm(spmat,multmat)
        print(round(time.time()-start_xp,3))
        q.put(experiment.STOP_MESSAGE)

# Sparse result with a varying size n and a fixed density of 1/100
def sparse_fixed(sizes = [2**i for i in range(2,13)]):
    for i,size in enumerate(sizes):
        print('Multiplying matrices of size = {0}x{1}'.format(size,size))
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        nnz = size*size//100
        spmat = torch.sparse_coo_tensor(indices = torch.tensor([torch.randint(size,(nnz,)).tolist(),torch.randint(size,(nnz,)).tolist()]), values = torch.randn(nnz), size=[size,size], dtype=torch.float32, device=device)
        multmat = torch.randn(size,size).to(device)

        dir_name = f"logs/sparse_fixed/{size}"
        driver = parsers.JsonParser(dir_name)
        exp = experiment.Experiment(driver)

        p, q = exp.measure_yourself(period=2) # measure every 2 seconds
        start_xp = time.time()
        for _ in range(iters):
            torch.sparse.mm(spmat,multmat)
        print(round(time.time()-start_xp,3))
        q.put(experiment.STOP_MESSAGE)

# Print the consumption results for all the size at the end of the experiment
def print_results(sizes = [2**i for i in range(2,13)]):
    for i,size in enumerate(sizes):
        filename = f"input_{size}"
        
        print(f"Power Consumption from dense matrices multiplication of size {size}")
        driver = parsers.JsonParser(f"logs/dense_varying/{size}")
        exp_result = experiment.ExpResults(driver)
        exp_result.print()

        print(f"Power Consumption from sparse matrices multiplication of size {size}")
        driver = parsers.JsonParser(f"logs/sparse_varying/{size}")
        exp_result = experiment.ExpResults(driver)
        exp_result.print()

        print(f"Power Consumption from sparse matrices multiplication of size {size} with fixed density of 1/100")
        driver = parsers.JsonParser(f"logs/sparse_fixed/{size}")
        exp_result = experiment.ExpResults(driver)
        exp_result.print()


# Main execution
if __name__ == '__main__':

    sizes = [2**i for i in range(2,13)]

    # Execute the multiplication and save the results
    # You can chose to change the sizes and comment/uncomment the experiments if you want to logs specific tasks
    dense_varying(sizes)
    sparse_varying(sizes)
    sparse_fixed(sizes)

    # Print the results at the end
    print_results(sizes)