import torch
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from AIPowerMeter.deep_learning_power_measure.power_measure import experiment, parsers
import os
import time

log_file = 'wm_log'

def dense_varying(sizes = [2**i for i in range(2,13)]):
    for i,size in enumerate(sizes):
        iters = int(1000000)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        densemat = torch.randn(size,size).to(device)
        multmat = torch.randn(size,size).to(device)

        dir_name = f"logs/dense_varying/{size}"
        #driver = parsers.JsonParser(dir_name)
        #exp = experiment.Experiment(driver)

        #p, q = exp.measure_yourself(period=2) # measure every 2 seconds
        start_xp = time.time()
        for _ in range(iters):
            torch.mm(densemat,multmat)
        print(round(time.time()-start_xp,3))
        
        #q.put(experiment.STOP_MESSAGE)


def sparse_varying(sizes = [2**i for i in range(2,13)]):
    for i,size in enumerate(sizes):
        iters = int(1000000)
        #print('Multiplying matrices of size = {0}x{1}'.format(size,size))
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        spmat = torch.sparse_coo_tensor(indices = torch.tensor([torch.arange(size).tolist(),torch.arange(size).tolist()]), values = torch.randn(size), size=[size,size], dtype=torch.float32, device=device)
        multmat = torch.randn(size,size).to(device)

        dir_name = f"logs/sparse_varying/{size}"
        driver = parsers.JsonParser(dir_name)
        #exp = experiment.Experiment(driver)

        #p, q = exp.measure_yourself(period=2) # measure every 2 seconds
        start_xp = time.time()
        for _ in range(iters):
            torch.sparse.mm(spmat,multmat)
        print(round(time.time()-start_xp,3))
        #q.put(experiment.STOP_MESSAGE)

def sparse_fixed(sizes = [2**i for i in range(2,13)]):
    for i,size in enumerate(sizes):
        iters = int(1000000)
        print('Multiplying matrices of size = {0}x{1}'.format(size,size))
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        nnz = size*size//100
        spmat = torch.sparse_coo_tensor(indices = torch.tensor([torch.randint(size,(nnz,)).tolist(),torch.randint(size,(nnz,)).tolist()]), values = torch.randn(nnz), size=[size,size], dtype=torch.float32, device=device)
        multmat = torch.randn(size,size).to(device)

        dir_name = f"logs/sparse_fixed/{size}"
        driver = parsers.JsonParser(dir_name)
        exp = experiment.Experiment(driver)

        p, q = exp.measure_yourself(period=2) # measure every 2 seconds
        for _ in range(iters):
            torch.sparse.mm(spmat,multmat)
        q.put(experiment.STOP_MESSAGE)

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

        print(f"Power Consumption from sparse matrices multiplication of size {size}")
        driver = parsers.JsonParser(f"logs/sparse_fixed/{size}")
        exp_result = experiment.ExpResults(driver)
        exp_result.print()


# Main execution
if __name__ == '__main__':
    sizes = [2**i for i in range(2,13)]
    #subprocess.call(['ssh', 'ntirel@10.0.12.102', '/home/ntirel/Documents/wattmetre-read', '--tty=/dev/ttyUSB0', '--nb=6', '>', log_file, '2>&1', '&', 'echo', '$!', '>', 'wm_pid'])
    #dense_varying(sizes)
    #print("------------------------")
    sparse_varying(sizes)
    #sparse_fixed(sizes)

    #print_results(sizes)

    #subprocess.call(['ssh', 'ntirel@10.0.12.102', 'kill', '-10', '`cat', 'wm_pid`'])
    #subprocess.call(['scp', f'ntirel@10.0.12.102:{log_file}', f'/data/ntirel/sparsity/{dir_name}{log_file}'])