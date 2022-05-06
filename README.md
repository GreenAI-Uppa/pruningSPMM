# Pruning in Neural Network 
## Improving the computation performances using sparse matrix multiplications

This repository contains all you need to repeat the experiments in the article : [ADD LINK WHEN PUBLISHED]

It is separated in two parts :

- The matrices multiplication with pytorch.sparse

To repeat this experiment, all you need to do is to change the code in the two files gpu_torch.py and plot_results.txt where you want to store the logs, you need as well to install AIPowerMeter with :

`git clone https://github.com/GreenAI-Uppa/AIPowerMeter`

in the same repository as pruningSPMM.

Then you can install the requirements.txt with pip in a virtual env :

`pip -r install requirements.txt`

and then 

`python gpu_torch.py` to start the experiment

`python plot_results.py` to save the results aftewards as a plot on a file


- cuSPARSE :
