"""
Author: Yanis Chaigneau
Organization: GreenAI UPPA
Date: 05/04/2022
Description: This code is inspired by the tutorial of the sparseLinear library: 
https://github.com/hyeon95y/SparseLinear/blob/master/tutorials/SparseLinearDemo.ipynb

Energy consumption monitoring of the training of sparse pruned models and dense equivalent.

Outputs:
The energy consumption files

Usage:
Simply launch the code with python in a terminal.

Modify the script to evaluate more model sizes and pruning levels.
"""

import time
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import sampler

import sparselinear as sl
import torch
import torch.nn as nn
import numpy as np

import warnings

from snip import *

from deep_learning_power_measure.power_measure import experiment, parsers
warnings.filterwarnings('ignore')



def train_model(model, optimizer, criterion, train_dataloader, test_dataloader, num_epochs=5):
    """
    Simple training function
    """
    since = time.time()
    for epoch in range(num_epochs):
        cum_loss, total, correct = 0, 0, 0
        model.train()
        
        # Training epoch
        for i, (images, labels) in enumerate(train_dataloader, 0):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass & statistics
            out = model(images)
            predicted = out.argmax(dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            loss = criterion(out, labels)
            cum_loss += loss.item()

            # Backwards pass & update
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
        epoch_loss = images.shape[0] * cum_loss / total
        epoch_acc = 100 * (correct / total)
        print('Epoch %d' % (epoch + 1))
        print('Training Loss: {:.4f}; Training Acc: {:.4f}'.format(epoch_loss, epoch_acc))
        
        cum_loss, total, correct = 0, 0, 0
        model.eval()
        
        # Test epoch
        for i, (images, labels) in enumerate(test_dataloader, 0):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass & statistics
            out = model(images)
            predicted = out.argmax(dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            loss = criterion(out, labels)
            cum_loss += loss.item()
            
        epoch_loss = images.shape[0] * cum_loss / total
        epoch_acc = 100 * (correct / total)
        
        print('Test Loss: {:.4f}; Test Acc: {:.4f}'.format(epoch_loss, epoch_acc))
        print('------------')
    
    time_elapsed = time.time() - since
    print('\nTraining complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

def flatten(x):
	N = x.shape[0]
	return x.view(N, -1)

class Flatten(nn.Module):
    def forward(self, x):
        return flatten(x)



# Load the GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Import the MNIST DATASET
tf = transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize((0.1307,), (0.3081,))])

batch_size = 64
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=tf)
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=tf)
train_dataloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, drop_last=True)
test_dataloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, drop_last=True)
criterion = nn.CrossEntropyLoss()


# All the model sizes to study
size_models = [1000, 3000, 5000, 7500, 10000, 12500, 15000]
pruning_levels=[0.01]

# Main experiment
for size_model in size_models:
	print('Studying with size_model='+str(size_model))
    
    # Creating the dense model
	model_b = nn.Sequential(
        Flatten(),
        nn.Linear(784, size_model),
        nn.LayerNorm(size_model),
        nn.ReLU(),
        nn.Linear(size_model, size_model),
        nn.LayerNorm(size_model),
        nn.ReLU(),
        nn.Linear(size_model,10),
)
	model_b = model_b.to(device)

    
	# For all the pruning levels specified
	for pruning_level in pruning_levels:
        
        # Launch the energy consumption monitoring
		driver = parsers.JsonParser("output_folder/results_sparse_"+str(pruning_level)+"_"+str(size_model))
		exp = experiment.Experiment(driver)
		p, q = exp.measure_yourself(period=2)

        # Prune the dense model
		print("Applying SNIP for pruning level: "+str(pruning_level))
		mask = SNIP(model_b, pruning_level, train_dataloader,device)
	
        # Get the indices of the non-zeros weights
		print("Generate model")
		connections_1 = mask[0].to_sparse().indices()
		connections_2 = mask[1].to_sparse().indices()
		connections_3 = mask[2].to_sparse().indices()
        
        # Create the sparse pruned model with the connections obtained with SNIP
		sparse_model_user = nn.Sequential(
        Flatten(),
        sl.SparseLinear(784, size_model, connectivity=connections_1),
        nn.LayerNorm(size_model),
        nn.ReLU(),
        sl.SparseLinear(size_model,size_model,connectivity=connections_2),
        nn.LayerNorm(size_model),
        nn.ReLU(),
       	sl.SparseLinear(size_model, 10, connectivity=connections_3)
	)
		sparse_model_user = sparse_model_user.to(device)
	
        # Train the pruned model
		print("Training pruned model")
		learning_rate = 1e-1
		optimizer = optim.SGD(sparse_model_user.parameters(), lr=learning_rate, momentum=0.9)
		train_model(sparse_model_user, optimizer, criterion, train_dataloader, test_dataloader)

		q.put(experiment.STOP_MESSAGE)


	print("Training model dense")
	
    ## Launch the energy consumption monitoring
	driver = parsers.JsonParser("output_folder/results_dense_"+str(size_model))
	exp = experiment.Experiment(driver)
	p, q = exp.measure_yourself(period=2)

    ## Define the dense model (with all the weights)
	model = nn.Sequential(
        Flatten(),
        nn.Linear(784, size_model),
        nn.LayerNorm(size_model),
        nn.ReLU(),
        nn.Linear(size_model, size_model),
        nn.LayerNorm(size_model),
        nn.ReLU(),
        nn.Linear(size_model,10),
)
	model = model.to(device)

    # Train the dense model
	optimizer_2 = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
	train_model(model, optimizer_2, criterion, train_dataloader, test_dataloader)

	q.put(experiment.STOP_MESSAGE)
