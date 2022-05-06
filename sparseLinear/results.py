"""
Author: Yanis Chaigneau
Organization: GreenAI UPPA
Date: 05/04/2022
Description: This code load the experiment results of the test_snip_sparse.py file.

Outputs:
The energy consumption of the experiment

Usage:
Simply launch the code with python in a terminal.

"""

from deep_learning_power_measure.power_measure import experiment, parsers

size_models = [1000, 3000, 5000, 7500, 10000, 12500, 15000]
pruning_levels = [0.05]


dict_gpu = {}
for pruning_level in pruning_levels:
	dict_sparse = {}
	for size_model in size_models:
		print('Sparse: Size model: '+str(size_model)+', Pruning level: '+str(pruning_level))
		driver = parsers.JsonParser('output_folder/results_sparse_'+str(pruning_level)+'_'+str(size_model))
		exp_result = experiment.ExpResults(driver)
		exp_result.print()
		print()
		dict_sparse[size_model] = exp_result.total_('nvidia_draw_absolute')
	dict_gpu[pruning_level]= dict_sparse

dict_dense = {}
for size_model in size_models:
	print('Dense: Size model: '+str(size_model))
	driver = parsers.JsonParser('output_folder/results_dense_'+str(size_model))
	exp_result = experiment.ExpResults(driver)
	exp_result.print()
	dict_dense[size_model] = exp_result.total_('nvidia_draw_absolute')
	print()
dict_gpu[1] = dict_dense

print(dict_gpu)
