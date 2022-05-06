"""
Author: Nicolas Tirel
Organization: GreenAI UPPA
Date: 05/09/2022
Description: This code is used to plot the results after running gpu_torch.py
The dense_time and sparse_time are copied from the results on the terminal from the previous experiments 
"""


from AIPowerMeter.deep_learning_power_measure.power_measure import experiment, parsers
import pandas as pd
import matplotlib.pyplot as plt

sizes = [2**i for i in range(2,13)]
dense = []
sparse = []
dense_time = [5.188,4.704,4.855,4.858,4.882,6.077,7.457,12.932,78.153,600.904,4010.097]
sparse_time = [128.377,127.922,129.095,129.968,131.497,132.283,138.695,162.529,226.504,409.913,1139.658]
fixed = []
for i,size in enumerate(sizes):
    driver = parsers.JsonParser(f"logs/dense_varying/{size}")
    exp_result = experiment.ExpResults(driver)
    rel_nvidia_power = exp_result.total_('nvidia_attributable_power')
    dense.append(rel_nvidia_power)
    driver = parsers.JsonParser(f"logs/sparse_varying/{size}")
    exp_result = experiment.ExpResults(driver)
    rel_nvidia_power = exp_result.total_('nvidia_attributable_power')
    sparse.append(rel_nvidia_power)
    driver = parsers.JsonParser(f"logs/sparse_fixed/{size}")
    exp_result = experiment.ExpResults(driver)
    rel_nvidia_power = exp_result.total_('nvidia_attributable_power')
    fixed.append(rel_nvidia_power)


data = {"sizes" : sizes, "dense" : dense, "dense_time" : dense_time, "sparsee" : sparse, "sparse_time" : sparse_time, "fixed" : fixed}
df_results = pd.DataFrame(data)
print(df_results)
  
#define colors to use
col11 = 'lightcoral'
col12 = 'cornflowerblue'
col21 = 'orangered'
col22 = 'navy'

#define subplots
fig,ax = plt.subplots()
#add first line to plot
ax.plot(df_results.sizes, df_results.dense, color=col21, marker='o', linewidth=1)
ax.plot(df_results.sizes, df_results["sparsee"], color=col11, marker='o', linewidth=1)

#add x-axis label
ax.set_xlabel('Sizes', fontsize=14)

#add y-axis label
ax.set_ylabel('Consumption (joules)', color="blue", fontsize=16)

#define second y-axis that shares x-axis with current plot
ax2 = ax.twinx()

#add second line to plot
ax2.plot(df_results.sizes, df_results.dense_time, color=col22, marker='o', linewidth=1)
ax2.plot(df_results.sizes, df_results.sparse_time, color=col12, marker='o', linewidth=1)

#add second y-axis label
ax2.set_ylabel('Time (seconds)', color="red", fontsize=16)

plt.semilogx(base=2)

# set the title
plt.title('Dense vs Sparse')
plt.grid()
  
# Save the plot
#plt.show()
plt.savefig("comparison.png", bbox_inches='tight')



# Another graph but zoomed on a specific point

df_results = df_results[df_results["sizes"] >= 512]

#define subplots
fig,ax = plt.subplots()
#add first line to plot
ax.plot(df_results.sizes, df_results.dense, color=col21, marker='o', linewidth=1)
ax.plot(df_results.sizes, df_results.sparsee, color=col11, marker='o', linewidth=1)

#add x-axis label
ax.set_xlabel('Sizes', fontsize=14)

#add y-axis label
ax.set_ylabel('Consumption (joules)', color="blue", fontsize=16)

#define second y-axis that shares x-axis with current plot
ax2 = ax.twinx()

#add second line to plot
ax2.plot(df_results.sizes, df_results.dense_time, color=col22, marker='o', linewidth=1)
ax2.plot(df_results.sizes, df_results.sparse_time, color=col12, marker='o', linewidth=1)

#add second y-axis label
ax2.set_ylabel('Time (seconds)', color="red", fontsize=16)

plt.semilogx(base=2)

# set the title
plt.title('Dense vs Sparse')
plt.grid()
  
# Save the plot
plt.savefig("comparisonZoom.png", bbox_inches='tight')