from AIPowerMeter.deep_learning_power_measure.power_measure import experiment, parsers
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt

sizes = [2**i for i in range(2,13)]
dense = []
sparse = []
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


data = {"size" : sizes, "dense" : dense, "sparse" : sparse, "fixed" : fixed}
df_results = pd.DataFrame(data)
print(df_results)

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

# scatter plot
"""
df_results.plot(kind = 'line',
        x = 'size',
        y = 'dense',
        logx = True,
        color = 'green')

df_results.plot(kind = 'line',
        x = 'size',
        logx = True,
        y = 'sparse',
        color = 'blue')

df_results.plot(kind = 'line',
        x = 'size',
        logx = True,
        y = 'fixed',
        color = 'black')"""
  
df_results[["size", "dense", "fixed"]].plot(x="size")
plt.semilogx(base=2)
# set the title
plt.title('Dense vs Sparse')
plt.xlabel('Size (n) of nxn matrix', fontsize=16)
plt.ylabel('Consumption (joules)', fontsize=16)
plt.grid()
  
# show the plot
#plt.show()
plt.savefig("fig_1.png")
#print(df_results[df_results["size"] <= 4096])
df_results[df_results["size"] < 512][["size", "dense", "fixed"]].plot(x="size")
plt.semilogx(base=2)
# set the title
plt.title('Dense vs Sparse')
plt.xlabel('Size (n) of nxn matrix', fontsize=16)
plt.ylabel('Consumption (joules)', fontsize=16)
plt.grid()
plt.savefig("fig_2.png")