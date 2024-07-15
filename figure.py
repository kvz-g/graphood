import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import scienceplots

plt.style.use(['science'])

# load data
in_distribution = np.random.normal(loc=-0.7, scale=0.05, size=100)
out_of_distribution = np.random.normal(loc=-0.65, scale=0.05, size=100)

# mean of the ood score
mean_in = np.mean(in_distribution)
mean_out = np.mean(out_of_distribution)

# 创建一个3x1的子图布局
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# first subgraph
sns.kdeplot(in_distribution, fill=True, color="green", ax=axes[0], label='in-distribution')
sns.kdeplot(out_of_distribution, fill=True, color="red", ax=axes[0], label='out-of-distribution')
axes[0].axvline(mean_in, color='green', linestyle='--')
axes[0].axvline(mean_out, color='red', linestyle='--')
axes[0].set_title('GNNSafe w/o energy propagation')
axes[0].set_xlabel('Energy score')
axes[0].set_ylabel('Frequency')
axes[0].legend()

# secend subgrah
sns.kdeplot(in_distribution, fill=True, color="green", ax=axes[1], label='in-distribution')
sns.kdeplot(out_of_distribution, fill=True, color="red", ax=axes[1], label='out-of-distribution')
axes[1].axvline(mean_in, color='green', linestyle='--')
axes[1].axvline(mean_out, color='red', linestyle='--')
axes[1].set_title('GNNSafe')
axes[1].set_xlabel('Energy score')
axes[1].set_ylabel('Frequency')
axes[1].legend()

file_path = 'results/vis_scores/coranc.csv'
nc = np.array([])
with open(file_path, 'r') as file:
    data = file.readlines()
    for line in data:
        nc = np.append(nc, np.array(float(line[:-2])))

# mean

# Third subgraph
sns.lineplot(nc, color='green', ax=axes[2], label='nc_oth')
axes[2].set_title('GNNSafe++')
axes[2].set_xlabel('Energy score')
axes[2].set_ylabel('Frequency')
axes[2].legend()

plt.tight_layout()
plt.show()
