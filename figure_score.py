import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scienceplots

plt.style.use(['science'])
sns.set_theme(style="white",context="paper")

# load the Orthogonality Coefficient on mlp backbone
file_path = 'results/vis_scores/cora_nc_mlp.csv'
nc_cora_mlp = np.array([])
with open(file_path, 'r') as file:
    data = file.readlines()
    for line in data:
        nc_cora_mlp = np.append(nc_cora_mlp, np.array(float(line[:-2])))

file_path = 'results/vis_scores/amazon-computer_nc_mlp.csv'
nc_amazonc_mlp = np.array([])
with open(file_path, 'r') as file:
    data = file.readlines()
    for line in data:
        nc_amazonc_mlp = np.append(nc_amazonc_mlp, np.array(float(line[:-2])))

file_path = 'results/vis_scores/amazon-photo_nc_mlp.csv'
nc_amazonp_mlp = np.array([])
with open(file_path, 'r') as file:
    data = file.readlines()
    for line in data:
        nc_amazonp_mlp = np.append(nc_amazonp_mlp, np.array(float(line[:-2])))

file_path = 'results/vis_scores/coauthor-cs_nc_mlp.csv'
nc_coauthorc_mlp = np.array([])
with open(file_path, 'r') as file:
    data = file.readlines()
    for line in data:
        nc_coauthorc_mlp = np.append(nc_coauthorc_mlp, np.array(float(line[:-2])))

file_path = 'results/vis_scores/coauthor-physics_nc_mlp.csv'
nc_coauthorp_mlp = np.array([])
with open(file_path, 'r') as file:
    data = file.readlines()
    for line in data:
        nc_coauthorp_mlp = np.append(nc_coauthorp_mlp, np.array(float(line[:-2])))

# creat 1 x 2 canvas
fig, axes = plt.subplots(1, 2, figsize=(12.8, 6.4))

# plot (1, 1) subgraph
sns.lineplot(nc_cora_mlp / 7, ax=axes[0], label='ID(Cora)/OOD(Feature)')
sns.lineplot(nc_amazonc_mlp / 10, ax=axes[0], label='ID(Amazon-Computers)/OOD(Feature)')
sns.lineplot(nc_amazonp_mlp / 8, ax=axes[0], label='ID(Amazon-Photo)/OOD(Feature)')
sns.lineplot(nc_coauthorc_mlp / 15, ax=axes[0], label='ID(Coauthor-CS)/OOD(Feature)')
sns.lineplot(nc_coauthorp_mlp / 5, ax=axes[0], label='ID(Coauthor-Physics)/OOD(Feature)')
axes[0].tick_params(axis='both', which='major', labelsize=12)
axes[0].set_title('Backbone: MLP', fontsize=23)
axes[0].set_xlabel('Epoch', fontsize=23)
axes[0].set_ylabel('Orthogonality Coefficient', fontsize=23)
axes[0].legend(fontsize=14)
axes[0].grid(True, linestyle='--')

#load the Orthogonality Coefficient on GCN backbone
file_path = 'results/vis_scores/coranc.csv'
nc_cora_gcn = np.array([])
with open(file_path, 'r') as file:
    data = file.readlines()
    for line in data:
        nc_cora_gcn = np.append(nc_cora_gcn, np.array(float(line[:-2])))

file_path = 'results/vis_scores/amazon-computer_nc_gcn.csv'
nc_amazonc_gcn = np.array([])
with open(file_path, 'r') as file:
    data = file.readlines()
    for line in data:
        nc_amazonc_gcn = np.append(nc_amazonc_gcn, np.array(float(line[:-2])))

file_path = 'results/vis_scores/amazon-photo_nc_gcn.csv'
nc_amazonp_gcn = np.array([])
with open(file_path, 'r') as file:
    data = file.readlines()
    for line in data:
        nc_amazonp_gcn = np.append(nc_amazonp_gcn, np.array(float(line[:-2])))

file_path = 'results/vis_scores/coauthor-cs_nc_gcn.csv'
nc_coauthorc_gcn = np.array([])
with open(file_path, 'r') as file:
    data = file.readlines()
    for line in data:
        nc_coauthorc_gcn = np.append(nc_coauthorc_gcn, np.array(float(line[:-2])))

file_path = 'results/vis_scores/coauthor-physics_nc_gcn.csv'
nc_coauthorp_gcn = np.array([])
with open(file_path, 'r') as file:
    data = file.readlines()
    for line in data:
        nc_coauthorp_gcn = np.append(nc_coauthorp_gcn, np.array(float(line[:-2])))

# plot (1, 2) subgraph
sns.lineplot(nc_cora_gcn / 7, ax=axes[1], label='ID(Cora)/OOD(Feature)')
sns.lineplot(nc_amazonc_gcn / 10, ax=axes[1], label='ID(Amazon-Computers)/OOD(Feature)')
sns.lineplot(nc_amazonp_gcn / 8, ax=axes[1], label='ID(Amazon-Photo)/OOD(Feature)')
sns.lineplot(nc_coauthorc_gcn / 15, ax=axes[1], label='ID(Coauthor-CS)/OOD(Feature)')
sns.lineplot(nc_coauthorp_gcn / 5, ax=axes[1], label='ID(Coauthor-Physics)/OOD(Feature)')
axes[1].tick_params(axis='both', which='major', labelsize=12)
axes[1].set_title('Backbone: GCN', fontsize=23)
axes[1].set_xlabel('Epoch', fontsize=23)
axes[1].set_ylabel('Orthogonality Coefficient', fontsize=23)
axes[1].legend(fontsize=14)
plt.grid(True, linestyle='--')
plt.tight_layout()
plt.show()
