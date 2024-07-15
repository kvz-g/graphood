import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scienceplots

plt.style.use(['science'])
sns.set_theme(style="white",context="paper")
# 生成示例数据
np.random.seed(42)
in_distribution = np.random.normal(loc=-0.7, scale=0.05, size=100)
out_of_distribution = np.random.normal(loc=-0.65, scale=0.05, size=100)

# 计算均值
mean_in = np.mean(in_distribution)
mean_out = np.mean(out_of_distribution)

# 创建一个3x1的子图布局
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# 绘制第一个子图
sns.kdeplot(in_distribution, shade=True, ax=axes[0], label='in-distribution')
sns.kdeplot(out_of_distribution, shade=True,ax=axes[0], label='out-of-distribution')
axes[0].axvline(mean_in, linestyle='--')
axes[0].axvline(mean_out,linestyle='--')
axes[0].set_title('GNNSafe w/o energy propagation')
axes[0].set_xlabel('Energy score')
axes[0].set_ylabel('Frequency')
axes[0].legend()

# 绘制第二个子图
sns.kdeplot(in_distribution, shade=True, ax=axes[1], label='in-distribution')
sns.kdeplot(out_of_distribution,shade=True,ax=axes[1], label='out-of-distribution')
axes[1].axvline(mean_in,  linestyle='--')
axes[1].axvline(mean_out,linestyle='--')
axes[1].set_title('GNNSafe')
axes[1].set_xlabel('Energy score')
axes[1].set_ylabel('Frequency')
axes[1].legend()

in_distribution2 = np.random.normal(loc=-6, scale=0.5, size=100)
out_of_distribution2 = np.random.normal(loc=-2, scale=0.5, size=100)

# 计算均值
mean_in2 = np.mean(in_distribution2)
mean_out2 = np.mean(out_of_distribution2)

# 绘制第三个子图
sns.kdeplot(in_distribution2, shade=True, ax=axes[2], label='in-distribution')
sns.kdeplot(out_of_distribution2,shade=True,ax=axes[2], label='out-of-distribution')
axes[2].axvline(mean_in2,  linestyle='--')
axes[2].axvline(mean_out2,linestyle='--')
axes[2].set_title('GNNSafe++')
axes[2].set_xlabel('Energy score')
axes[2].set_ylabel('Frequency')
axes[2].legend()

plt.tight_layout()
plt.show()
