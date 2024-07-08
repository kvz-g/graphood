import matplotlib.pyplot as plt
import numpy as np

# 假设 output 是你之前生成的二维数组
#output1 = np.loadtxt('datapoint2/rep_only_mulde_s2_m.txt')
output1 = np.loadtxt('datapoint/true_output_vani_m.txt')
output2 = np.loadtxt('datapoint/true_output_our_m.txt')

# 创建一个从0到1，每隔0.05的bins
bins = np.arange(-1, 1.01, 0.01)

# 使用matplotlib的hist函数来生成两个条形图
plt.hist(output1, bins=bins, color=(0/255.,60/255.,255/255.), alpha=0.5, label='Vanilla')
plt.hist(output2, bins=bins, color=(255/255.,20/255.,20/255.), alpha=0.5, label='Ours')
#plt.hist(output1, bins=bins, color='blue', alpha=0.4, label='vanila')
#plt.hist(output2, bins=bins, color='red', alpha=0.5, label='ours')

# 在x=0处画一条红线
plt.axvline(x=0, color='black', linewidth=2.5)

# 在x=0处画一条红线
plt.axvline(x=np.mean(output1), color=(70/255.,70/255.,255/255.), linewidth=1.5)
# 在x=0处画一条红线
plt.axvline(x=np.mean(output2), color=(255/255.,50/255.,50/255.), linewidth=1.5)

plt.ylim(0,440)

# 设置图表的标题和坐标轴标签
#plt.title('Distribution of Output')
plt.xlabel(r'$\Delta p$', fontsize=32)
plt.ylabel('Density', fontsize=32)

# 放大x轴上的数字
plt.tick_params(axis='x', labelsize=26)

# 放大y轴上的数字
plt.tick_params(axis='y', labelsize=26)

# 添加图例
plt.legend(loc='upper right', fontsize=32)

# 这里设置y轴的间隔
plt.yticks(np.arange(0, 441, 50))

# 显示图表
plt.show()