#!/bin/bash
#SBATCH --job-name=hyper           # 作业名称
#SBATCH --output=hyper.txt        # 输出日志的文件名
#SBATCH --error=error.txt         #标准错误输出
#SBATCH --time=06:00:00            # 执行时间限制为1小时
#SBATCH --nodes=1                  # 申请1个节点
#SBATCH --cpus-per-task=8          # 每个任务使用2个 CPU 核心
#SBATCH --mem=16G                   # 每个任务使用16G内存
#SBATCH --partition=gpujl          # 队列名称为gpujl
#SBATCH --gres=gpu:1               # 如果需要，使用1个GPU

echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES # 查看自己被分配到了哪张显卡上。
gnns=('gcn' 'gat' 'mixhop' 'gcnjk' 'gatjk')
datasets=('cora' 'amazon-photo' 'coauthor-cs')
ood_types=('feature' 'label' 'structure')
nc_dims=(7 12 15 17 22 27 32 37 40 45 50)
dev=1
gnn='gcn'
data='cora'

for data in ${datasets[@]}
do
  for type in ${ood_types[@]}
  do
    for dim in ${nc_dims[@]}
    do
      python main.py --method ours --backbone $gnn --dataset $data --ood_type $type --mode detect --use_bn --nc_dim $dim --use_prop --device $dev
      python main.py --method ours --backbone $gnn --dataset $data --ood_type $type --mode detect --use_bn --nc_dim $dim --use_prop --scale --device $dev
    done
  done
done
