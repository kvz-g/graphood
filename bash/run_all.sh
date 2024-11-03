#!/bin/bash
#SBATCH --job-name=baseline           # 作业名称
#SBATCH --output=output.txt        # 输出日志的文件名
#SBATCH --time=06:00:00            # 执行时间限制为1小时
#SBATCH --nodes=1                  # 申请1个节点
#SBATCH --ntasks=1                 # 任务数为1
#SBATCH --cpus-per-task=8          # 每个任务使用2个 CPU 核心
#SBATCH --mem=16G                   # 每个任务使用16G内存
#SBATCH --partition=gpujl          # 队列名称为gpujl
#SBATCH --gres=gpu:1               # 如果需要，使用1个GPU

echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES # 查看自己被分配到了哪张显卡上。
### Cora with structure ood

python main.py --method msp --backbone gcn --dataset cora --ood_type structure --mode detect --use_bn --device 1
python main.py --method gnnsafe --backbone gcn --dataset cora --ood_type structure --mode detect --use_bn --device 1
python main.py --method gnnsafe --backbone gcn --dataset cora --ood_type structure --mode detect --use_bn --use_prop --device 1
python main.py --method ours --backbone gcn --dataset cora --ood_type structure --mode detect --use_bn --nc_dim 7 --use_prop --device 1
python main.py --method ours --backbone gcn --dataset cora --ood_type structure --mode detect --use_bn --nc_dim 7 --use_prop --scale --device 1

### Cora with feature ood

python main.py --method msp --backbone gcn --dataset cora --ood_type feature --mode detect --use_bn --device 1
python main.py --method gnnsafe --backbone gcn --dataset cora --ood_type feature --mode detect --use_bn --device 1
python main.py --method gnnsafe --backbone gcn --dataset cora --ood_type feature --mode detect --use_bn --use_prop --device 1
python main.py --method ours --backbone gcn --dataset cora --ood_type feature --mode detect --use_bn --nc_dim 7 --use_prop --device 1
python main.py --method ours --backbone gcn --dataset cora --ood_type feature --mode detect --use_bn --nc_dim 7 --use_prop --scale --device 1

### Cora with label ood

python main.py --method msp --backbone gcn --dataset cora --ood_type label --mode detect --use_bn --device 1
python main.py --method gnnsafe --backbone gcn --dataset cora --ood_type label --mode detect --use_bn --device 1
python main.py --method gnnsafe --backbone gcn --dataset cora --ood_type label --mode detect --use_bn --use_prop --device 1
python main.py --method ours --backbone gcn --dataset cora --ood_type label --mode detect --use_bn --nc_dim 7 --use_prop --device 1
python main.py --method ours --backbone gcn --dataset cora --ood_type label --mode detect --use_bn --nc_dim 7 --use_prop --scale --device 1


### Amazon-photo with structure ood

python main.py --method msp --backbone gcn --dataset amazon-photo --ood_type structure --mode detect --device 1
python main.py --method gnnsafe --backbone gcn --dataset amazon-photo --ood_type structure --mode detect --use_bn --device 1
python main.py --method gnnsafe --backbone gcn --dataset amazon-photo --ood_type structure --mode detect --use_bn --use_prop --device 1
python main.py --method ours --backbone gcn --dataset amazon-photo --ood_type structure --mode detect --use_bn --nc_dim 8 --use_prop --device 1
python main.py --method ours --backbone gcn --dataset amazon-photo --ood_type structure --mode detect --use_bn --nc_dim 8 --use_prop --scale --device 1


### Amazon-photo with feature ood

python main.py --method msp --backbone gcn --dataset amazon-photo --ood_type feature --mode detect --use_bn --device 1
python main.py --method gnnsafe --backbone gcn --dataset amazon-photo --ood_type feature --mode detect --use_bn --device 1
python main.py --method gnnsafe --backbone gcn --dataset amazon-photo --ood_type feature --mode detect --use_bn --use_prop --device 1
python main.py --method ours --backbone gcn --dataset amazon-photo --ood_type feature --mode detect --use_bn --nc_dim 8 --use_prop --device 1
python main.py --method ours --backbone gcn --dataset amazon-photo --ood_type feature --mode detect --use_bn --nc_dim 8 --use_prop --scale --device 1


### Amazon-photo with label ood

python main.py --method msp --backbone gcn --dataset amazon-photo --ood_type label --mode detect --device 1
python main.py --method gnnsafe --backbone gcn --dataset amazon-photo --ood_type label --mode detect --use_bn --device 1
python main.py --method gnnsafe --backbone gcn --dataset amazon-photo --ood_type label --mode detect --use_bn --use_prop --device 1
python main.py --method ours --backbone gcn --dataset amazon-photo --ood_type label --mode detect --use_bn --nc_dim 8 --use_prop --device 1
python main.py --method ours --backbone gcn --dataset amazon-photo --ood_type label --mode detect --use_bn --nc_dim 8 --use_prop --scale --device 1


### Coauthor with structure ood

python main.py --method msp --backbone gcn --dataset coauthor-cs --ood_type structure --mode detect --use_bn --device 1
python main.py --method gnnsafe --backbone gcn --dataset coauthor-cs --ood_type structure --mode detect --use_bn --device 1
python main.py --method gnnsafe --backbone gcn --dataset coauthor-cs --ood_type structure --mode detect --use_bn --use_prop --device 1
python main.py --method ours --backbone gcn --dataset coauthor-cs --ood_type structure --mode detect --use_bn --nc_dim 15 --use_prop --device 1
python main.py --method ours --backbone gcn --dataset coauthor-cs --ood_type structure --mode detect --use_bn --nc_dim 15 --use_prop --scale --device 1


### Coauthor with feature ood

python main.py --method msp --backbone gcn --dataset coauthor-cs --ood_type feature --mode detect --use_bn --device 1
python main.py --method gnnsafe --backbone gcn --dataset coauthor-cs --ood_type feature --mode detect --use_bn --device 1
python main.py --method gnnsafe --backbone gcn --dataset coauthor-cs --ood_type feature --mode detect --use_bn --use_prop --device 1
python main.py --method ours --backbone gcn --dataset coauthor-cs --ood_type feature --mode detect --use_bn --nc_dim 15 --use_prop --device 1
python main.py --method ours --backbone gcn --dataset coauthor-cs --ood_type feature --mode detect --use_bn --nc_dim 15 --use_prop --scale --device 1


### Coauthor with label ood

python main.py --method msp --backbone gcn --dataset coauthor-cs --ood_type label --mode detect --use_bn --device 1
python main.py --method gnnsafe --backbone gcn --dataset coauthor-cs --ood_type label --mode detect --use_bn --device 1
python main.py --method gnnsafe --backbone gcn --dataset coauthor-cs --ood_type label --mode detect --use_bn --use_prop --device 1
python main.py --method ours --backbone gcn --dataset coauthor-cs --ood_type label --mode detect --use_bn --nc_dim 15 --use_prop --device 1
python main.py --method ours --backbone gcn --dataset coauthor-cs --ood_type label --mode detect --use_bn --nc_dim 15 --use_prop --scale --device 1


### Twitch

python main.py --method msp --backbone gcn --dataset twitch --mode detect --use_bn --device 1
python main.py --method gnnsafe --backbone gcn --dataset twitch --mode detect --use_bn --device 1
python main.py --method gnnsafe --backbone gcn --dataset twitch --mode detect --use_bn --use_prop --device 1
python main.py --method ours --backbone gcn --dataset twitch --mode detect --use_bn --nc_dim 4 --use_prop --device 1
python main.py --method ours --backbone gcn --dataset twitch --mode detect --use_bn --nc_dim 4 --use_prop --scale --device 1

### Arxiv

python main.py --method msp --backbone gcn --dataset arxiv --mode detect --use_bn --device 1
python main.py --method gnnsafe --backbone gcn --dataset arxiv --mode detect --use_bn --device 1
python main.py --method gnnsafe --backbone gcn --dataset arxiv --mode detect --use_bn --use_prop --device 1
python main.py --method ours --backbone gcn --dataset arxiv --mode detect --use_bn --nc_dim 4 --use_prop --device 1
python main.py --method ours --backbone gcn --dataset arxiv --mode detect --use_bn --nc_dim 4 --use_prop --scale --device 1
echo "Job completed successfully."