### Cora with label ood

python main.py --method msp --backbone mixhop --dataset roman-empire --ood_type label --mode detect --use_bn --device 1
python main.py --method gnnsafe --backbone mixhop --dataset roman-empire --ood_type label --mode detect --use_bn --device 1
python main.py --method gnnsafe --backbone mixhop --dataset roman-empire --ood_type label --mode detect --use_bn --use_prop --device 1
python main.py --method ours --backbone mixhop --dataset roman-empire --ood_type label --mode detect --use_bn --nc_dim 15 --use_prop --device 1
python main.py --method ours --backbone mixhop --dataset roman-empire --ood_type label --mode detect --use_bn --nc_dim 15 --use_prop --scale --device 1

### 
python main.py --method msp --backbone mixhop --dataset amazon-ratings --ood_type label --mode detect --use_bn --device 1
python main.py --method gnnsafe --backbone mixhop --dataset amazon-ratings --ood_type label --mode detect --use_bn --device 1
python main.py --method gnnsafe --backbone mixhop --dataset amazon-ratings --ood_type label --mode detect --use_bn --use_prop --device 1
python main.py --method ours --backbone mixhop --dataset amazon-ratings --ood_type label --mode detect --use_bn --nc_dim 5 --use_prop --device 1
python main.py --method ours --backbone mixhop --dataset amazon-ratings --ood_type label --mode detect --use_bn --nc_dim 7 --use_prop --scale --device 1