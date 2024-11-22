#!/bin/sh

#CoraFull
# for dataset in Cora Citeseer Pubmed Computers Photo CS Physics 

# do for model in GCN GAT

# do

# case $dataset in
#     Cora|Citeseer|Pubmed) wdecay=5e-4;;
#     *)                    wdecay=0;;
# esac

# PYTHONPATH=. python src/train_crl.py --dataset $dataset\
#         --model $model \
#         --wdecay $wdecay --verbose 
# done
# done

PYTHONPATH=. python src/train_crl.py --model GCN --dataset Citeseer  --alpha 0.98 --wdecay 5e-4 --verbose 
PYTHONPATH=. python src/train_crl.py --model GAT --dataset Citeseer  --alpha 0.98 --wdecay 5e-4 --verbose 

PYTHONPATH=. python src/train_crl.py --model GCN --dataset Cora  --alpha 0.98 --wdecay 5e-4 --verbose 
PYTHONPATH=. python src/train_crl.py --model GAT --dataset Cora  --alpha 0.98 --wdecay 5e-4 --verbose 

PYTHONPATH=. python src/train_crl.py --dataset Pubmed --model GCN --wdecay 5e-4 --verbose --alpha 0.98
PYTHONPATH=. python src/train_crl.py --dataset Pubmed --model GAT --wdecay 5e-4 --verbose --alpha 0.98

PYTHONPATH=. python src/train_crl.py --dataset CS --model GCN --wdecay 0 --verbose --alpha 0.98
PYTHONPATH=. python src/train_crl.py --dataset CS --model GAT --wdecay 0 --verbose --alpha 0.98

PYTHONPATH=. python src/train_crl.py --dataset Photo --model GCN --wdecay 0 --verbose --alpha 0.98
PYTHONPATH=. python src/train_crl.py --dataset Photo --model GAT --wdecay 0 --verbose --alpha 0.98

PYTHONPATH=. python src/train_crl.py --dataset Computers --model GCN --wdecay 0 --verbose --alpha 0.98
PYTHONPATH=. python src/train_crl.py --dataset Computers --model GAT --wdecay 0 --verbose --alpha 0.98

PYTHONPATH=. python src/train_crl.py --dataset Computers --model GCN --wdecay 0 --verbose --alpha 0.98
PYTHONPATH=. python src/train_crl.py --dataset Computers --model GAT --wdecay 0 --verbose --alpha 0.98