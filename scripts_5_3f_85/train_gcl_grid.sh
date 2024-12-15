#!/bin/sh

#CoraFull
for dataset in Cora Citeseer Pubmed 
# Computers Photo CS Physics 

do for model in GCN GAT

do

case $dataset in
    Cora|Citeseer|Pubmed) wdecay=5e-4;;
    *)                    wdecay=0;;
esac

PYTHONPATH=. python src/train_gcl_grid.py --dataset $dataset\
        --model $model \
        --wdecay $wdecay >> Log/train_gcl_grid-1117.log
done
done
# --gamma_list 0.01 
# --verbose >> Log_train/train_gcl_grid.log
