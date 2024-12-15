#!/bin/sh

# PYTHONPATH=. python src/calibration_rbs.py --model GCN --dataset Pubmed --num_bins_rbs 2

# Cora Citeseer Pubmed Computers Photo
for dataset in  CS Physics 

do for model in GCN GAT

do for num_bins in 2 3 4 6 8

do 

case $dataset in
    Cora|Citeseer|Pubmed) wdecay=5e-4;;
    *)                    wdecay=0;;
esac

PYTHONPATH=. python src/calibration_rbs.py --dataset $dataset \
        --model $model \
        --wdecay $wdecay \
        --num_bins_rbs $num_bins >> Log_1111/${dataset}_${model}_RBS.log
done
done
done