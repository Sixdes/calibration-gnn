#!/bin/sh

# PYTHONPATH=. python src/calibration_rbs.py --model GCN --dataset Pubmed --num_bins_rbs 2

# Cora Citeseer Pubmed Computers Photo Physics
for rate in 20 40 60
# do for alpha in 0.1 0.3 0.5 1
do for alpha in 0.5 
do for beta in 10
do for dataset in  Cora Citeseer Pubmed 

do for model in GCN


do 

case $dataset in
    Cora|Citeseer|Pubmed) wdecay=5e-4;;
    *)                    wdecay=0;;
esac

PYTHONPATH=. python src/labelrate_setting/calibration_dcgc.py --dataset $dataset \
        --model $model \
        --wdecay $wdecay \
        --labelrate $rate \
        --dcgc_alpha $alpha \
        --dcgc_beta $beta \
        --calibration TS >> Log_labelrate/calibration_hd16_op_drop/labelrate_${rate}/${dataset}_${model}_DCGC.log
done
done
done
done 
done