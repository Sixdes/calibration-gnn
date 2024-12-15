#!/bin/sh

# PYTHONPATH=. python src/calibration_rbs.py --model GCN --dataset Pubmed --num_bins_rbs 2

# Cora Citeseer Pubmed Computers Photo Physics
for dataset in  Cora Citeseer Pubmed 

do for model in GCN


do 

case $dataset in
    Cora|Citeseer|Pubmed) wdecay=5e-4;;
    *)                    wdecay=0;;
esac

# PYTHONPATH=. python src/5_3f_85_setting/calibration_dcgc.py --dataset $dataset \
#         --model $model \
#         --wdecay $wdecay \
#         --calibration TS >> Log_5_3f_85/calibration_dcgc/1212/${dataset}_${model}.log
PYTHONPATH=. python src/5_3f_85_setting/calibration_dcgc.py --dataset $dataset \
        --model $model \
        --wdecay $wdecay \
        --calibration TS >> Log_5_3f_85/calibration_hd16/${dataset}_${model}_hd64_op_drop.log
done
done