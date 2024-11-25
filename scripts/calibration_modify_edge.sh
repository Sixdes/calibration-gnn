#!/bin/sh

# CoraFull
# Computers Photo CS Physics Cora Citeseer 
for dataset in Cora Citeseer Pubmed 

do for calibration in TS 
# VS ETS CaGCN GATS

do for model in GCN GAT
do 

case $dataset in
    Cora|Citeseer|Pubmed) wdecay=5e-4;;
    *)                    wdecay=0;;
esac

# PYTHONPATH=. python src/calibration_edge_struct.py --dataset $dataset \
#         --model $model \
#         --wdecay $wdecay \
#         --calibration $calibration \
#         --is_edge_delete \
# 	--config >> Log/modify_edge/cal_train_all_modify_edge/${dataset}_delete.log

PYTHONPATH=. python src/calibration_edge_struct.py --dataset $dataset \
        --model $model \
        --wdecay $wdecay \
        --calibration $calibration \
        --is_edge_add \
	--config >> Log/modify_edge/cal_train_all_modify_edge/${dataset}_add.log
done
done
done