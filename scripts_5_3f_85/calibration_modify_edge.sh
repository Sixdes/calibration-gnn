#!/bin/sh

# CoraFull
# Computers Photo CS Physics 
for dataset in Cora Citeseer Pubmed 

do for calibration in TS 
# VS ETS CaGCN GATS

do for model in GCN
do 

case $dataset in
    Cora|Citeseer|Pubmed) wdecay=5e-4;;
    *)                    wdecay=0;;
esac

PYTHONPATH=. python src/5_3f_85_setting/calibration_edge_struct_dcgc.py --dataset $dataset \
        --model $model \
        --wdecay $wdecay \
        --calibration $calibration \
        --is_edge_delete \
	--config >> Log_5_3f_85/modify_edge/cal_train_all_modify_edge_dcgc/${dataset}_delete.log

PYTHONPATH=. python src/5_3f_85_setting/calibration_edge_struct_dcgc.py --dataset $dataset \
        --model $model \
        --wdecay $wdecay \
        --calibration $calibration \
        --is_edge_add \
	--config >> Log_5_3f_85/modify_edge/cal_train_all_modify_edge_dcgc/${dataset}_add.log
done
done
done