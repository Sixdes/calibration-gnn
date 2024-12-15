#!/bin/sh

# CoraFull
# Computers Photo CS Physics Cora Citeseer 
for rate in 20 40 60
do for dataset in Cora Citeseer Pubmed 

do for calibration in TS 
# VS ETS CaGCN GATS

do for model in GCN
do 

case $dataset in
    Cora|Citeseer|Pubmed) wdecay=5e-4;;
    *)                    wdecay=0;;
esac

PYTHONPATH=. python src/labelrate_setting/calibration_edge_struct.py --dataset $dataset \
        --model $model \
        --wdecay $wdecay \
        --calibration $calibration \
        --is_edge_delete \
        --labelrate $rate \
	--config >> Log_labelrate/modify_edge/cal_train_all_modify_edge_dcgc/${dataset}_labelrate${rate}_delete.log

PYTHONPATH=. python src/labelrate_setting/calibration_edge_struct.py --dataset $dataset \
        --model $model \
        --wdecay $wdecay \
        --calibration $calibration \
        --is_edge_add \
        --labelrate $rate \
	--config >> Log_labelrate/modify_edge/cal_train_all_modify_edge_dcgc/${dataset}_labelrate${rate}_add.log
done
done
done
done