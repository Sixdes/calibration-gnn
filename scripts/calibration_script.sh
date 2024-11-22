#!/bin/sh

# for dataset in Cora Citeseer Pubmed Computers Photo CS Physics CoraFull
# # for dataset in Physics

# do for calibration in TS VS ETS CaGCN GATS
# # do for calibration in TS 



# do 

# case $dataset in
#     Cora|Citeseer|Pubmed) wdecay=5e-4;;
#     *)                    wdecay=0;;
# esac

# CUDA_VISIBLE_DEVICES=0 nohup bash -c "PYTHONPATH=. python src/calibration.py --dataset $dataset \
#         --model GCN \
#         --wdecay $wdecay \
#         --calibration $calibration \
# 	--config" > Log/${dataset}_GCN_${calibration}.log 2>&1 &
# CUDA_VISIBLE_DEVICES=1 nohup bash -c "PYTHONPATH=. python src/calibration.py --dataset $dataset \
#         --model GAT \
#         --wdecay $wdecay \
#         --calibration $calibration \
# 	--config" > Log/${dataset}_GAT_${calibration}.log 2>&1 &

# done
# done

for dataset in Cora Citeseer Pubmed Computers Photo CS Physics 
#Cora Citeseer Pubmed 
# do for calibration in TS VS ETS CaGCN GATS

do for model in GCN GAT

do 

case $dataset in
    Cora|Citeseer|Pubmed) wdecay=5e-4;;
    *)                    wdecay=0;;
esac


PYTHONPATH=. python src/calibration.py --dataset $dataset \
        --model $model \
        --wdecay $wdecay \
        --calibration TS \
	--config > Log/train_with_cal/logit_norm_t1.5/${dataset}_${model}_TS.log
done
done

