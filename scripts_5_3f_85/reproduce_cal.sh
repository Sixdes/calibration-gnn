#!/bin/sh

# CoraFull
# Cora Citeseer
for dataset in Pubmed Computers Photo CS Physics 

do for calibration in TS VS ETS CaGCN GATS

do for model in GCN GAT

do 

case $dataset in
    Cora|Citeseer|Pubmed) wdecay=5e-4;;
    *)                    wdecay=0;;
esac

PYTHONPATH=. python src/calibration.py --dataset $dataset \
        --model $model \
        --wdecay $wdecay \
        --calibration $calibration \
	--config > Log_1111/${dataset}_${model}_${calibration}.log
done
done
done

# PYTHONPATH=. python src/calibration.py --dataset Cora --model GCN --calibration RBS --config
# PYTHONPATH=. python src/calibration_rbs.py --model GCN --dataset Pubmed --num_bins_rbs 2

