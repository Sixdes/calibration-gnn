#!/bin/sh

#CoraFull Photo CS Physics Computers
for rate in 20 40 60
do for dataset in Cora Citeseer Pubmed  

do for model in GCN

do

case $dataset in
    Cora|Citeseer|Pubmed) wdecay=5e-4;;
    *)                    wdecay=0;;
esac

PYTHONPATH=. python src/labelrate_setting/train.py --dataset $dataset\
        --model $model \
        --labelrate $rate \
        --wdecay $wdecay >> Log_labelrate/train/original_hd16/${dataset}.log
done
done
done


