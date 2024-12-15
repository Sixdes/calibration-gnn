#!/bin/sh

#CoraFull Computers Photo CS Physics
for dataset in Cora Citeseer Pubmed  

do for model in GCN 

do

case $dataset in
    Cora|Citeseer|Pubmed) wdecay=5e-4;;
    *)                    wdecay=0;;
esac

PYTHONPATH=. python src/5_3f_85_setting/train.py --dataset $dataset\
        --model $model \
        --wdecay $wdecay
done
done

