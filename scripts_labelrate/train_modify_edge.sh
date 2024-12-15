#!/bin/sh

#CoraFull
# for dataset in Photo CS Physics 

for rate in 20 40 60 

do for dataset in Cora Citeseer Pubmed

do for model in GCN

do

case $dataset in
    Cora|Citeseer|Pubmed) wdecay=5e-4;;
    *)                    wdecay=0;;
esac

PYTHONPATH=. python src/labelrate_setting/train_modify_edge.py --dataset $dataset\
        --model $model \
        --is_edge_delete \
        --labelrate $rate \
        --wdecay $wdecay >> Log_labelrate/modify_edge/train_all_modify_edge_hidden16/${dataset}_delete.log

PYTHONPATH=. python src/labelrate_setting/train_modify_edge.py --dataset $dataset\
        --model $model \
        --is_edge_add \
        --labelrate $rate \
        --wdecay $wdecay >> Log_labelrate/modify_edge/train_all_modify_edge_hidden16/${dataset}_add.log
done
done
done

