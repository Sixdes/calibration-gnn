#!/bin/sh

# for dataset in Cora Citeseer Pubmed Computers Photo CS Physics CoraFull
for dataset in Citeseer

# do for model in GCN GAT

do

case $dataset in
    Cora|Citeseer|Pubmed) wdecay=5e-4;;
    *)                    wdecay=0;;
esac

echo "processing the dataset"$dataset
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python -u src/train.py --dataset $dataset\
        --model GCN \
        --wdecay $wdecay

# CUDA_VISIBLE_DEVICES=1 nohup bash -c "PYTHONPATH=. python -u src/train.py --dataset $dataset\
#         --model GAT \
#         --wdecay $wdecay" > train_stage.log 2>&1 &

done





