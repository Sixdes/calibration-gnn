#!/bin/sh

# for dataset in Cora Citeseer Pubmed Computers Photo CS Physics CoraFull
# for dataset in Citeseer

# # do for model in GCN GAT

# do

# case $dataset in
#     Cora|Citeseer|Pubmed) wdecay=5e-4;;
#     *)                    wdecay=0;;
# esac

# echo "processing the dataset"$dataset
# CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python -u src/train.py --dataset $dataset\
#         --model GCN \
#         --wdecay $wdecay
# done

# Cora Citeseer
for dataset in Pubmed

do for model in GCN GAT

do

case $dataset in
    Cora|Citeseer|Pubmed) wdecay=5e-4;;
    *)                    wdecay=0;;
esac

PYTHONPATH=. python src/5_3f_85_setting/train_modify_edge.py --dataset $dataset\
        --model $model \
        --is_edge_delete \
        --wdecay $wdecay >> Log_5_3f_85/modify_edge/train_all_modify_edge_hidden16/${dataset}_delete.log

PYTHONPATH=. python src/5_3f_85_setting/train_modify_edge.py --dataset $dataset\
        --model $model \
        --is_edge_add \
        --wdecay $wdecay >> Log_5_3f_85/modify_edge/train_all_modify_edge_hidden16/${dataset}_add.log
done
done



