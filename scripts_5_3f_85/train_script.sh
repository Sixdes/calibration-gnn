#!/bin/sh

# for dataset in Cora Citeseer Pubmed Computers Photo CS Physics CoraFull
# for dataset in Cora Citeseer Pubmed Computers Photo CS Physics

# # do for model in GCN GAT

# do

# case $dataset in
#     Cora|Citeseer|Pubmed) wdecay=5e-4;;
#     *)                    wdecay=0;;
# esac

# echo "processing the dataset"$dataset
# CUDA_VISIBLE_DEVICES=0 nohup bash -c "PYTHONPATH=. python -u src/train.py --dataset $dataset\
#         --model GCN \
#         --wdecay $wdecay" > Log/train/${dataset}_GCN.log 2>&1 &

# CUDA_VISIBLE_DEVICES=1 nohup bash -c "PYTHONPATH=. python -u src/train.py --dataset $dataset\
#         --model GAT \
#         --wdecay $wdecay" > Log/train/${dataset}_GAT.log 2>&1 &
# # done
# done

#!/bin/sh

#CoraFull
# for dataset in Photo CS Physics 

for dataset in Cora Citeseer Pubmed

do for model in GCN

do

case $dataset in
    Cora|Citeseer|Pubmed) wdecay=5e-4;;
    *)                    wdecay=0;;
esac

# PYTHONPATH=. python src/train_modify_edge.py --dataset $dataset\
#         --model $model \
#         --is_edge_delete \
#         --wdecay $wdecay >> Log/modify_edge/train_all_modify_edge/${dataset}_delete.log

PYTHONPATH=. python src/train_modify_edge.py --dataset $dataset\
        --model $model \
        --is_edge_add \
        --wdecay $wdecay >> Log/5_3f_85_setting/modify_edge/train_valid_modify_edge/${dataset}_add.log
done
done

