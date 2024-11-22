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

for dataset in Computers Photo CS Physics 

do for model in GCN GAT

do

case $dataset in
    Cora|Citeseer|Pubmed) wdecay=5e-4;;
    *)                    wdecay=0;;
esac

PYTHONPATH=. python src/train_gcl_grid.py --dataset $dataset\
        --model $model \
        --wdecay $wdecay >> Log/train_gcl_grid-1115-2.log
done
done

