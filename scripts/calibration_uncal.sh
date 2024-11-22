#!/bin/sh

for re_name in  gcl-grid-1115
# for re_name in lognorm gcl-grid crl gcl-tune

# do for dataset in Photo CS Physics  
do for dataset in  Cora Citeseer Pubmed Computers Photo CS Physics 

do for model in GCN GAT 
do 

case $dataset in
    Cora|Citeseer|Pubmed) wdecay=5e-4;;
    *)                    wdecay=0;;
esac

PYTHONPATH=. python src/calibration_uncal.py --dataset $dataset \
        --model $model \
        --model_retrain_name $re_name \
        --wdecay $wdecay  >> Log/uncal/${re_name}_${model}_1115.log
done
done
done

# >> Log_train/uncal/${re_name}_${model}.log
# sh src/calibration_uncal.py --dataset Cora --model GCN --model_retrain_name lognorm --wdecay 5e-4 