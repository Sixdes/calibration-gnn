# Calibration loss

# CUDA_VISIBLE_DEVICES=1 python train_and_calibrate/main_loss.py --model gcn --dataset Cora --early_stopping True --patience 100 --epochs 1000 --add_cal_loss True --alpha 0.98 --num_runs 10
CUDA_VISIBLE_DEVICES=1 python train_and_calibrate/main_loss.py --model gcn --dataset Citeseer --early_stopping True --patience 100 --epochs 1000 --add_cal_loss True --alpha 0.99 --num_runs 10
CUDA_VISIBLE_DEVICES=1 python train_and_calibrate/main_loss.py --model gcn --dataset Pubmed --early_stopping True --patience 100 --epochs 1000 --add_cal_loss True --alpha 0.97 --num_runs 10

CUDA_VISIBLE_DEVICES=1 python train_and_calibrate/main_loss.py --model gat --dataset Cora --early_stopping True --patience 100 --epochs 1000 --add_cal_loss True --alpha 0.97 --num_runs 10
CUDA_VISIBLE_DEVICES=1 python train_and_calibrate/main_loss.py --model gat --dataset Citeseer --early_stopping True --patience 100 --epochs 1000 --add_cal_loss True --alpha 0.98 --num_runs 10
CUDA_VISIBLE_DEVICES=1 python train_and_calibrate/main_loss.py --model gat --dataset Pubmed --early_stopping True --patience 100 --epochs 1000 --add_cal_loss True --alpha 0.98 --num_runs 10