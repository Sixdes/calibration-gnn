# train_and_calibrate

python main.py --model gcn --dataset Cora --early_stopping True --num_bins_rbs 2
python main.py --model gcn --dataset Citeseer --early_stopping True --num_bins_rbs 3
python main.py --model gcn --dataset Pubmed --early_stopping True --num_bins_rbs 2

python main.py --model gat --dataset Cora --early_stopping True --patience 100 --epochs 1000 --num_bins_rbs 2
python main.py --model gat --dataset Citeseer --early_stopping True --patience 100 --epochs 1000 --num_bins_rbs 3
python main.py --model gat --dataset Pubmed --early_stopping True --patience 100 --epochs 1000 --num_bins_rbs 4