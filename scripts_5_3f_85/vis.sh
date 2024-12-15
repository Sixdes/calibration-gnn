#!/bin/sh

# PYTHONPATH=. python src/calibration_uncal.py --dataset Cora --model GCN --model_retrain_name lognorm --wdecay 5e-4 
# PYTHONPATH=. python src/calibration_uncal.py --dataset Citeseer --model GCN --model_retrain_name lognorm --wdecay 5e-4 

# PYTHONPATH=. python src/calibration_uncal.py --dataset Cora --model GCN --model_retrain_name gcl-grid --wdecay 5e-4 
# PYTHONPATH=. python src/calibration_uncal.py --dataset Citeseer --model GCN --model_retrain_name gcl-grid --wdecay 5e-4 

# PYTHONPATH=. python src/calibration_uncal.py --dataset Cora --model GCN --model_retrain_name latest --wdecay 5e-4 
# PYTHONPATH=. python src/calibration_uncal.py --dataset Citeseer --model GCN --model_retrain_name latest --wdecay 5e-4 

# PYTHONPATH=. python src/calibration_uncal.py --dataset Cora --model GCN --model_retrain_name crl --wdecay 5e-4 
# PYTHONPATH=. python src/calibration_uncal.py --dataset Citeseer --model GCN --model_retrain_name crl --wdecay 5e-4 


PYTHONPATH=. python src/calibration_vis.py --dataset Cora --model GCN --calibration ETS --wdecay 5e-4 
PYTHONPATH=. python src/calibration_vis.py --dataset Citeseer --model GCN --calibration ETS --wdecay 5e-4 

PYTHONPATH=. python src/calibration_vis.py --dataset Cora --model GCN --calibration CaGCN --wdecay 5e-4 
PYTHONPATH=. python src/calibration_vis.py --dataset Citeseer --model GCN --calibration CaGCN --wdecay 5e-4
 
PYTHONPATH=. python src/calibration_vis.py --dataset Cora --model GCN --calibration GATS --wdecay 5e-4 
PYTHONPATH=. python src/calibration_vis.py --dataset Citeseer --model GCN --calibration GATS --wdecay 5e-4 