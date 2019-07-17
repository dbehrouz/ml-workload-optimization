#!/usr/bin/env bash


### Run Materializer ####
# Runing on Local machine
python code/collaborative-optimizer/paper/experiments/materializer.py '/Users/bede01/Documents/work/phd-papers/ml-workload-optimization/code/collaborative-optimizer/' 'root=/Users/bede01/Documents/work/phd-papers/ml-workload-optimization/' \
'experiment=kaggle_home_credit' 'workload=introduction_to_manual_feature_engineering' 'mat_rates=0.25, 0.50, 0.75, 1.0'

### Running on cloud ###
python ~/collaborative-optimization/code/collaborative-optimizer/paper/experiments/materializer.py '/home/behrouz/collaborative-optimization/code/collaborative-optimizer/' 'root=/home/behrouz/collaborative-optimization' \
'experiment=kaggle_home_credit' 'workload=introduction_to_manual_feature_engineering' 'mode=remote' 'mat_rates=0.25, 0.50, 0.75, 1.0'

python ~/collaborative-optimization/code/collaborative-optimizer/paper/experiments/materializer.py '/home/behrouz/collaborative-optimization/code/collaborative-optimizer/' 'root=/home/behrouz/collaborative-optimization' \
'experiment=kaggle_home_credit' 'workload=introduction_to_manual_feature_engineering_p2' 'mode=remote' 'mat_rates=0.25, 0.50, 0.75, 1.0'

python ~/collaborative-optimization/code/collaborative-optimizer/paper/experiments/materializer.py '/home/behrouz/collaborative-optimization/code/collaborative-optimizer/' 'root=/home/behrouz/collaborative-optimization' \
'experiment=kaggle_home_credit' 'workload=start_here_a_gentle_introduction' 'mode=remote' 'mat_rates=0.25, 0.50, 0.75, 1.0'





scp -r behrouz@cloud-41.dima.tu-berlin.de:/home/behrouz/collaborative-optimization/experiment_results/remote /Users/bede01/Documents/work/phd-papers/ml-workload-optimization/experiment_results