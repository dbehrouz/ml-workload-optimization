#!/usr/bin/env bash


### Running on cloud ###
for materialization_rate in 0.0 0.25 0.50 0.75 1.0
do

python ~/collaborative-optimization/code/collaborative-optimizer/paper/experiments/reuse/runner-same-workload-inmemory.py '/home/behrouz/collaborative-optimization/code/collaborative-optimizer/' 'root=/home/behrouz/collaborative-optimization' \
'experiment=kaggle_home_credit' 'workload=introduction_to_manual_feature_engineering_p2' 'mode=remote'  'mat_rate='${materialization_rate} 'run_baseline=no'

done


python ~/collaborative-optimization/code/collaborative-optimizer/paper/experiments/reuse/runner-same-workload-inmemory.py  '/home/behrouz/collaborative-optimization/code/collaborative-optimizer/' 'root=/home/behrouz/collaborative-optimization' \
'experiment=kaggle_home_credit' 'workload=introduction_to_manual_feature_engineering_p2' 'mode=remote' 'run_baseline=yes'

