#!/usr/bin/env bash

#### Same Workload All Experiments ####
# Runing on Local machine
#for materialization_rate in 0.0 0.25 0.50 0.75 1.0
#do
#python runner-same-workload.py '/Users/bede01/Documents/work/phd-papers/ml-workload-optimization/code/collaborative-optimizer/' 'root=/Users/bede01/Documents/work/phd-papers/ml-workload-optimization/' \
#'experiment=kaggle_home_credit' 'workload=introduction_to_manual_feature_engineering' 'mode=local' 'mat_rate='${materialization_rate} 'run_baseline=no'
#
#python runner-same-workload.py '/Users/bede01/Documents/work/phd-papers/ml-workload-optimization/code/collaborative-optimizer/' 'root=/Users/bede01/Documents/work/phd-papers/ml-workload-optimization/' \
#'experiment=kaggle_home_credit' 'workload=introduction_to_manual_feature_engineering_p2' 'mode=local' 'mat_rate='${materialization_rate} 'run_baseline=no'
#
#
#python runner-same-workload.py '/Users/bede01/Documents/work/phd-papers/ml-workload-optimization/code/collaborative-optimizer/' 'root=/Users/bede01/Documents/work/phd-papers/ml-workload-optimization/' \
#'experiment=kaggle_home_credit' 'workload=start_here_a_gentle_introduction' 'mode=local' 'mat_rate='${materialization_rate} 'run_baseline=no'
#done


python runner-same-workload.py '/Users/bede01/Documents/work/phd-papers/ml-workload-optimization/code/collaborative-optimizer/' 'root=/Users/bede01/Documents/work/phd-papers/ml-workload-optimization/' \
'experiment=kaggle_home_credit' 'workload=introduction_to_manual_feature_engineering' 'mode=local' 'run_baseline=yes'

python runner-same-workload.py '/Users/bede01/Documents/work/phd-papers/ml-workload-optimization/code/collaborative-optimizer/' 'root=/Users/bede01/Documents/work/phd-papers/ml-workload-optimization/' \
'experiment=kaggle_home_credit' 'workload=introduction_to_manual_feature_engineering_p2' 'mode=local' 'run_baseline=yes'


python runner-same-workload.py '/Users/bede01/Documents/work/phd-papers/ml-workload-optimization/code/collaborative-optimizer/' 'root=/Users/bede01/Documents/work/phd-papers/ml-workload-optimization/' \
'experiment=kaggle_home_credit' 'workload=start_here_a_gentle_introduction' 'mode=local' 'run_baseline=yes'
