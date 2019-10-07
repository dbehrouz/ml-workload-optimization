#!/usr/bin/env bash

root='/Users/bede01/Documents/work/phd-papers/ml-workload-optimization'

python /Users/bede01/Documents/work/phd-papers/ml-workload-optimization/code/collaborative-optimizer/paper/experiment_workloads/kaggle_home_credit/optimized/start_here_a_gentle_introduction.py \
${root}'/code/collaborative-optimizer/' 'root='${root} 'storage_type=dedup' \
'experiment_graph=/Users/bede01/Documents/work/phd-papers/ml-workload-optimization/data/experiment_graphs/kaggle_home_credit/start_here_a_gentle_introduction/all' \
'update_graph=yes'
