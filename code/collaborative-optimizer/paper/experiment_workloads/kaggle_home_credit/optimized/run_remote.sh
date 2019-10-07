#!/usr/bin/env bash


root='/home/behrouz/collaborative-optimization'

python ~/collaborative-optimization/code/collaborative-optimizer/paper/experiment_workloads/kaggle_home_credit/optimized/start_here_a_gentle_introduction.py \
${root}'/code/collaborative-optimizer/' 'root='${root} 'storage_type=dedup' \
'experiment_graph'=${root}'/data/experiment_graphs/kaggle_home_credit/start_here_a_gentle_introduction/all' \
'update_graph=yes'
