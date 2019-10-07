#!/usr/bin/env bash

root='/home/behrouz/collaborative-optimization'

python  ~/collaborative-optimization/code/collaborative-optimizer/experiment_graph/storage_managers/storage_profiler.py \
${root}'/code/collaborative-optimizer/' \
'experiment_graph'=${root}'/data/experiment_graphs/kaggle_home_credit/start_here_a_gentle_introduction/all' \
'trial=10' 'profile=cloud-41-dedup' 'result_folder='${root}'/data/profiles'
