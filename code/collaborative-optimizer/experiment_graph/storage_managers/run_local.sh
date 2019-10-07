#!/usr/bin/env bash

root='/Users/bede01/Documents/work/phd-papers/ml-workload-optimization'

python /Users/bede01/Documents/work/phd-papers/ml-workload-optimization/code/collaborative-optimizer/experiment_graph/storage_managers/storage_profiler.py \
${root}'/code/collaborative-optimizer/' \
'experiment_graph=/Users/bede01/Documents/work/phd-papers/ml-workload-optimization/data/experiment_graphs/kaggle_home_credit/start_here_a_gentle_introduction/all' \
'trial=5' 'profile=local-dedup-1' 'result_folder=/Users/bede01/Documents/work/phd-papers/ml-workload-optimization/data/profiles'
