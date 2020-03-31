#!/usr/bin/env bash

current_date=$(date +'%Y-%m-%d/%H-%M')
experiment='kaggle_home_credit'
root='/Users/bede01/Documents/work/phd-papers/ml-workload-optimization'
result_path=${root}'/experiment_results/local/reuse_benchmarking/kaggle_home_credit/'${current_date}'.csv'

python /Users/bede01/Documents/work/phd-papers/ml-workload-optimization/code/collaborative-optimizer/paper/experiments/reuse_benchmarking/run_experiment.py \
${root}'/code/collaborative-optimizer/' 'root='${root} 'result='${result_path} 'experiment='${experiment} 'graph_path='${root}'/data/experiment_graphs/kaggle_home_credit/benchmarking/graph'

