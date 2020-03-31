#!/usr/bin/env bash

file_name=$1

current_date=$(date +'%Y-%m-%d/%H-%M')
experiment='kaggle_home_credit'
root='/home/behrouz/collaborative-optimization'
result_path=${root}'/experiment_results/remote/reuse_benchmarking/kaggle_home_credit/'${file_name}'/'${current_date}'.csv'

python ~/collaborative-optimization/code/collaborative-optimizer/paper/experiments/reuse_benchmarking/run_experiment.py \
${root}'/code/collaborative-optimizer/' 'root='${root} 'result='${result_path} 'experiment='${experiment} \
'number_of_workloads=1000' 'iter=10000'
