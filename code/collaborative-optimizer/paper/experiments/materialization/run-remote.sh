#!/usr/bin/env bash

file_name=$1

current_date=$(date +'%Y-%m-%d')
experiment='kaggle_home_credit'
root='/home/behrouz/collaborative-optimization'
result_path=${root}'/experiment_results/remote/materialization/kaggle_home_credit/'${file_name}'/'${current_date}'.csv'


python ~/collaborative-optimization/code/collaborative-optimizer/paper/experiments/materialization/run_experiment.py \
${root}'/code/collaborative-optimizer/' 'root='${root} 'result='${result_path} 'experiment='${experiment} \
'mat_budget=32.0' 'method=optimized' 'materializer=storage_aware'

python ~/collaborative-optimization/code/collaborative-optimizer/paper/experiments/materialization/run_experiment.py \
${root}'/code/collaborative-optimizer/' 'root='${root} 'result='${result_path} 'experiment='${experiment} \
'mat_budget=32.0' 'method=optimized' 'materializer=simple'
