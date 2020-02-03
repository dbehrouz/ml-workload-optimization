#!/usr/bin/env bash

file_name=$1

current_date=$(date +'%Y-%m-%d/%H-%M')
experiment='kaggle_home_credit'
root='/home/behrouz/collaborative-optimization'
result_path=${root}'/experiment_results/remote/execution_time/different_workloads/kaggle_home_credit/'${file_name}'/'${current_date}'.csv'


#python ~/collaborative-optimization/code/collaborative-optimizer/paper/experiments/execution_time/different_workloads/run_experiment.py \
#${root}'/code/collaborative-optimizer/' 'root='${root} 'result='${result_path} 'experiment='${experiment} \
#'mat_budget=16.0' 'method=optimized' 'profile='${root}'/data/profiles/cloud-41-dedup'

python ~/collaborative-optimization/code/collaborative-optimizer/paper/experiments/execution_time/different_workloads/run_experiment.py \
${root}'/code/collaborative-optimizer/' 'root='${root} 'result='${result_path} 'experiment='${experiment} \
'mat_budget=16.0' 'method=helix' 'profile='${root}'/data/profiles/cloud-41-dedup'

#python ~/collaborative-optimization/code/collaborative-optimizer/paper/experiments/execution_time/different_workloads/run_experiment.py \
#${root}'/code/collaborative-optimizer/' 'root='${root} 'result='${result_path} 'experiment='${experiment} \
# 'mat_budget=0.0' 'method=baseline' 'profile='${root}'/data/profiles/cloud-41-dedup'
#
# python ~/collaborative-optimization/code/collaborative-optimizer/paper/experiments/execution_time/different_workloads/run_experiment.py \
#${root}'/code/collaborative-optimizer/' 'root='${root} 'result='${result_path} 'experiment='${experiment} \
#'mat_budget=16.0' 'method=optimized' 'profile='${root}'/data/profiles/cloud-41-dedup'

 python ~/collaborative-optimization/code/collaborative-optimizer/paper/experiments/execution_time/different_workloads/run_experiment.py \
${root}'/code/collaborative-optimizer/' 'root='${root} 'result='${result_path} 'experiment='${experiment} \
'mat_budget=16.0' 'method=helix' 'profile='${root}'/data/profiles/cloud-41-dedup'

#python ~/collaborative-optimization/code/collaborative-optimizer/paper/experiments/execution_time/different_workloads/run_experiment.py \
#${root}'/code/collaborative-optimizer/' 'root='${root} 'result='${result_path} 'experiment='${experiment} \
# 'mat_budget=0.0' 'method=baseline' 'profile='${root}'/data/profiles/cloud-41-dedup'
#
#python ~/collaborative-optimization/code/collaborative-optimizer/paper/experiments/execution_time/different_workloads/run_experiment.py \
#${root}'/code/collaborative-optimizer/' 'root='${root} 'result='${result_path} 'experiment='${experiment} \
#'mat_budget=16.0' 'method=optimized' 'profile='${root}'/data/profiles/cloud-41-dedup'

python ~/collaborative-optimization/code/collaborative-optimizer/paper/experiments/execution_time/different_workloads/run_experiment.py \
${root}'/code/collaborative-optimizer/' 'root='${root} 'result='${result_path} 'experiment='${experiment} \
'mat_budget=16.0' 'method=helix' 'profile='${root}'/data/profiles/cloud-41-dedup'

#python ~/collaborative-optimization/code/collaborative-optimizer/paper/experiments/execution_time/different_workloads/run_experiment.py \
#${root}'/code/collaborative-optimizer/' 'root='${root} 'result='${result_path} 'experiment='${experiment} \
# 'mat_budget=0.0' 'method=baseline' 'profile='${root}'/data/profiles/cloud-41-dedup'