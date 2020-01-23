#!/usr/bin/env bash

current_date=$(date +'%Y-%m-%d/%H-%M')
experiment='kaggle_home_credit'
root='/Users/bede01/Documents/work/phd-papers/ml-workload-optimization'
result_path=${root}'/experiment_results/local/execution_time/different_workloads/kaggle_home_credit/'${current_date}'.csv'

python /Users/bede01/Documents/work/phd-papers/ml-workload-optimization/code/collaborative-optimizer/paper/experiments/execution_time/different_workloads/run_experiment.py \
${root}'/code/collaborative-optimizer/' 'root='${root} 'result='${result_path} 'experiment='${experiment} \
'mat_budget=0.0' 'method=baseline'

python /Users/bede01/Documents/work/phd-papers/ml-workload-optimization/code/collaborative-optimizer/paper/experiments/execution_time/different_workloads/run_experiment.py \
${root}'/code/collaborative-optimizer/' 'root='${root} 'result='${result_path} 'experiment='${experiment} \
'mat_budget=5.0' 'method=optimized'

python /Users/bede01/Documents/work/phd-papers/ml-workload-optimization/code/collaborative-optimizer/paper/experiments/execution_time/different_workloads/run_experiment.py \
${root}'/code/collaborative-optimizer/' 'root='${root} 'result='${result_path} 'experiment='${experiment} \
'mat_budget=5.0' 'method=helix'
