#!/usr/bin/env bash

file_name=$1

current_date=$(date +'%Y-%m-%d/%H-%M')
experiment='openml'
root='/home/behrouz/collaborative-optimization'
result_path=${root}'/experiment_results/remote/warmstarting/openml/'${file_name}'/'${current_date}'.csv'

python ~/collaborative-optimization/code/collaborative-optimizer/paper/experiments/warmstarting/run_experiment.py \
${root}'/code/collaborative-optimizer/' 'root='${root} 'result='${result_path} 'experiment='${experiment} 'mat_budget=2.0' \
'method=optimized' 'limit=2000' 'profile='${root}'/data/profiles/cloud-41-dedup'

python ~/collaborative-optimization/code/collaborative-optimizer/paper/experiments/warmstarting/run_experiment.py \
${root}'/code/collaborative-optimizer/' 'root='${root} 'result='${result_path} 'experiment='${experiment} 'method=baseline' \
'limit=2000' 'profile='${root}'/data/profiles/cloud-41-dedup'

python ~/collaborative-optimization/code/collaborative-optimizer/paper/experiments/warmstarting/run_experiment.py \
${root}'/code/collaborative-optimizer/' 'root='${root} 'result='${result_path} 'experiment='${experiment} 'mat_budget=2.0' \
'method=optimized' 'limit=2000' 'profile='${root}'/data/profiles/cloud-41-dedup' 'warmstart=0'
