#!/usr/bin/env bash

file_name=$1

current_date=$(date +'%Y-%m-%d')
experiment='kaggle_home_credit'
root='/home/behrouz/collaborative-optimization'
result_path=${root}'/experiment_results/remote/reuse/same-workload/kaggle_home_credit/'${current_date}'/'${file_name}

python ~/collaborative-optimization/code/collaborative-optimizer/paper/experiments/reuse/runner-same-workload.py \
${root}'/code/collaborative-optimizer/' 'root='${root} 'result='${result_path} 'experiment='${experiment} \
'workload=introduction_to_manual_feature_engineering' 'mat_budget=0.0' 'method=baseline' 'rep=2'

python ~/collaborative-optimization/code/collaborative-optimizer/paper/experiments/reuse/runner-same-workload.py \
${root}'/code/collaborative-optimizer/' 'root='${root} 'result='${result_path} 'experiment='${experiment} \
'workload=introduction_to_manual_feature_engineering' 'mat_budget=16.0' 'method=optimized' 'rep=2'


python ~/collaborative-optimization/code/collaborative-optimizer/paper/experiments/reuse/runner-same-workload.py \
${root}'/code/collaborative-optimizer/' 'root='${root} 'result='${result_path} 'experiment='${experiment} \
'workload=introduction_to_manual_feature_engineering_p2' 'mat_budget=0.0' 'method=baseline' 'rep=2'

python ~/collaborative-optimization/code/collaborative-optimizer/paper/experiments/reuse/runner-same-workload.py \
${root}'/code/collaborative-optimizer/' 'root='${root} 'result='${result_path} 'experiment='${experiment} \
'workload=introduction_to_manual_feature_engineering_p2' 'mat_budget=16.0' 'method=optimized' 'rep=2'


python ~/collaborative-optimization/code/collaborative-optimizer/paper/experiments/reuse/runner-same-workload.py \
${root}'/code/collaborative-optimizer/' 'root='${root} 'result='${result_path} 'experiment='${experiment} \
'workload=start_here_a_gentle_introduction' 'mat_budget=0.0' 'method=baseline' 'rep=2'

python ~/collaborative-optimization/code/collaborative-optimizer/paper/experiments/reuse/runner-same-workload.py \
${root}'/code/collaborative-optimizer/' 'root='${root} 'result='${result_path} 'experiment='${experiment} \
'workload=start_here_a_gentle_introduction' 'mat_budget=16.0' 'method=optimized' 'rep=2'
