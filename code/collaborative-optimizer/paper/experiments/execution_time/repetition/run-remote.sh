#!/usr/bin/env bash

file_name=$1

current_date=$(date +'%Y-%m-%d/%H-%M')
experiment='kaggle_home_credit'
root='/home/behrouz/collaborative-optimization'
result_path=${root}'/experiment_results/remote/execution_time/repetition/kaggle_home_credit/'${file_name}'/'${current_date}'.csv'

python ~/collaborative-optimization/code/collaborative-optimizer/paper/experiments/execution_time/repetition/runner-same-workload.py \
${root}'/code/collaborative-optimizer/' 'root='${root} 'result='${result_path} 'experiment='${experiment} \
'workload=introduction_to_manual_feature_engineering' 'mat_budget=0.0' 'method=baseline' 'rep=2'

python ~/collaborative-optimization/code/collaborative-optimizer/paper/experiments/execution_time/repetition/runner-same-workload.py \
${root}'/code/collaborative-optimizer/' 'root='${root} 'result='${result_path} 'experiment='${experiment} \
'workload=introduction_to_manual_feature_engineering' 'mat_budget=16.0' 'method=optimized' 'rep=2'

python ~/collaborative-optimization/code/collaborative-optimizer/paper/experiments/execution_time/repetition/runner-same-workload.py \
${root}'/code/collaborative-optimizer/' 'root='${root} 'result='${result_path} 'experiment='${experiment} \
'workload=introduction_to_manual_feature_engineering' 'mat_budget=16.0' 'method=helix' 'rep=2'

python ~/collaborative-optimization/code/collaborative-optimizer/paper/experiments/execution_time/repetition/runner-same-workload.py \
${root}'/code/collaborative-optimizer/' 'root='${root} 'result='${result_path} 'experiment='${experiment} \
'workload=introduction_to_manual_feature_engineering_p2' 'mat_budget=0.0' 'method=baseline' 'rep=2'

python ~/collaborative-optimization/code/collaborative-optimizer/paper/experiments/execution_time/repetition/runner-same-workload.py \
${root}'/code/collaborative-optimizer/' 'root='${root} 'result='${result_path} 'experiment='${experiment} \
'workload=introduction_to_manual_feature_engineering_p2' 'mat_budget=16.0' 'method=optimized' 'rep=2'

python ~/collaborative-optimization/code/collaborative-optimizer/paper/experiments/execution_time/repetition/runner-same-workload.py \
${root}'/code/collaborative-optimizer/' 'root='${root} 'result='${result_path} 'experiment='${experiment} \
'workload=introduction_to_manual_feature_engineering_p2' 'mat_budget=16.0' 'method=helix' 'rep=2'

python ~/collaborative-optimization/code/collaborative-optimizer/paper/experiments/execution_time/repetition/runner-same-workload.py \
${root}'/code/collaborative-optimizer/' 'root='${root} 'result='${result_path} 'experiment='${experiment} \
'workload=start_here_a_gentle_introduction' 'mat_budget=0.0' 'method=baseline' 'rep=2'

python ~/collaborative-optimization/code/collaborative-optimizer/paper/experiments/execution_time/repetition/runner-same-workload.py \
${root}'/code/collaborative-optimizer/' 'root='${root} 'result='${result_path} 'experiment='${experiment} \
'workload=start_here_a_gentle_introduction' 'mat_budget=16.0' 'method=optimized' 'rep=2'

python ~/collaborative-optimization/code/collaborative-optimizer/paper/experiments/execution_time/repetition/runner-same-workload.py \
${root}'/code/collaborative-optimizer/' 'root='${root} 'result='${result_path} 'experiment='${experiment} \
'workload=start_here_a_gentle_introduction' 'mat_budget=16.0' 'method=helix' 'rep=2'

python ~/collaborative-optimization/code/collaborative-optimizer/paper/experiments/execution_time/repetition/runner-same-workload.py \
${root}'/code/collaborative-optimizer/' 'root='${root} 'result='${result_path} 'experiment='${experiment} \
'workload=introduction_to_manual_feature_engineering' 'mat_budget=0.0' 'method=baseline' 'rep=2'

python ~/collaborative-optimization/code/collaborative-optimizer/paper/experiments/execution_time/repetition/runner-same-workload.py \
${root}'/code/collaborative-optimizer/' 'root='${root} 'result='${result_path} 'experiment='${experiment} \
'workload=introduction_to_manual_feature_engineering' 'mat_budget=16.0' 'method=optimized' 'rep=2'

python ~/collaborative-optimization/code/collaborative-optimizer/paper/experiments/execution_time/repetition/runner-same-workload.py \
${root}'/code/collaborative-optimizer/' 'root='${root} 'result='${result_path} 'experiment='${experiment} \
'workload=introduction_to_manual_feature_engineering' 'mat_budget=16.0' 'method=helix' 'rep=2'

python ~/collaborative-optimization/code/collaborative-optimizer/paper/experiments/execution_time/repetition/runner-same-workload.py \
${root}'/code/collaborative-optimizer/' 'root='${root} 'result='${result_path} 'experiment='${experiment} \
'workload=introduction_to_manual_feature_engineering_p2' 'mat_budget=0.0' 'method=baseline' 'rep=2'

python ~/collaborative-optimization/code/collaborative-optimizer/paper/experiments/execution_time/repetition/runner-same-workload.py \
${root}'/code/collaborative-optimizer/' 'root='${root} 'result='${result_path} 'experiment='${experiment} \
'workload=introduction_to_manual_feature_engineering_p2' 'mat_budget=16.0' 'method=optimized' 'rep=2'

python ~/collaborative-optimization/code/collaborative-optimizer/paper/experiments/execution_time/repetition/runner-same-workload.py \
${root}'/code/collaborative-optimizer/' 'root='${root} 'result='${result_path} 'experiment='${experiment} \
'workload=introduction_to_manual_feature_engineering_p2' 'mat_budget=16.0' 'method=helix' 'rep=2'

python ~/collaborative-optimization/code/collaborative-optimizer/paper/experiments/execution_time/repetition/runner-same-workload.py \
${root}'/code/collaborative-optimizer/' 'root='${root} 'result='${result_path} 'experiment='${experiment} \
'workload=start_here_a_gentle_introduction' 'mat_budget=0.0' 'method=baseline' 'rep=2'

python ~/collaborative-optimization/code/collaborative-optimizer/paper/experiments/execution_time/repetition/runner-same-workload.py \
${root}'/code/collaborative-optimizer/' 'root='${root} 'result='${result_path} 'experiment='${experiment} \
'workload=start_here_a_gentle_introduction' 'mat_budget=16.0' 'method=optimized' 'rep=2'

python ~/collaborative-optimization/code/collaborative-optimizer/paper/experiments/execution_time/repetition/runner-same-workload.py \
${root}'/code/collaborative-optimizer/' 'root='${root} 'result='${result_path} 'experiment='${experiment} \
'workload=start_here_a_gentle_introduction' 'mat_budget=16.0' 'method=helix' 'rep=2'

python ~/collaborative-optimization/code/collaborative-optimizer/paper/experiments/execution_time/repetition/runner-same-workload.py \
${root}'/code/collaborative-optimizer/' 'root='${root} 'result='${result_path} 'experiment='${experiment} \
'workload=introduction_to_manual_feature_engineering' 'mat_budget=0.0' 'method=baseline' 'rep=2'

python ~/collaborative-optimization/code/collaborative-optimizer/paper/experiments/execution_time/repetition/runner-same-workload.py \
${root}'/code/collaborative-optimizer/' 'root='${root} 'result='${result_path} 'experiment='${experiment} \
'workload=introduction_to_manual_feature_engineering' 'mat_budget=16.0' 'method=optimized' 'rep=2'

python ~/collaborative-optimization/code/collaborative-optimizer/paper/experiments/execution_time/repetition/runner-same-workload.py \
${root}'/code/collaborative-optimizer/' 'root='${root} 'result='${result_path} 'experiment='${experiment} \
'workload=introduction_to_manual_feature_engineering' 'mat_budget=16.0' 'method=helix' 'rep=2'

python ~/collaborative-optimization/code/collaborative-optimizer/paper/experiments/execution_time/repetition/runner-same-workload.py \
${root}'/code/collaborative-optimizer/' 'root='${root} 'result='${result_path} 'experiment='${experiment} \
'workload=introduction_to_manual_feature_engineering_p2' 'mat_budget=0.0' 'method=baseline' 'rep=2'

python ~/collaborative-optimization/code/collaborative-optimizer/paper/experiments/execution_time/repetition/runner-same-workload.py \
${root}'/code/collaborative-optimizer/' 'root='${root} 'result='${result_path} 'experiment='${experiment} \
'workload=introduction_to_manual_feature_engineering_p2' 'mat_budget=16.0' 'method=optimized' 'rep=2'

python ~/collaborative-optimization/code/collaborative-optimizer/paper/experiments/execution_time/repetition/runner-same-workload.py \
${root}'/code/collaborative-optimizer/' 'root='${root} 'result='${result_path} 'experiment='${experiment} \
'workload=introduction_to_manual_feature_engineering_p2' 'mat_budget=16.0' 'method=helix' 'rep=2'

python ~/collaborative-optimization/code/collaborative-optimizer/paper/experiments/execution_time/repetition/runner-same-workload.py \
${root}'/code/collaborative-optimizer/' 'root='${root} 'result='${result_path} 'experiment='${experiment} \
'workload=start_here_a_gentle_introduction' 'mat_budget=0.0' 'method=baseline' 'rep=2'

python ~/collaborative-optimization/code/collaborative-optimizer/paper/experiments/execution_time/repetition/runner-same-workload.py \
${root}'/code/collaborative-optimizer/' 'root='${root} 'result='${result_path} 'experiment='${experiment} \
'workload=start_here_a_gentle_introduction' 'mat_budget=16.0' 'method=optimized' 'rep=2'

python ~/collaborative-optimization/code/collaborative-optimizer/paper/experiments/execution_time/repetition/runner-same-workload.py \
${root}'/code/collaborative-optimizer/' 'root='${root} 'result='${result_path} 'experiment='${experiment} \
'workload=start_here_a_gentle_introduction' 'mat_budget=16.0' 'method=helix' 'rep=2'
