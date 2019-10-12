#!/usr/bin/env bash

current_date=$(date +'%Y-%m-%d/%H-%M')
experiment='openml'
root='/Users/bede01/Documents/work/phd-papers/ml-workload-optimization'
result_path=${root}'/experiment_results/local/model_materialization/openml/'${current_date}'.csv'

python /Users/bede01/Documents/work/phd-papers/ml-workload-optimization/code/collaborative-optimizer/paper/experiments/model_materialization/run_experiment.py \
${root}'/code/collaborative-optimizer/' 'root='${root} 'result='${result_path} 'experiment='${experiment} 'mat_budget=1.0' 'method=optimized' 'limit=500'

python /Users/bede01/Documents/work/phd-papers/ml-workload-optimization/code/collaborative-optimizer/paper/experiments/model_materialization/run_experiment.py \
${root}'/code/collaborative-optimizer/' 'root='${root} 'result='${result_path} 'experiment='${experiment} 'method=baseline' 'limit=500'
