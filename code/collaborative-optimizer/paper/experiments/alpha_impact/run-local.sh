#!/usr/bin/env bash

current_date=$(date +'%Y-%m-%d/%H-%M')
experiment='openml'
root='/Users/bede01/Documents/work/phd-papers/ml-workload-optimization'
result_path=${root}'/experiment_results/local/alpha_impact/openml/'${current_date}'.csv'

python /Users/bede01/Documents/work/phd-papers/ml-workload-optimization/code/collaborative-optimizer/paper/experiments/alpha_impact/run_experiment.py \
${root}'/code/collaborative-optimizer/' 'root='${root} 'result='${result_path} 'experiment='${experiment} 'mat_type=best_n' 'alpha=0.1' 'limit=200'

python /Users/bede01/Documents/work/phd-papers/ml-workload-optimization/code/collaborative-optimizer/paper/experiments/alpha_impact/run_experiment.py \
${root}'/code/collaborative-optimizer/' 'root='${root} 'result='${result_path} 'experiment='${experiment} 'mat_type=best_n' 'alpha=0.5' 'limit=200'

python /Users/bede01/Documents/work/phd-papers/ml-workload-optimization/code/collaborative-optimizer/paper/experiments/alpha_impact/run_experiment.py \
${root}'/code/collaborative-optimizer/' 'root='${root} 'result='${result_path} 'experiment='${experiment} 'mat_type=best_n' 'alpha=0.9' 'limit=200'

python /Users/bede01/Documents/work/phd-papers/ml-workload-optimization/code/collaborative-optimizer/paper/experiments/alpha_impact/run_experiment.py \
${root}'/code/collaborative-optimizer/' 'root='${root} 'result='${result_path} 'experiment='${experiment} 'mat_type=oracle' 'limit=200'
