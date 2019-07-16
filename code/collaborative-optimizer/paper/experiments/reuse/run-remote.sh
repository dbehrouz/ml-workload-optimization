#!/usr/bin/env bash



python ~/collaborative-optimization/code/collaborative-optimizer/paper/experiments/reuse/runner-same-workload.py '/home/behrouz/collaborative-optimization/code/collaborative-optimizer/' 'root=/home/behrouz/collaborative-optimization' \
'experiment=kaggle_home_credit' 'workload=introduction_to_manual_feature_engineering' 'mode=remote' 'mat_rate=0.0' 'run_baseline=no' 'rep=1'


#python collaborative-optimization/code/collaborative-optimizer/paper/experiment_workloads/reuse/runner-same-workload.py '/home/behrouz/collaborative-optimization/code/collaborative-optimizer/' 'root=/home/behrouz/collaborative-optimization' \
#'experiment=kaggle_home_credit' 'workload=introduction_to_manual_feature_engineering_p2' 'mode=remote'  'mat_rate='${materialization_rate} 'run_baseline=no'


python ~/collaborative-optimization/code/collaborative-optimizer/paper/experiments/reuse/runner-same-workload.py '/home/behrouz/collaborative-optimization/code/collaborative-optimizer/' 'root=/home/behrouz/collaborative-optimization' \
'experiment=kaggle_home_credit' 'workload=start_here_a_gentle_introduction' 'mode=remote'  'mat_rate=0.0' 'run_baseline=no' 'rep=1'

### Running on cloud ###
for materialization_rate in 0.0 0.25 0.50 0.75 1.0
do
python ~/collaborative-optimization/code/collaborative-optimizer/paper/experiments/reuse/runner-same-workload.py '/home/behrouz/collaborative-optimization/code/collaborative-optimizer/' 'root=/home/behrouz/collaborative-optimization' \
'experiment=kaggle_home_credit' 'workload=introduction_to_manual_feature_engineering' 'mode=remote' 'mat_rate='${materialization_rate} 'run_baseline=no' 'rep=2'


#python collaborative-optimization/code/collaborative-optimizer/paper/experiment_workloads/reuse/runner-same-workload.py '/home/behrouz/collaborative-optimization/code/collaborative-optimizer/' 'root=/home/behrouz/collaborative-optimization' \
#'experiment=kaggle_home_credit' 'workload=introduction_to_manual_feature_engineering_p2' 'mode=remote'  'mat_rate='${materialization_rate} 'run_baseline=no'


python ~/collaborative-optimization/code/collaborative-optimizer/paper/experiments/reuse/runner-same-workload.py '/home/behrouz/collaborative-optimization/code/collaborative-optimizer/' 'root=/home/behrouz/collaborative-optimization' \
'experiment=kaggle_home_credit' 'workload=start_here_a_gentle_introduction' 'mode=remote'  'mat_rate='${materialization_rate} 'run_baseline=no' 'rep=2'
done

python ~/collaborative-optimization/code/collaborative-optimizer/paper/experiments/reuse/runner-same-workload.py '/home/behrouz/collaborative-optimization/code/collaborative-optimizer/' 'root=/home/behrouz/collaborative-optimization' \
'experiment=kaggle_home_credit' 'workload=introduction_to_manual_feature_engineering' 'mode=remote' 'run_baseline=yes' 'rep=1'

python ~/collaborative-optimization/code/collaborative-optimizer/paper/experiments/reuse/runner-same-workload.py '/home/behrouz/collaborative-optimization/code/collaborative-optimizer/' 'root=/home/behrouz/collaborative-optimization' \
'experiment=kaggle_home_credit' 'workload=introduction_to_manual_feature_engineering' 'mode=remote' 'run_baseline=yes' 'rep=2'


#python collaborative-optimization/code/collaborative-optimizer/paper/experiment_workloads/reuse/runner-same-workload.py '/home/behrouz/collaborative-optimization/code/collaborative-optimizer/' 'root=/home/behrouz/collaborative-optimization' \
#'experiment=kaggle_home_credit' 'workload=introduction_to_manual_feature_engineering_p2' 'mode=remote' 'run_baseline=yes'


python ~/collaborative-optimization/code/collaborative-optimizer/paper/experiments/reuse/runner-same-workload.py '/home/behrouz/collaborative-optimization/code/collaborative-optimizer/' 'root=/home/behrouz/collaborative-optimization' \
'experiment=kaggle_home_credit' 'workload=start_here_a_gentle_introduction' 'mode=remote' 'run_baseline=yes' 'rep=1'

python ~/collaborative-optimization/code/collaborative-optimizer/paper/experiments/reuse/runner-same-workload.py '/home/behrouz/collaborative-optimization/code/collaborative-optimizer/' 'root=/home/behrouz/collaborative-optimization' \
'experiment=kaggle_home_credit' 'workload=start_here_a_gentle_introduction' 'mode=remote' 'run_baseline=yes' 'rep=2'


# scp -r behrouz@cloud-41.dima.tu-berlin.de:/home/behrouz/collaborative-optimization/experiment_results/remote /Users/bede01/Documents/work/phd-papers/ml-workload-optimization/experiment_results
