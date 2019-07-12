#!/usr/bin/env bash

# copy scripts
rsync -rav -e ssh --include='*.py' ./code/ behrouz@cloud-40.dima.tu-berlin.de:/home/behrouz/collaborative-optimization/code/

# preprocessing kaggle home credit
python collaborative-optimization/code/collaborative-optimizer/paper/experiment_workloads/reuse/kaggle_home_credit/competition_preprocessing.py '/home/behrouz/collaborative-optimization/code/collaborative-optimizer/' 'root=/home/behrouz/collaborative-optimization'


# Runing on Local machine
python runner-same-workload.py '/Users/bede01/Documents/work/phd-papers/ml-workload-optimization/code/collaborative-optimizer/' 'root=/Users/bede01/Documents/work/phd-papers/ml-workload-optimization/' \
'experiment=kaggle_home_credit' 'workload=introduction_to_manual_feature_engineering_p2' 'mode=local' 'rep=3'

# Running on cloud
python collaborative-optimization/code/collaborative-optimizer/paper/experiment_workloads/reuse/runner-same-workload.py '/home/behrouz/collaborative-optimization/code/collaborative-optimizer/' 'root=/home/behrouz/collaborative-optimization' \
'experiment=kaggle_home_credit' 'workload=introduction_to_manual_feature_engineering_p2' 'mode=remote' 'rep=3'