#!/usr/bin/env bash

# Runing on Local machine
python runner-same-workload.py 'root=/Users/bede01/Documents/work/phd-papers/ml-workload-optimization/' \
'experiment=kaggle_home_credit' 'workload=introduction_to_manual_feature_engineering_p2' 'mode=local' 'rep=3'

# Running on cloud
python runner-same-workload.py 'root=/Users/bede01/Documents/work/phd-papers/ml-workload-optimization/' \
'experiment=kaggle_home_credit' 'workload=introduction_to_manual_feature_engineering_p2' 'mode=remote' 'rep=3'