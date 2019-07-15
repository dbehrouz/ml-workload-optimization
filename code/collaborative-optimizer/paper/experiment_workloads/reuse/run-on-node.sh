#!/usr/bin/env bash

# install packages
pip2 install -U pandas --user
pip2 install -U scikit-learn --user
pip2 install -U networkx --user
pip2 install -U kaggle --user
pip2 install -U matplotlib --user
pip2 install -U seaborn --user
pip2 install -U pympler --user
pip2.7 install -U lightgbm --user
# copy scripts
rsync -rav -e ssh --include='*.py' ./code/ behrouz@cloud-41.dima.tu-berlin.de:/home/behrouz/collaborative-optimization/code/

# preprocessing kaggle home credit
kaggle competitions download -c home-credit-default-risk
unzip '*.zip'
mkdir original_train_test
mv application_train.csv original_train_test/
mv application_test.csv original_train_test/
python collaborative-optimization/code/collaborative-optimizer/paper/experiment_workloads/reuse/kaggle_home_credit/competition_preprocessing.py '/home/behrouz/collaborative-optimization/code/collaborative-optimizer/' 'root=/home/behrouz/collaborative-optimization'


#### Same Workload All Experiments ####
# Runing on Local machine
python runner-same-workload.py '/Users/bede01/Documents/work/phd-papers/ml-workload-optimization/code/collaborative-optimizer/' 'root=/Users/bede01/Documents/work/phd-papers/ml-workload-optimization/' \
'experiment=kaggle_home_credit' 'workload=start_here_a_gentle_introduction' 'mode=local' 'rep=2' 'mat_rate=0.75' 'run_baseline=yes'

### Running on cloud ###
python collaborative-optimization/code/collaborative-optimizer/paper/experiment_workloads/reuse/runner-same-workload.py '/home/behrouz/collaborative-optimization/code/collaborative-optimizer/' 'root=/home/behrouz/collaborative-optimization' \
'experiment=kaggle_home_credit' 'workload=introduction_to_manual_feature_engineering' 'mode=remote' 'rep=2' 'mat_rate=[1.0, 0.75, 0.50, 0.25, 0.0]' 'run_baseline=yes'

python collaborative-optimization/code/collaborative-optimizer/paper/experiment_workloads/reuse/runner-same-workload.py '/home/behrouz/collaborative-optimization/code/collaborative-optimizer/' 'root=/home/behrouz/collaborative-optimization' \
'experiment=kaggle_home_credit' 'workload=introduction_to_manual_feature_engineering_p2' 'mode=remote' 'rep=2' 'mat_rate=[1.0, 0.75, 0.50, 0.25, 0.0]' 'run_baseline=yes'


python collaborative-optimization/code/collaborative-optimizer/paper/experiment_workloads/reuse/runner-same-workload.py '/home/behrouz/collaborative-optimization/code/collaborative-optimizer/' 'root=/home/behrouz/collaborative-optimization' \
'experiment=kaggle_home_credit' 'workload=start_here_a_gentle_introduction' 'mode=remote' 'rep=2' 'mat_rate=[1.0, 0.75, 0.50, 0.25, 0.0]' 'run_baseline=yes'



scp -r behrouz@cloud-41.dima.tu-berlin.de:/home/behrouz/collaborative-optimization/experiment_results/ /Users/bede01/Documents/work/phd-papers/ml-workload-optimization/experiment_results