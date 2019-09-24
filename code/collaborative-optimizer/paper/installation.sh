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


# load results from remote nodes
scp -r behrouz@cloud-41.dima.tu-berlin.de:/home/behrouz/collaborative-optimization/experiment_results/remote/ ./experiment_results/