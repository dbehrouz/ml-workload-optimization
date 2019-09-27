#!/usr/bin/env python

"""Optimized Fork of Workload 1

This script is the optimized version of the fork of workload 1 submitted by user taozhongxiao
which utilizes our Experiment Graph for optimizing the workload.
"""
import os
import warnings
# matplotlib and seaborn for plotting
from datetime import datetime

import matplotlib

from experiment_graph.workload import Workload

matplotlib.use('ps')

import matplotlib.pyplot as plt
import numpy as np
# numpy and pandas for data manipulation
import pandas as pd
import seaborn as sns

# Experiment Graph

# Suppress warnings
warnings.filterwarnings('ignore')


class fork_taozhongxiao_start_here_a_gentle_introduction(Workload):

    def run(self, execution_environment, root_data, verbose=0):
        MAX_EVALS = 5
        print(os.listdir(root_data))
        train_features = execution_environment.load(root_data + '/kaggle_home_credit/application_train.csv')
        train_features = train_features.sample(n=16000, random_state=42)
        print(train_features.shape)
        return True
