from paper.experiment_helper import ExperimentWorkloadFactory

scenario = ['start_here_a_gentle_introduction', 'introduction_to_manual_feature_engineering',
            'introduction_to_manual_feature_engineering_p2', 'fork_cridata_start_here_a_gentle_introduction']


def get_kaggle_baseline_scenario():
    experiment_name = 'kaggle_home_credit'
    method = 'baseline'

    return [ExperimentWorkloadFactory.get_workload(experiment_name, method, w) for w in scenario]


def get_kaggle_optimized_scenario():
    experiment_name = 'kaggle_home_credit'
    method = 'optimized'

    return [ExperimentWorkloadFactory.get_workload(experiment_name, method, w) for w in scenario]
