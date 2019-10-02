from paper.experiment_helper import ExperimentWorkloadFactory

EXPERIMENT_SCENARIO = ['start_here_a_gentle_introduction',
                       'introduction_to_manual_feature_engineering',
                       'introduction_to_manual_feature_engineering_p2',
                       'fork_cridata_start_here_a_gentle_introduction',
                       'fork_taozhongxiao_start_here_a_gentle_introduction']

MOCK_SCENARIO = ['mock_workload_1', 'mock_workload_2', 'mock_workload_3']


def get_scenario(scenario_type):
    if scenario_type == 'baseline':
        return get_kaggle_baseline_scenario(EXPERIMENT_SCENARIO)
    elif scenario_type == 'optimized':
        return get_kaggle_optimized_scenario(scenario=EXPERIMENT_SCENARIO)
    elif scenario_type == 'mock_optimized':
        return get_kaggle_optimized_scenario(package='mock_optimized', scenario=EXPERIMENT_SCENARIO)
    elif scenario_type == 'mock':
        return get_mock_scenario(MOCK_SCENARIO)


def get_mock_scenario(scenario=None):
    scenario = MOCK_SCENARIO if scenario is None else scenario
    print scenario
    experiment_name = 'mock'
    method = 'baseline'

    return [ExperimentWorkloadFactory.get_workload(experiment_name, method, w) for w in scenario]


def get_kaggle_baseline_scenario(scenario=None):
    scenario = EXPERIMENT_SCENARIO if scenario is None else scenario
    experiment_name = 'kaggle_home_credit'
    method = 'baseline'

    return [ExperimentWorkloadFactory.get_workload(experiment_name, method, w) for w in scenario]


def get_kaggle_optimized_scenario(package='optimized', scenario=None):
    scenario = EXPERIMENT_SCENARIO if scenario is None else scenario
    experiment_name = 'kaggle_home_credit'

    return [ExperimentWorkloadFactory.get_workload(experiment_name, package, w) for w in scenario]
