from unittest import TestCase

from Reuse import HelixReuse
from execution_environment import ExecutionEnvironment
from executor import HelixExecutor
from workload import Workload


class BaseWorkload(Workload):
    def run(self, execution_environment, root_data, verbose):
        # Load Data
        train = execution_environment.load(root_data + '/openml/task_id=31/datasets/train.csv')
        test = execution_environment.load(root_data + '/openml/task_id=31/datasets/test.csv')

        test_labels = test['class']
        test = test.drop('class')

        train_labels = train['class']
        train = train.drop(columns=['class'])

        from experiment_graph.sklearn_helper.preprocessing import MinMaxScaler
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(train)
        train = scaler.transform(train)
        # train.data()
        test = scaler.transform(test)

        # Random Forest 1 (n_estimator = 10)
        from experiment_graph.sklearn_helper.ensemble import RandomForestClassifier
        random_forest10 = RandomForestClassifier(n_estimators=10, random_state=50, verbose=1, n_jobs=-1)
        random_forest10.fit(train, train_labels)

        # Execute
        # random_forest10.trained_node.data()

        print('score: {}'.format(random_forest10.score(test, test_labels).data(verbose=verbose)))
        return True


class ReuseMinMaxWorkload(Workload):
    def run(self, execution_environment, root_data, verbose):
        # Load Data
        train = execution_environment.load(root_data + '/openml/task_id=31/train.csv')
        test = execution_environment.load(root_data + '/openml/task_id=31/test.csv')

        test_labels = test['class']
        test = test.drop('class')

        train_labels = train['class']
        train = train.drop(columns=['class'])

        from experiment_graph.sklearn_helper.preprocessing import MinMaxScaler
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(train)
        train = scaler.transform(train)
        test = scaler.transform(test)
        train.data(verbose=verbose)
        return True


class TestReuse(TestCase):
    def setUp(self):
        self.min_max_reuse = ReuseMinMaxWorkload()
        self.executor = HelixExecutor()
        self.base_workload = BaseWorkload()
        self.root_data = '../../data'

    def test_run_helix(self):
        self.executor.end_to_end_run(self.base_workload, root_data=self.root_data, verbose=1)

        self.executor.end_to_end_run(self.min_max_reuse, root_data=self.root_data, verbose=1)
