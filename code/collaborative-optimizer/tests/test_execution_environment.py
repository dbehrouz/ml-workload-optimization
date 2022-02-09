from unittest import TestCase

import pandas as pd

from execution_environment import ExecutionEnvironment, UserDefinedFunction


class TestExecutionEnvironment(TestCase):
    class LoadWithData(UserDefinedFunction):
        def __init__(self, path):
            super().__init__(return_type='Dataset')
            self.path = path

        def run(self, underlying_data):
            return pd.read_csv(self.path)

    def test_empty_dataset(self):
        execution_environment = ExecutionEnvironment()
        empty_data = execution_environment.empty_node(node_type='Dataset')
        load_oper = TestExecutionEnvironment.LoadWithData('data/openml/task_id=31/datasets/train.csv')
        loaded_data = empty_data.run_udf(load_oper)
        print(loaded_data.data().head())
        print(f'columns: {loaded_data.get_column()}')
        print(f'column hashes: {loaded_data.get_column_hash()}')
