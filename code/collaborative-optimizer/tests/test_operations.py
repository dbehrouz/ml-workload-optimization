from unittest import TestCase

import pandas as pd

from execution_environment import ExecutionEnvironment, UserDefinedFunction


class TestExecutionEnvironment(TestCase):
    def setUp(self) -> None:
        self.execution_environment = ExecutionEnvironment()

    def test_simple_udf(self):
        class ClipDataset(UserDefinedFunction):
            def __init__(self, lower=None, upper=None, axis=None):
                super().__init__(return_type='Dataset')
                self.lower = lower
                self.upper = upper
                self.axis = axis

            def run(self, underlying_data):
                # here the underlying_data is a pandas dataframe and we are directly calling the pandas clip function
                return underlying_data.clip(lower=self.lower, upper=self.upper, axis=self.axis)

        data = self.execution_environment.load('data/sample.csv')

        clip_oper = ClipDataset(lower=2, upper=4)
        clipped = data.run_udf(clip_oper)

        expected = pd.DataFrame.from_dict({'a': [2, 4, 2], 'b': [2, 4, 2], 'c': [3, 4, 2], 'd': [4, 4, 3]})
        pd.testing.assert_frame_equal(clipped.data(), expected)

    def test_2_input_udf(self):
        class ColumnWiseSum(UserDefinedFunction):
            def __init__(self):
                super().__init__(return_type='Dataset')

            def run(self, *underlying_data):
                """
                Here, we assume underlying_data is a list of only 2 dataframes
                :param underlying_data:
                :return:
                """
                return underlying_data[0] + underlying_data[1]

        data = self.execution_environment.load('data/sample.csv')
        data_other = self.execution_environment.load('data/sample_2.csv')

        column_wise_sum = ColumnWiseSum()
        result = data.run_udf(operation=column_wise_sum, other_inputs=data_other)

        expected_result = pd.DataFrame.from_dict({'a': [-5, 9, 0], 'b': [3, 9, 0], 'c': [6, 12, 8], 'd': [8, 12, 4]})

        pd.testing.assert_frame_equal(result.data(), expected_result)

    def test_n_input_udf(self):
        class ColumnWiseSum(UserDefinedFunction):
            def __init__(self):
                super().__init__(return_type='Dataset')

            def run(self, *underlying_data):
                """
                Here, we assume underlying_data is a list of multiple dataframes
                :param underlying_data:
                :return:
                """
                return sum(underlying_data)

        data = self.execution_environment.load('data/sample.csv')
        data_1 = self.execution_environment.load('data/sample_2.csv')
        data_2 = data.copy()
        data_3 = data_1.copy()

        column_wise_sum = ColumnWiseSum()
        result = data.run_udf(operation=column_wise_sum, other_inputs=[data_1, data_2, data_3])
        expected_result = pd.DataFrame.from_dict(
            {'a': [-10, 18, 0], 'b': [6, 18, 0], 'c': [12, 24, 16], 'd': [16, 24, 8]})

        pd.testing.assert_frame_equal(result.data(), expected_result)
