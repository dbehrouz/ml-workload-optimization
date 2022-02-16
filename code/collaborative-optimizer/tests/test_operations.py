from unittest import TestCase

import pandas as pd

from execution_environment import ExecutionEnvironment, UserDefinedFunction, MultiInputUserDefinedFunction


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

    def test_multi_input_udf(self):
        class OthersNotSet(MultiInputUserDefinedFunction):
            def __init__(self):
                super().__init__(return_type='Dataset')

            def run(self, underlying_data):
                # here the underlying_data is a pandas dataframe and we are directly calling the pandas clip function
                return underlying_data

        with self.assertRaises(ValueError):
            data = self.execution_environment.load('data/sample.csv')
            multi_input_oper = OthersNotSet()
            result = data.run_udf(multi_input_oper)
            result.data()

    
