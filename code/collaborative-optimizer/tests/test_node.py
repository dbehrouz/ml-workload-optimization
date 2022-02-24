from unittest import TestCase

from execution_environment import ExecutionEnvironment, UserDefinedFunction, SuperNode, GroupBy


class TestNode(TestCase):
    def setUp(self) -> None:
        self.execution_environment = ExecutionEnvironment()

    def test_set_super_node_cannot_be_materializable(self):
        super_node = SuperNode('id', self.execution_environment, [])
        with self.assertRaises(ValueError):
            super_node.unmaterializable = False

        groupby_node = GroupBy('group-id', self.execution_environment)
        with self.assertRaises(ValueError):
            groupby_node.unmaterializable = False

    def test_unmaterializable_nodes(self):
        class IdentityFunction(UserDefinedFunction):
            def __init__(self):
                super().__init__(return_type='Dataset')

            def run(self, underlying_data):
                # here the underlying_data is a pandas dataframe and we are directly calling the pandas clip function
                return underlying_data

        data = self.execution_environment.load('data/sample.csv')

        identity_function = IdentityFunction()
        unmaterializable_dataset = data.run_udf(operation=identity_function, unmaterializable_result=True)

        unmaterializable_dataset.data()
