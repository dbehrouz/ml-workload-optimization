from unittest import TestCase

from nose_parameterized import parameterized

from execution_environment import ExecutionEnvironment, UserDefinedFunction, SuperNode, GroupBy
from executor import CollaborativeExecutor
from materialization_methods import AllMaterializer, HeuristicsMaterializer, StorageAwareMaterializer, HelixMaterializer


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

    @parameterized.expand([
        ('AllMaterializer', AllMaterializer()),
        ('HeuristicsMaterializer', HeuristicsMaterializer(1000)),
        ('StorageAwareMaterializer', StorageAwareMaterializer(1000)),
        ('HelixMaterializer', HelixMaterializer(1000))
    ])
    def test_unmaterializable_nodes(self, name, materializer):
        class IdentityFunction(UserDefinedFunction):
            def __init__(self):
                super().__init__(return_type='Dataset')

            def run(self, underlying_data):
                # here the underlying_data is a pandas dataframe and we are directly calling the pandas clip function
                return underlying_data

        executor = CollaborativeExecutor(execution_environment=self.execution_environment,
                                         materializer=materializer)
        print(self.execution_environment.experiment_graph.graph.nodes())
        data = self.execution_environment.load('data/sample.csv')
        identity_function = IdentityFunction()
        # Normal call, result is materializable, hash = E2C4590331CE548DB3614C3AF20B7177
        materializable_dataset = data.run_udf(operation=identity_function)
        materializable_dataset.data()

        # Result is unmaterializable hash = C7064173E8FE5BF4E661D546091DE811
        unmaterializable_dataset = materializable_dataset.run_udf(operation=identity_function,
                                                                  unmaterializable_result=True)
        unmaterializable_dataset.data()

        # Calling the post processing and materialization
        executor.local_process()
        executor.global_process()

        history_graph = executor.execution_environment.experiment_graph.graph

        # E2C4590331CE548DB3614C3AF20B7177
        self.assertTrue(history_graph.nodes['E2C4590331CE548DB3614C3AF20B7177']['mat'])

        # C7064173E8FE5BF4E661D546091DE811
        self.assertFalse(history_graph.nodes['C7064173E8FE5BF4E661D546091DE811']['mat'])
