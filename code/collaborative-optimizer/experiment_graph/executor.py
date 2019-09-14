from abc import abstractmethod
from experiment_graph.execution_environment import ExecutionEnvironment
from experiment_graph.workload import Workload
from heuristics import compute_recreation_cost, compute_vertex_potential
from materialization_methods import AllMaterializer


class Executor:

    def __init__(self):
        pass

    @abstractmethod
    def run(self, workload, **args):
        """

        :type workload: Workload
        """
        pass


class CollaborativeExecutor(Executor):
    def __init__(self, execution_environment, materializer=None):
        """

        :type execution_environment: ExecutionEnvironment
        """
        Executor.__init__(self)
        self.execution_environment = execution_environment
        self.materializer = AllMaterializer(storage_budget=0) if materializer is None else materializer

    def run(self, workload, **args):
        """
        Here is the function for the main workflow of the collaborative optimizer system:
        1. Run the workload and return the result
        2. Update the Experiment Graph with edges and nodes of the workload (do not store underlying data yet)
        3. Run Materialization on the Graph and store the underlying either on the storage manager or graph itself
        :type workload: Workload
        """
        args['execution_environment'] = self.execution_environment
        # execute the workload and post process the workload dag
        workload.run(**args)
        self.execution_environment.workload_dag.post_process()

        self.execution_environment.experiment_graph.extend(self.execution_environment.workload_dag)
        # TODO: implementing this in a truly online manner, i.e., only computing nodes which are
        # affected is a bit of work. For now, we recompute for the entire graph
        self.compute_heuristics(self.execution_environment.experiment_graph.graph)
        self.materializer.run_and_materialize(self.execution_environment.experiment_graph,
                                              self.execution_environment.workload_dag)
        self.execution_environment.new_workload()

    @staticmethod
    def compute_heuristics(graph):
        compute_recreation_cost(graph)
        compute_vertex_potential(graph)


class BaselineExecutor(Executor):
    def __init__(self):
        Executor.__init__(self)

    def run(self, workload, **args):
        """

        :type workload: Workload
        """
        workload.run(**args)
