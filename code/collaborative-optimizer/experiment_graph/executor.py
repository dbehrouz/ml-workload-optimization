from abc import abstractmethod
from experiment_graph.execution_environment import ExecutionEnvironment
from experiment_graph.workload import Workload
from heuristics import compute_recreation_cost, compute_vertex_potential
from experiment_graph.materialization_algorithms.materialization_methods import AllMaterializer


class Executor:

    def __init__(self):
        pass

    @abstractmethod
    def end_to_end_run(self, workload, **args):
        """
        end to end run with all the optimizations
        :type workload: Workload
        """
        pass

    @abstractmethod
    def run_workload(self, workload, **args):
        """
        only runs the script, for running experiments and capturing run time
        :param workload:
        :param args:
        :return:
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

    def end_to_end_run(self, workload, **args):
        """
        Here is the function for the main workflow of the collaborative optimizer system:
        1. Run the workload and return the result
        2. Update the Experiment Graph with edges and nodes of the workload (do not store underlying data yet)
        3. Run Materialization on the Graph and store the underlying either on the storage manager or graph itself
        :type workload: Workload
        """

        self.run_workload(workload, **args)

        self.update_and_materialize()

        self.workload_cleanup()

    def run_workload(self, workload, **args):
        args['execution_environment'] = self.execution_environment
        # execute the workload and post process the workload dag
        workload.run(**args)

    def update_and_materialize(self):
        """
        update the experiment graph and run materialization algorithm
        """
        self.execution_environment.workload_dag.post_process()
        self.execution_environment.experiment_graph.extend(self.execution_environment.workload_dag)
        # TODO: implementing this in a truly online manner, i.e., only computing nodes which are
        # affected is a bit of work. For now, we recompute for the entire graph
        self.compute_heuristics(self.execution_environment.experiment_graph.graph)
        self.materializer.run_and_materialize(self.execution_environment.experiment_graph,
                                              self.execution_environment.workload_dag)

    def workload_cleanup(self):
        """
        clean up the workload from the executor
        """
        self.execution_environment.new_workload()

    def store_experiment_graph(self, database_path, overwrite=False):
        self.execution_environment.save_history(database_path, overwrite=overwrite)

    @staticmethod
    def compute_heuristics(graph):
        compute_recreation_cost(graph)
        compute_vertex_potential(graph)


class BaselineExecutor(Executor):
    def __init__(self):
        Executor.__init__(self)

    def end_to_end_run(self, workload, **args):
        """

        :type workload: Workload
        """
        self.run_workload(workload, **args)

    def run_workload(self, workload, **args):
        workload.run(**args)
