from abc import abstractmethod
from experiment_graph.execution_environment import ExecutionEnvironment
from experiment_graph.workload import Workload
from heuristics import compute_recreation_cost, compute_vertex_potential
from experiment_graph.materialization_algorithms.materialization_methods import AllMaterializer


class Executor:

    def __init__(self):
        pass

    def end_to_end_run(self, workload, **args):
        """
        end to end run with all the optimizations
        calls these methods in order:
        1. run_workload
        2. local_process
        3. global_process
        4. cleanup
        :type workload: Workload
        """
        self.run_workload(workload, **args)
        self.local_process()
        self.global_process()
        self.cleanup()

    @abstractmethod
    def run_workload(self, workload, **args):
        """
        only runs the script, for running experiments and capturing run time
        :param workload:
        :param args:
        :return:
        """
        pass

    @abstractmethod
    def local_process(self):
        pass

    @abstractmethod
    def global_process(self):
        pass

    @abstractmethod
    def cleanup(self):
        pass


class CollaborativeExecutor(Executor):
    def __init__(self, execution_environment, materializer=None):
        """

        :type execution_environment: ExecutionEnvironment
        """
        Executor.__init__(self)
        self.execution_environment = execution_environment
        self.materializer = AllMaterializer(storage_budget=0) if materializer is None else materializer
        self.time_manager = {}

    # def end_to_end_run(self, workload, **args):
    #     """
    #     Here is the function for the main workflow of the collaborative optimizer system:
    #     1. Run the workload and return the result
    #     2. Update the Experiment Graph with edges and nodes of the workload (do not store underlying data yet)
    #     3. Run Materialization on the Graph and store the underlying either on the storage manager or graph itself
    #     :type workload: Workload
    #     """

    def run_workload(self, workload, **args):
        """
        Run the workload and return the result
        :param workload:
        :param args:
        :return:
        """
        args['execution_environment'] = self.execution_environment
        # execute the workload and post process the workload dag
        workload.run(**args)

    def local_process(self):
        """
        post process the workload dag by computing the sizes and adding model scores
        :return:
        """
        self.execution_environment.workload_dag.post_process()

    def global_process(self):
        """
        Run Materialization on the Graph and store the underlying either on the storage manager or graph itself
        """
        self.execution_environment.experiment_graph.extend(self.execution_environment.workload_dag)
        # TODO: implementing this in a truly online manner, i.e., only computing nodes which are
        # affected is a bit of work. For now, we recompute for the entire graph
        self.compute_heuristics(self.execution_environment.experiment_graph.graph)
        self.materializer.run_and_materialize(self.execution_environment.experiment_graph,
                                              self.execution_environment.workload_dag)

    def cleanup(self):
        """
        clean up the workload from the executor
        """
        # copy the time manager
        for k, v in self.execution_environment.time_manager.iteritems():
            if k in self.time_manager:
                self.time_manager[k] += v
            else:
                self.time_manager[k] = v
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

    def run_workload(self, workload, **args):
        workload.run(**args)

    def local_process(self):
        pass

    def global_process(self):
        pass

    def cleanup(self):
        pass
