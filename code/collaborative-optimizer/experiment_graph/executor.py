from abc import abstractmethod

from data_storage import DedupedStorageManager
from experiment_graph.benchmark_helper import BenchmarkMetrics
from experiment_graph.execution_environment import ExecutionEnvironment
from experiment_graph.materialization_algorithms.materialization_methods import AllMaterializer, \
    StorageAwareMaterializer, HelixMaterializer
from experiment_graph.optimizations.Reuse import HelixReuse
from experiment_graph.workload import Workload
from heuristics import compute_recreation_cost, compute_vertex_potential, compute_load_costs


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
        if not self.run_workload(workload, **args):
            return False
        if not self.local_process():
            return False
        if not self.global_process():
            return False
        if not self.cleanup():
            return False
        return True

    @abstractmethod
    def run_workload(self, workload, **args):
        """
        only runs the script, for running experiments and capturing run time
        :param workload:
        :param args:
        :rtype: bool
        """
        pass

    @abstractmethod
    def local_process(self):
        """

        :rtype: bool
        """
        pass

    @abstractmethod
    def global_process(self):
        """

        :rtype: bool
        """
        pass

    @abstractmethod
    def cleanup(self):
        """

        :rtype: bool
        """
        pass

    @abstractmethod
    def num_of_executed_operations(self):
        """

        :return:
        """
        pass


class CollaborativeExecutor(Executor):
    DEFAULT_PROFILE = {"Agg": 0.08959999999999999, "SK_Model": 0.0002258063871079322, "Evaluation": 0.02909090909090909,
                       "Feature": 2.5703229163279242e-05, "Dataset": 0.0005039403928584662}

    def __init__(self, execution_environment, cost_profile=None, materializer=None):
        """

        :type execution_environment: ExecutionEnvironment
        """
        Executor.__init__(self)
        self.cost_profile = CollaborativeExecutor.DEFAULT_PROFILE if cost_profile is None else cost_profile
        self.execution_environment = execution_environment
        self.materializer = AllMaterializer() if materializer is None else materializer
        # storage aware materialization only works with deduped storage manager
        if isinstance(self.materializer, StorageAwareMaterializer):
            assert isinstance(execution_environment.experiment_graph.data_storage, DedupedStorageManager)

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
        return workload.run(**args)

    def local_process(self):
        """
        post process the workload dag by computing the sizes and adding model scores
        :return:
        """
        self.execution_environment.workload_dag.post_process()
        return True

    def global_process(self):
        """
        Run Materialization on the Graph and store the underlying either on the storage manager or graph itself
        """
        self.execution_environment.experiment_graph.extend(self.execution_environment.workload_dag)
        # TODO: implementing this in a truly online manner, i.e., only computing nodes which are
        # affected is a bit of work. For now, we recompute for the entire graph
        self.compute_heuristics(self.execution_environment.experiment_graph.graph, self.cost_profile)
        self.materializer.run_and_materialize(self.execution_environment.experiment_graph,
                                              self.execution_environment.workload_dag)
        return True

    def cleanup(self):
        """
        clean up the workload from the executor
        """
        # copy the time manager
        self.execution_environment.compute_total_reuse_optimization_time()
        for k, v in self.execution_environment.time_manager.iteritems():
            if k in self.time_manager:
                self.time_manager[k] += v
            else:
                self.time_manager[k] = v
        self.execution_environment.new_workload()
        return True

    def num_of_executed_operations(self):
        return self.execution_environment.scheduler.get_num_of_operations()

    def store_experiment_graph(self, database_path, overwrite=False):
        self.execution_environment.save_history(database_path, overwrite=overwrite)

    def get_benchmark_results(self, keys=None):
        if keys is None:
            return ','.join(
                ['NOT CAPTURED' if key not in self.time_manager else str(self.time_manager[key]) for key in
                 BenchmarkMetrics.keys])
        else:
            return ','.join([self.time_manager[key] for key in keys])

    @staticmethod
    def compute_heuristics(graph, profile):
        compute_load_costs(graph, profile)
        compute_recreation_cost(graph)
        compute_vertex_potential(graph)


class HelixExecutor(Executor):
    DEFAULT_PROFILE = {"Agg": 0.08959999999999999, "SK_Model": 0.0002258063871079322, "Evaluation": 0.02909090909090909,
                       "Feature": 2.5703229163279242e-05, "Dataset": 0.0005039403928584662}

    def __init__(self, cost_profile=None, budget=16.0):
        """
        :type budget: object
        :type cost_profile: object

        """
        Executor.__init__(self)
        self.cost_profile = CollaborativeExecutor.DEFAULT_PROFILE if cost_profile is None else cost_profile
        self.execution_environment = ExecutionEnvironment(reuse_type=HelixReuse.NAME)
        self.materializer = HelixMaterializer(storage_budget=budget)
        # storage aware materialization only works with deduped storage manager

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
        return workload.run(**args)

    def local_process(self):
        """
        post process the workload dag by computing the sizes and adding model scores
        :return:
        """
        self.execution_environment.workload_dag.post_process()
        return True

    def global_process(self):
        """
        Run Materialization on the Graph and store the underlying either on the storage manager or graph itself
        """
        self.execution_environment.experiment_graph.extend(self.execution_environment.workload_dag)
        # TODO: implementing this in a truly online manner, i.e., only computing nodes which are
        # affected is a bit of work. For now, we recompute for the entire graph
        self.compute_heuristics(self.execution_environment.experiment_graph.graph, self.cost_profile)
        self.materializer.run_and_materialize(self.execution_environment.experiment_graph,
                                              self.execution_environment.workload_dag)
        return True

    def cleanup(self):
        """
        clean up the workload from the executor
        """
        # copy the time manager
        self.execution_environment.compute_total_reuse_optimization_time()
        for k, v in self.execution_environment.time_manager.iteritems():
            if k in self.time_manager:
                self.time_manager[k] += v
            else:
                self.time_manager[k] = v
        self.execution_environment.new_workload()
        return True

    def num_of_executed_operations(self):
        return self.execution_environment.scheduler.get_num_of_operations()

    def store_experiment_graph(self, database_path, overwrite=False):
        self.execution_environment.save_history(database_path, overwrite=overwrite)

    def get_benchmark_results(self, keys=None):
        if keys is None:
            return ','.join(
                ['NOT CAPTURED' if key not in self.time_manager else str(self.time_manager[key]) for key in
                 BenchmarkMetrics.keys])
        else:
            return ','.join([self.time_manager[key] for key in keys])

    @staticmethod
    def compute_heuristics(graph, profile):
        compute_load_costs(graph, profile)
        compute_recreation_cost(graph)
        compute_vertex_potential(graph)


class BaselineExecutor(Executor):
    def __init__(self):
        Executor.__init__(self)

    def run_workload(self, workload, **args):
        return workload.run(**args)

    def local_process(self):
        return True

    def global_process(self):
        return True

    def cleanup(self):
        return True

    def num_of_executed_operations(self):
        return [0]
