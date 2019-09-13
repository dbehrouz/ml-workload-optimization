from abc import abstractmethod
from experiment_graph.execution_environment import ExecutionEnvironment
from experiment_graph.workload import Workload


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
    def __init__(self, execution_environment):
        """

        :type execution_environment: ExecutionEnvironment
        """
        Executor.__init__(self)
        self.execution_environment = execution_environment

    def run(self, workload, **args):
        """
        Here is the function for the main workflow of the collaborative optimizer system:
        1. Run the workload and return the result
        2. Update the Experiment Graph with edges and nodes of the workload (do not store underlying data yet)
        3. Run Materialization on the Graph and store the underlying either on the storage manager or graph itself
        :type workload: Workload
        """
        args['execution_environment'] = self.execution_environment
        workload.run(**args)
        self.execution_environment.workload_dag.post_process()
        self.execution_environment.experiment_graph.extend(self.execution_environment.workload_dag)
        # Materialize
        # Store the Dataset and Features in the Storage Manager


class BaselineExecutor(Executor):
    def __init__(self):
        Executor.__init__(self)

    def run(self, workload, **args):
        """

        :type workload: Workload
        """
        workload.run(**args)
