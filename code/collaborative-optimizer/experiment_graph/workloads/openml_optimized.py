import warnings
from datetime import datetime

from experiment_graph.openml_helper.openml_connectors import *
from experiment_graph.optimizations.Reuse import LinearTimeReuse
from experiment_graph.workload import Workload

warnings.filterwarnings("ignore")


class OpenMLOptimizedWorkload(Workload):
    def __init__(self, setup, pipeline, task_id, should_warmstart=False):
        Workload.__init__(self)
        self.setup = setup
        self.pipeline = pipeline
        self.task_id = task_id
        self.score = 0
        self.should_warmstart = should_warmstart

    def run(self, execution_environment, root_data, verbose=0):
        try:
            train_data = execution_environment.load(
                root_data + '/openml/task_id={}/datasets/train.csv'.format(self.task_id))
            test_data = execution_environment.load(
                root_data + '/openml/task_id={}/datasets/test.csv'.format(self.task_id))

            y = train_data['class']
            x = train_data.drop('class')

            test_y = test_data['class']
            test_x = test_data.drop('class')

            edges = skpipeline_to_edge_list(pipeline=self.pipeline, setup=self.setup)

            for i in range(len(edges) - 1):
                model = x.fit_sk_model(edges[i])
                x = model.transform(x)
                test_x = model.transform(test_x)

            # TODO is it OK to assume all the openml pipelines end with a scikit learn model?
            model = x.fit_sk_model_with_labels(edges[-1], y, should_warmstart=self.should_warmstart)

            score = model.score(test_x,
                                test_y,
                                score_type='accuracy').data(verbose=verbose)

            # print 'pipeline: {}, setup: {}, score: {}'.format(self.setup.flow_id, self.setup.setup_id, score)
            self.score = score['accuracy']

            return True
        except Exception as err:
            print('error for pipeline: {}, setup: {}, err: {}'.format(self.pipeline, self.setup.setup_id, err))
            return False

    def get_score(self):
        return self.score


if __name__ == "__main__":
    from paper.experiment_helper import Parser
    from experiment_graph.data_storage import StorageManagerFactory
    from experiment_graph.executor import CollaborativeExecutor
    from experiment_graph.execution_environment import ExecutionEnvironment
    from experiment_graph.materialization_algorithms.materialization_methods import AllMaterializer, \
        StorageAwareMaterializer, HeuristicsMaterializer

    parser = Parser(sys.argv)
    verbose = parser.get('verbose', 0)
    DEFAULT_ROOT = '/Users/bede01/Documents/work/phd-papers/published/ml-workload-optimization'
    ROOT = parser.get('root', DEFAULT_ROOT)
    ROOT_DATA_DIRECTORY = ROOT + '/data'
    mat_budget = float(parser.get('mat_budget', '1.0')) * 1024.0 * 1024.0
    openml_task = int(parser.get('task', 31))
    materializer_type = parser.get('materializer', 'all')
    storage_type = parser.get('storage_type', 'dedup')
    if materializer_type == 'storage_aware':
        materializer = StorageAwareMaterializer(storage_budget=mat_budget)
    elif materializer_type == 'simple':
        materializer = HeuristicsMaterializer(storage_budget=mat_budget)
    elif materializer_type == 'all':
        materializer = AllMaterializer()
    else:
        raise Exception('invalid materializer: {}'.format(materializer_type))

    limit = int(parser.get('limit', 10))
    storage_manager = StorageManagerFactory.get_storage(parser.get('storage_type', 'dedup'))

    OPENML_DIR = ROOT_DATA_DIRECTORY + '/openml'
    config.set_cache_directory(OPENML_DIR + '/cache')
    OPENML_DATASET = ROOT_DATA_DIRECTORY + '/openml/task_id={}'.format(openml_task)
    setup_and_pipelines = get_setup_and_pipeline(OPENML_DATASET + '/all_runs.csv', limit)

    ee = ExecutionEnvironment(storage_manager, reuse_type=LinearTimeReuse.NAME)

    if parser.has('experiment_graph'):
        database_path = parser.get('experiment_graph')
        if os.path.exists(database_path):
            ee.load_history_from_disk(database_path)

    executor = CollaborativeExecutor(ee, materializer=materializer)
    total = 0.0

    for setup, pipeline in setup_and_pipelines:
        execution_start = datetime.now()
        workload = OpenMLOptimizedWorkload(setup, pipeline, task_id=openml_task)
        executor.run_workload(workload=workload, root_data=ROOT_DATA_DIRECTORY, verbose=0)
        print(workload.get_score())
        total += (datetime.now() - execution_start).total_seconds()
        executor.local_process()
        executor.global_process()
        executor.cleanup()

    if parser.get('update_graph', 'No').lower() == 'yes':
        executor.store_experiment_graph(database_path, overwrite=True)

    print('finished execution in {} seconds'.format(total))
