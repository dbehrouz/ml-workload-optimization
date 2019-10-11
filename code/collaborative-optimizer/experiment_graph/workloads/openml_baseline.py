import warnings
from datetime import datetime

from sklearn.metrics import accuracy_score

from experiment_graph.openml_helper.openml_connectors import *
from experiment_graph.workload import Workload

warnings.filterwarnings("ignore")
from openml import config


class OpenMLBaselineWorkload(Workload):
    def __init__(self, setup, pipeline, task_id):
        Workload.__init__(self)
        self.setup = setup
        self.pipeline = pipeline
        self.task_id = task_id
        self.score = 0.0

    def run(self, root_data):
        try:
            train_data = pd.read_csv(
                root_data + '/openml/task_id={}/datasets/train.csv'.format(self.task_id), header='infer',
                index_col=False)
            test_data = pd.read_csv(
                root_data + '/openml/task_id={}/datasets/test.csv'.format(self.task_id), header='infer',
                index_col=False)

            y = train_data['class']
            X = train_data.drop(['class'], axis=1)

            test_y = test_data['class']
            test_x = test_data.drop(['class'], axis=1)

            edges = skpipeline_to_edge_list(pipeline=self.pipeline, setup=self.setup)

            for i in range(len(edges) - 1):
                transformer = edges[i]
                transformer.fit(X)
                X = transformer.transform(X)
                test_x = transformer.transform(test_x)

            model = edges[-1]
            model.fit(X, y)
            predictions = model.predict(test_x)
            self.score = accuracy_score(test_y, predictions)
            return True
        except:
            print 'error for pipeline: {}, setup: {}'.format(self.pipeline, self.setup)
            return False

        # print 'pipeline: {}, setup: {}, score: {0:.6f}'.format(self.setup.flow_id, self.setup.setup_id, score)

    def get_score(self):
        return self.score


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        SOURCE_CODE_ROOT = sys.argv[1]
    else:
        SOURCE_CODE_ROOT = '/Users/bede01/Documents/work/phd-papers/ml-workload-optimization/code/collaborative' \
                           '-optimizer/ '
    sys.path.append(SOURCE_CODE_ROOT)
    from paper.experiment_helper import Parser
    from experiment_graph.executor import BaselineExecutor

    parser = Parser(sys.argv)
    verbose = parser.get('verbose', 0)
    DEFAULT_ROOT = '/Users/bede01/Documents/work/phd-papers/ml-workload-optimization'
    ROOT = parser.get('root', DEFAULT_ROOT)
    ROOT_DATA_DIRECTORY = ROOT + '/data'

    openml_task = int(parser.get('task', 31))

    limit = int(parser.get('limit', 20))

    OPENML_DIR = ROOT_DATA_DIRECTORY + '/openml/'
    config.set_cache_directory(OPENML_DIR + '/cache')
    OPENML_DATASET = ROOT_DATA_DIRECTORY + '/openml/task_id={}'.format(openml_task)
    setup_and_pipelines = get_setup_and_pipeline(OPENML_DATASET + '/all_runs.csv', limit)

    executor = BaselineExecutor()

    execution_start = datetime.now()

    for setup, pipeline in setup_and_pipelines:
        workload = OpenMLBaselineWorkload(setup, pipeline, task_id=openml_task)
        executor.end_to_end_run(workload=workload, root_data=ROOT_DATA_DIRECTORY)

    execution_end = datetime.now()
    elapsed = (execution_end - execution_start).total_seconds()

    print('finished execution in {} seconds'.format(elapsed))
