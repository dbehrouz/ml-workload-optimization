from openml import datasets, tasks, runs, flows, setups, config, evaluations
import pandas as pd

class OpenMLReader:
    def getTopRuns(self, numberOfRuns, pipeline, task, evaluationMetric='predictive_accuracy'):
        """ returns the top runs.
        The function returns the top runs sorted by the given metrics.

        Args:
            numberOfRuns (int): number of runs to return for the specific pipeline and task, -1 returns everything
            pipeline (int): id of the openml flow
            task (int): id of the openml task
            evaluationMetric (string): evaluation metric for querying and sorting the runs (default: predictive_accuracy)
        Returns:
            DataFrame: top runs for the given flow and task with the following headers
            'run_id','task_id','flow_id', 'accuracy','setup'
        """
        openMlEvaluations = evaluations.list_evaluations(evaluationMetric, task= [task], flow = [pipeline])
        evaluationData = pd.DataFrame.from_dict(openMlEvaluations, orient='index')
        evaluationData['accuracy'] = evaluationData.apply(lambda eva: eva.values[0].value, axis = 1)
        evaluationData['run_id'] = evaluationData.apply(lambda eva: eva.values[0].run_id, axis = 1)
        # extracting the top ''numberOfRuns' runs
        topRuns = evaluationData.sort_values('accuracy',ascending=False)
        if (numberOfRuns != -1):
            topRuns = topRuns[0:numberOfRuns]
        # retreiving the run objects from the top runs
        openMLRuns = runs.list_runs(task=[task], flow=[pipeline])
        experiments = pd.DataFrame.from_dict(openMLRuns,orient='index')
        Experiment = experiments.merge(topRuns,on='run_id').drop(columns=['uploader',0])
        idx = 0
        dataSize = topRuns.shape[0]
        frames = []
        while idx < topRuns.shape[0]:
            batchEnd = idx + 500
            if batchEnd > dataSize:
                batchEnd = dataSize
            setup_frame = pd.DataFrame.from_dict(setups.list_setups(setup=Experiment.setup_id[idx:batchEnd], size = 500 ),orient='index').reset_index()
            setup_frame.columns=['id', 'setup']
            frames.append(setup_frame)
            idx = idx + 500
        Setup = pd.concat(frames).reset_index(drop=True)
        return pd.merge(Setup, Experiment, how = 'inner', left_on='id', right_on='setup_id').drop(columns = ['id','setup_id'])[['run_id','task_id','flow_id', 'accuracy','setup']]