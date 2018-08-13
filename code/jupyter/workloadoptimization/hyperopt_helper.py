from hyperopt import Trials


class TrialConverter:

    def __init__(self):
        pass

    def trialsFromExperimentObjects(self, space, experiment_objects):
        trials = Trials()
        tids = []
        specs = []
        results = []
        miscs = []
        index = 0
        errorCounter = 0
        for e in experiment_objects:
            try:
                doc = e.asTrialDoc(space, index)
                index = index + 1
                tids.append(doc['tid'])
                specs.append(doc['spec'])
                results.append(doc['result'])
                miscs.append(doc['misc'])
            except Exception as error:
                errorCounter = errorCounter + 1

        docs = trials.new_trial_docs(tids, specs, results, miscs)
        for doc in docs:
            doc['state'] = 2

        trials.insert_trial_docs(docs)
        trials.refresh()
        print 'Trial transformation completed, {} errors detected'.format(errorCounter)

        return trials
