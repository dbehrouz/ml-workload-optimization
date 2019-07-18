from sklearn import ensemble

from sklearn_wrappers import PredictiveModel


class RandomForestClassifier(PredictiveModel):
    def __init__(self, should_warmstart=False, **args):
        PredictiveModel.__init__(self, should_warmstart, ensemble.RandomForestClassifier(**args))
