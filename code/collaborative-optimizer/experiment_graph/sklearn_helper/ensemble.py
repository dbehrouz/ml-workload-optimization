from sklearn import ensemble

from sklearn_wrappers import PredictiveModel


class RandomForestClassifier(PredictiveModel):
    def __init__(self, **args):
        PredictiveModel.__init__(self, ensemble.RandomForestClassifier(**args))
