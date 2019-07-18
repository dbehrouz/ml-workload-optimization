from sklearn import linear_model

from sklearn_wrappers import PredictiveModel


class LogisticRegression(PredictiveModel):
    def __init__(self, should_warmstart=False, **args):
        PredictiveModel.__init__(self, should_warmstart, linear_model.LogisticRegression(**args))
