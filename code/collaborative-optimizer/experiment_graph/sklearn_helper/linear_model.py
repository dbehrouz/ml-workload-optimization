from sklearn import linear_model

from sklearn_wrappers import PredictiveModel


class LogisticRegression(PredictiveModel):
    def __init__(self, **args):
        PredictiveModel.__init__(self, linear_model.LogisticRegression(**args))
