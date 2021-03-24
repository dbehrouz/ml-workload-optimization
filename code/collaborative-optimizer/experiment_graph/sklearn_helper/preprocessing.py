from sklearn import preprocessing

from experiment_graph.sklearn_helper.sklearn_wrappers import SimpleModel


class Imputer(SimpleModel):
    def __init__(self, **args):
        SimpleModel.__init__(self, preprocessing.Imputer(**args))


class PolynomialFeatures(SimpleModel):
    def __init__(self, **args):
        SimpleModel.__init__(self, preprocessing.PolynomialFeatures(**args))

    def get_feature_names(self, input_features):
        return self.trained_node.data().get_feature_names(input_features=input_features)


class LabelEncoder(SimpleModel):
    def __init__(self):
        SimpleModel.__init__(self, preprocessing.LabelEncoder())

    # overriding the base class transform since the method is colling the transform_col method
    def transform(self, data):
        return self.trained_node.transform_col(data)


class MinMaxScaler(SimpleModel):
    def __init__(self, **args):
        SimpleModel.__init__(self, preprocessing.MinMaxScaler(**args))
