"""
Helper methods and classes to make conversion of scripts containing scikit learn models (or similar) to workload easier.
Current approach for fitting scikit learn models is to call 'fit_sk_model' method and pass the scikit learn model. This
creates overhead when modifying scripts. Here, we write some adapter codes, that make the APIs exactly the same as
scikit learn package and make the necessary changes inside.
"""

from sklearn import preprocessing, linear_model, ensemble
import lightgbm as lgb


class SimpleModel:
    def __init__(self, underlying_sk_model):
        self.underlying_sk_model = underlying_sk_model
        self.trained_node = None

    def fit(self, data):
        self.trained_node = data.fit_sk_model(self.underlying_sk_model)

    def transform(self, data):
        return self.trained_node.trasform(data)

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)


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


class PredictiveModel:
    def __init__(self, underlying_sk_model):
        self.underlying_sk_model = underlying_sk_model
        self.trained_node = None

    def fit(self, train, labels, custom_args=None):
        self.trained_node = train.fit_sk_model_with_labels(self.underlying_sk_model, labels, custom_args)

    def predict_proba(self, test, custom_args=None):
        return self.trained_node.predict_proba(test, custom_args)

    def feature_importances(self, features):
        self.trained_node.feature_importances(features)


class LogisticRegression(PredictiveModel):
    def __init__(self, **args):
        PredictiveModel.__init__(self, linear_model.LogisticRegression(**args))


class RandomForestClassifier(PredictiveModel):
    def __init__(self, **args):
        PredictiveModel.__init__(self, ensemble.RandomForestClassifier(**args))


class LGBMClassifier(PredictiveModel):
    def __init__(self, **args):
        PredictiveModel.__init__(self, lgb.LGBMClassifier(**args))

    def best_iteration(self):
        return self.trained_node.data().best_iteration_
