from sklearn import mixture

from experiment_graph.sklearn_helper.sklearn_wrappers import MixtureModel


class GaussianMixture(MixtureModel):
    def __init__(self, **args):
        MixtureModel.__init__(self, mixture.GaussianMixture(**args))
