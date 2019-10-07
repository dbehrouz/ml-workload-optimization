from importlib import import_module


class Parser:
    def __init__(self, args):
        self.key_val = dict()
        self.parse(args)

    def has(self, key):
        return key in self.key_val

    def parse(self, arguments):
        for arg in arguments:
            if '=' in arg:
                split = arg.split('=')
                if len(split) != 2:
                    raise Exception('Invalid argument: {}'.format(arg))
                self.key_val[split[0]] = split[1]

    def get(self, key, default=None):
        if default is None:
            return self.key_val[key]
        else:
            return self.key_val.get(key, default)


class ExperimentWorkloadFactory:
    def __init__(self):
        pass

    @staticmethod
    def get_workload(experiment_name, method, workload_name):
        """
        This is a hackish solution for getting the workload objects.
        A workload class with the name test_workload for test_experiment and baseline method is always in
        workload_obj = paper.test_experiment.baseline.test_workload.test_workload()
        :param experiment_name:
        :param method:
        :param workload_name:
        :return:
        """
        module = import_module(
            'paper.experiment_workloads.{}.{}.{}'.format(experiment_name, method, workload_name))
        return getattr(module, workload_name)()
