from abc import abstractmethod


class Workload:
    def __init__(self):
        pass

    @abstractmethod
    def run(self, **args):
        """

        :rtype: bool returns True when the script successfully finishes
        """
        raise Exception('{} class cannot be instantiated'.format(self.__class__.__name__))
