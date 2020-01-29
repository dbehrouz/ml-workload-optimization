from abc import abstractmethod


class Operation(object):
    def __init__(self, name, return_type, params):
        self.name = name
        self.return_type = return_type
        self.params = params

    @abstractmethod
    def run(self, node):
        pass
