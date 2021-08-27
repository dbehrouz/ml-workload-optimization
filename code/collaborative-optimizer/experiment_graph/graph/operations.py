from abc import abstractmethod, ABC


class UserDefinedFunction(ABC):
    def __init__(self, return_type):
        self.return_type = return_type

    @abstractmethod
    def run(self, underlying_data):
        pass
