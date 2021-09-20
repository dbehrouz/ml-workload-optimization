from abc import abstractmethod, ABC


class UserDefinedFunction(ABC):
    def __init__(self, return_type):
        self.return_type = return_type

    @abstractmethod
    def run(self, underlying_data):
        pass

    def __repr__(self):
        return self.__class__.__module__ + '.' + self.__class__.__qualname__ + str(self.__dict__)
