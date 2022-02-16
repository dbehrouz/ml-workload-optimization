from abc import abstractmethod, ABC


class UserDefinedFunction(ABC):
    def __init__(self, return_type):
        self.return_type = return_type

    @abstractmethod
    def run(self, underlying_data):
        pass

    def __repr__(self):
        return self.__class__.__module__ + '.' + self.__class__.__qualname__ + str(self.__dict__)


class MultiInputUserDefinedFunction(UserDefinedFunction):
    def __init__(self, return_type):
        super().__init__(return_type)
        self.other_inputs = None

    def set_other_inputs(self, others):
        self.other_inputs = others

    @abstractmethod
    def run(self, underlying_data):
        """
        For MultiInputUserDefinedFunction, underlying_data is a list of underlying_data coming from every input. Order
        is preserved.
        :param underlying_data:
        :return:
        """
        pass
