from abc import abstractmethod, ABC


class UserDefinedFunction(ABC):
    def __init__(self, return_type):
        self.return_type = return_type

    @abstractmethod
    def run(self, underlying_data):
        """
        For multi input operations, underlying_data is a list of underlying_data coming from every input. Order
        is the same as the order passed to the run_udf function.
        :param underlying_data:
        :return:
        """
        pass

    def __repr__(self):
        return self.__class__.__module__ + '.' + self.__class__.__qualname__ + str(self.__dict__)
