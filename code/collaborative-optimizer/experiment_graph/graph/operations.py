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
    def run(self, this_underlying_data, others_underlying_data):
        """

        :param this_underlying_data: underlying data of the calling node
        :param others_underlying_data: underlying data of the other nodes, if more than one other node is passed, the
               order is the same as the order of the node in others argument of set_other_inputs
        :return:
        """
        pass
