AS_KB = 1024.0

from abc import abstractmethod


class Pandas(object):
    def __init__(self):
        self.size = None

    @abstractmethod
    def get_size(self):
        """
        computes the size (if not computed already) and returns it
        :return:
        """
        raise Exception('{} class cannot be instantiated'.format(self.__class__.__name__))

    @abstractmethod
    def get_column(self):
        """
        return the list of the columns under the Pandas object
        :return:
        """
        raise Exception('{} class cannot be instantiated'.format(self.__class__.__name__))

    @abstractmethod
    def get_column_hash(self):
        """
        return the list of the column hashes under the Pandas object
        :return:
        """
        raise Exception('{} class cannot be instantiated'.format(self.__class__.__name__))

    @abstractmethod
    def get_data(self):
        """
        return the actual data
        :return:
        """
        raise Exception('{} class cannot be instantiated'.format(self.__class__.__name__))


class DataFrame(Pandas):
    def __init__(self, column_names, column_hashes, pandas_df=None):
        super(DataFrame, self).__init__()
        self.column_names = column_names
        self.column_hashes = column_hashes
        self.pandas_df = pandas_df

    # TODO check if get_size is always going to be called for every object, if it is we can move the computation code to
    # the constructor
    def get_size(self):
        if self.size is None:
            self.size = sum(self.pandas_df.memory_usage(index=True, deep=True)) / AS_KB
        return self.size

    def get_column(self):
        return self.column_names

    def get_column_hash(self):
        return self.column_hashes

    def get_data(self):
        return self.pandas_df


class DataSeries(Pandas):
    def __init__(self, column_name, column_hash, pandas_series):
        super(DataSeries, self).__init__()
        self.column_name = column_name
        self.column_hash = column_hash
        self.pandas_series = pandas_series

    def get_size(self):
        if self.size is None:
            self.size = self.pandas_series.memory_usage(index=True, deep=True) / AS_KB
        return self.size

    def get_column(self):
        return self.column_name

    def get_column_hash(self):
        return self.column_hash

    def get_data(self):
        return self.pandas_series
