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
        pass


class DataFrame(Pandas):
    def __init__(self, column_names, column_hashes, pandas_df):
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
