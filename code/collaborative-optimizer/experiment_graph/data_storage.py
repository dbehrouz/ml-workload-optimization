from abc import abstractmethod

import pandas

from experiment_graph.graph.auxilary import Pandas, DataFrame, DataSeries
import pandas as pd

AS_KB = 1024.0


class StorageManager(object):
    """ DataStorage class
        The super class of different Storage Manager classes.
        Responsible for actual storage of the data.
    """

    def __init__(self):
        """
        initializes the data dictionary and the conversion to MB value
        """
        self.key_value = {}
        self.key_value_size = {}

    def get(self, key):
        return self.key_value[key]

    @abstractmethod
    def put(self, key, artifact):
        """

        :param key:
        :type artifact: Pandas
        """
        raise Exception('{} class cannot be instantiated'.format(self.__class__.__name__))

    @abstractmethod
    def delete(self, key):
        """

        :param key:
        """
        raise Exception('{} class cannot be instantiated'.format(self.__class__.__name__))

    @abstractmethod
    def total_size(self):
        """
        computes and returns the total size of the data stored in the storage manager
        :return: total size of the data inside the storage manager
        """
        raise Exception('{} class cannot be instantiated'.format(self.__class__.__name__))

    @staticmethod
    def compute_series_size(pandas_series):
        return pandas_series.memory_usage(index=True, deep=True) / AS_KB

    @staticmethod
    def compute_frame_size(pandas_frame):
        return sum(pandas_frame.memory_usage(index=True, deep=True)) / AS_KB


class DedupedStorageManager(StorageManager):
    """ DedupedStorageManager
        A storage manager that ensures no duplicated column are stored.
        Deduplication is done using the hash of the columns passes to the save functions.
        Essentially, it is a simple dictionary of column hash and Series.
        This ensures no column are stored more than once
    """

    def __init__(self):
        """
        initialize the column store.
        A key value store that stores every column using the unique key
        """
        super(DedupedStorageManager, self).__init__()
        self.column_store = {}
        self.column_count = {}
        self.column_size = {}

    def put(self, key, artifact):
        if key not in self.key_value:
            if isinstance(artifact, DataFrame):
                column_hashes = artifact.get_column_hash()
                data = artifact.get_data()
                self.store_dataframe(column_hashes, data)
                self.key_value[key] = column_hashes
                data_frame_size = 0
                for c in column_hashes:
                    data_frame_size += self.column_size[c]
                self.key_value_size[key] = data_frame_size
            elif isinstance(artifact, DataSeries):
                column_hash = artifact.get_column_hash()
                data = artifact.get_data()
                self.store_dataseries(column_hash, data)
                self.key_value[key] = column_hash
                self.key_value_size[key] = self.column_size[column_hash]
        else:
            print 'warning: key exists, abort put!!!'

    def get(self, key):
        column_hashes = self.key_value[key]
        if isinstance(column_hashes, list):  # dataframe
            cache = []
            for i in range(len(column_hashes)):
                cache.append(self.column_store[column_hashes[i]])
            return pd.concat(cache, axis=1)
        elif isinstance(column_hashes, str):  # dataseries
            return pd.Series(self.column_store[column_hashes])

    def delete(self, key):
        column_hashes = self.key_value[key]
        for ch in column_hashes:
            if self.column_count[ch] == 1:
                del self.column_count[ch]
                del self.column_store[ch]
            elif self.column_count[ch] > 1:
                self.column_count[ch] -= 1

    def store_dataseries(self, column_hash, data_series):
        if column_hash in self.column_store.keys():
            self.column_count[column_hash] += 1
        else:
            self.column_store[column_hash] = data_series
            self.column_count[column_hash] = 1
            self.column_size[column_hash] = self.compute_series_size(data_series)

    def store_dataframe(self, column_hashes, dataframe):
        for i in range(len(column_hashes)):
            self.store_dataseries(column_hashes[i], dataframe.iloc[:, i])

    def total_size(self):
        """
        this return the compressed size not the total size of the artifacts
        :return:
        """
        return sum(self.column_size.values())

    def artifacts_total_size(self):
        """
        this method returns the sum of the size of all the artifacts
        :return:
        """
        return sum(self.key_value_size.values())


class SimpleStorageManager(StorageManager):
    """
        Naively store every column or dataset under the given hash
        Since, the user is typically passing an array of hashes (one for every column)
        we concat them and use md5 of the concatenated hashes to store the dataset
    """

    def __init__(self):
        super(SimpleStorageManager, self).__init__()

    def put(self, key, artifact):
        if key not in self.key_value:
            data = artifact.get_data()
            self.key_value[key] = data
            if isinstance(artifact, DataFrame):
                self.key_value_size[key] = self.compute_frame_size(data)
            elif isinstance(artifact, DataSeries):
                self.key_value_size[key] = self.compute_series_size(data)
            else:
                raise Exception('Invalid artifact type: {}'.format(artifact.__class__.__name__))
        else:
            print 'warning: key exists, abort put!!!'

    def delete(self, key):
        del self.key_value[key]
        del self.key_value_size[key]

    def total_size(self):
        return sum(self.key_value_size.values())
