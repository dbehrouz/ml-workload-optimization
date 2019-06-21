import copy

import pandas as pd
import hashlib


class StorageManager(object):
    """ DataStorage class
        The super class of different Storage Manager classes.
        Responsible for actual storage of the data.
    """

    def __init__(self):
        """
        initializes the data dictionary and the conversion to MB value
        """
        self.DATA = {}
        self.AS_MB = 1024.0 * 1024.0

    def get_column(self, c_name, c_hash):
        """
        returns a feature column, given the hash and renames it to the provided name
        :param c_name: desired name of feature
        :param c_hash: hash of the stored feature
        :return: actual feature column
        """
        raise Exception('{} class cannot be instantiated'.format(self.__class__.__name__))

    def get_dataset(self, names, hashes):
        """
        returns the dataset (dataframe), given the hash and renames the columns to the provided names
        :param names: names of the columns
        :param hashes: hash of the dataframe or the columns
        :return: the dataset (dataframe)
        """
        raise Exception('{} class cannot be instantiated'.format(self.__class__.__name__))

    def get_size(self, column_hashes):
        """
        return the size of the selected features or dataset
        :param column_hashes: hash of the features or dataset
        :return: size of the feature or dataset
        """
        raise Exception('{} class cannot be instantiated'.format(self.__class__.__name__))

    def total_size(self):
        """
        computes and returns the total size of the data stored in the storage manager
        :return: total size of the data inside the storage manager
        """
        raise Exception('{} class cannot be instantiated'.format(self.__class__.__name__))

    def store_column(self, column_hash, panda_series):
        """
        store a column (pandas data series) given the hash
        :param column_hash: input hash
        :param panda_series: pandas data series to the be stored
        """
        raise Exception('{} class cannot be instantiated'.format(self.__class__.__name__))

    def store_dataset(self, column_hashes, dataset):
        """
        store a dataset (pandas dataframe) in the storage given the hash
        :param column_hashes: hash of the columns or the dataset
        :param dataset: input dataset (pandas dataframe)
        """
        raise Exception('{} class cannot be instantiated'.format(self.__class__.__name__))


class DedupedStorageManager(StorageManager):
    """ DedupedStorageManager
        A storage manager that ensures no duplicated column are stored.
        Deduplication is done using the hash of the columns passes to the save functions.
        Essentially, it is a simple dictionary of column hash and Series.
        This ensures no column are stored more than once
    """

    def __init__(self):
        super(DedupedStorageManager, self).__init__()

    def get_column(self, c_name, c_hash):
        return pd.Series(self.DATA[c_hash], name=c_name)

    def get_dataset(self, names, hashes):
        cache = []
        assert len(names) == len(hashes)
        for i in range(len(names)):
            cache.append(self.DATA[hashes[i]].rename(names[i]))

        return pd.concat(cache, axis=1)

    def get_size(self, column_hashes):
        s = 0
        for k in column_hashes:
            s += self.DATA[k + '_size']
        return s

    def total_size(self):
        s = 0
        for k, v in self.DATA.iteritems():
            if k.endswith('_size'):
                s += v
        return s

    def store_column(self, column_hash, panda_series):
        if column_hash in self.DATA.keys():
            'column \'{}\' already exist'.format(column_hash)
        else:
            self.DATA[column_hash] = panda_series
            self.DATA[column_hash + '_size'] = panda_series.memory_usage(index=True, deep=True) / self.AS_MB

    def store_dataset(self, column_hashes, dataset):
        for i in range(len(column_hashes)):
            self.store_column(column_hashes[i], dataset.iloc[:, i])


class NaiveStorageManager(StorageManager):
    """
        Naively store every column or dataset under the given hash
        Since, the user is typically passing an array of hashes (one for every column)
        we concat them and use md5 of the concatenated hashes to store the dataset
    """

    def __init__(self):
        super(NaiveStorageManager, self).__init__()

    def get_column(self, c_name, c_hash):
        return copy.deepcopy(pd.Series(self.DATA[c_hash], name=c_name))

    def get_dataset(self, names, hashes):
        if isinstance(hashes, list):
            if len(hashes) > 1:
                combined_hash = hashlib.md5(''.join(hashes)).hexdigest()
            else:
                # TODO: We should find a solution
                # This will only happen in cases where the result of an operation is feature.
                # e.g., select_dtypes always returns a dataset node. but if the actual operation
                # returns only one column, the hashing the hash would mess up the computation
                # we should find a propoer solution
                # for example, for non-deterministic computations, such as select_dtype we can have the
                # node type decided at execution time rather than build time.
                combined_hash = hashes[0]
        else:
            combined_hash = hashes

        df = self.DATA[combined_hash].copy()
        df.columns = names
        return df

    def get_size(self, column_hashes):
        if isinstance(column_hashes, list):
            if len(column_hashes) > 1:
                combined_hash = hashlib.md5(''.join(column_hashes)).hexdigest()
            else:
                combined_hash = column_hashes[0]
        else:
            combined_hash = column_hashes

        return self.DATA[combined_hash + '_size']

    def total_size(self):
        s = 0
        for k, v in self.DATA.iteritems():
            if k.endswith('_size'):
                s += v
        return s

    def store_column(self, column_hash, panda_series):
        if column_hash in self.DATA.keys():
            print 'column \'{}\' already exist'.format(column_hash)
        else:
            self.DATA[column_hash] = panda_series
            self.DATA[column_hash + '_size'] = panda_series.memory_usage(index=True, deep=True) / self.AS_MB

    def store_dataset(self, column_hashes, dataset):
        if isinstance(column_hashes, list):
            if len(column_hashes) > 1:
                combined_hash = hashlib.md5(''.join(column_hashes)).hexdigest()
            else:
                # TODO: We should find a solution
                # This will only happen in cases where the result of an operation is feature.
                # e.g., select_dtypes always returns a dataset node. but if the actual operation
                # returns only one column, the hashing the hash would mess up the computation
                # we should find a propoer solution
                # for example, for non-deterministic computations, such as select_dtype we can have the
                # node type decided at execution time rather than build time.
                combined_hash = column_hashes[0]
        else:
            combined_hash = column_hashes

        if combined_hash in self.DATA.keys():
            print 'dataset \'{}\' already exist'.format(combined_hash)
        else:
            self.DATA[combined_hash] = dataset
            self.DATA[combined_hash + '_size'] = sum(dataset.memory_usage(index=True, deep=True)) / self.AS_MB
