import copy

import pandas as pd


class DataStorage(object):
    """DataStorage class
    Responsible for actual storage of the data.
    Essentially, it is a simple dictionary of column hash and Series.
    This ensures no column are stored more than once
    """

    def __init__(self):
        self.DATA = {}
        self.AS_MB = 1024.0 * 1024.0

    def get_feature(self, c_name, c_hash):
        return copy.deepcopy(pd.Series(self.DATA[c_hash], name=c_name))

    def get_dataset(self, names, hashes):
        cache = []
        assert len(names) == len(hashes)
        for i in range(len(names)):
            cache.append(self.DATA[hashes[i]].rename(names[i]))

        return copy.deepcopy(pd.concat(cache, axis=1))

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

    def store_dataframe(self, column_hashes, dataframe):
        for i in range(len(column_hashes)):
            self.store_column(column_hashes[i], dataframe.iloc[:, i])
