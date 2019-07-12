import numpy as np
import panda as pd


# simple random method
def random_selection(experiment_objects, rate):
    return np.random.choice(experiment_objects, int(rate * len(experiment_objects)), replace=False)


def sorted_selection(experiment_objects):
    return sorted(experiment_objects, key=lambda eo: 1 - eo.quality, reverse=True)


# return one random item from every bin based equal width based on quality
def histogram_sampling(experiment_objects, bins):
    import random
    sortedObjects = np.array(sorted(experiment_objects, key=lambda eo: 1 - eo.quality, reverse=True))
    data = pd.DataFrame({'Experiment': sortedObjects, 'Accuracy': [1 - eo.quality for eo in sortedObjects]})
    data['bin'] = pd.cut(data['Accuracy'], bins, labels=False)
    size = 1
    replace = False
    fn = lambda obj: obj.loc[np.random.choice(obj.index, size, replace), :]
    history = np.array(data.groupby('bin', as_index=False).apply(fn)['Experiment'])
    random.shuffle(history)
    return history


# return one item from bins of equal size
def count_histogram(experiment_objects, bins):
    import random
    sortedObjects = np.array(sorted(experiment_objects, key=lambda eo: 1 - eo.quality, reverse=True))
    size = len(sortedObjects)
    incr = size / bins
    history = sortedObjects[np.where(np.array(range(size)) % incr == 0)]
    random.shuffle(history)
    return history
