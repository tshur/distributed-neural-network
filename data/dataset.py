from math import sqrt
from random import choice

class Sample:
    def __init__(self, instance):
        """Create a sample object from a raw instance."""
        self._num_features = len(instance)
        self._features = [float(x) for x in instance[:-1] + [1]]
        self._label = float(instance[-1])

    def normalize(self, means, stdevs):
        """Normalize a Sample given the means and standard deviations for each feature."""
        for i in range(self._num_features - 1):
            self._features[i] = (self._features[i] - means[i]) / stdevs[i]

        return self

    @property
    def num_features(self):
        return self._num_features

    @property
    def features(self):
        return self._features

    @property
    def label(self):
        return self._label

def get_stats(samples):
    """Compute the statistics (mean and stdev) for each feature in a list of samples."""
    num_stats = samples[0].num_features - 1
    means = [0] * num_stats
    stdevs = [0] * num_stats
    for i in range(num_stats):
        means[i] = sum([sample.features[i] for sample in samples]) / len(samples)
        stdevs[i] = sqrt(sum([(sample.features[i] - means[i]) ** 2 for sample in samples])
                         / len(samples))

    return means, stdevs

def load_data(filename):
    """Load the energy dataset and normalize the samples."""
    with open(filename, 'r') as fp:
        samples = [Sample(line.strip().split(',')) for line in fp.readlines() if line]

    means, stdevs = get_stats(samples)
    samples = [sample.normalize(means, stdevs) for sample in samples]

    return samples

def divide_dataset(samples, train_r=0.65, validation_r=0.15, test_r=0.2):
    """Split the dataset into (train, valid, test) according to the given percentages."""
    return (
        [choice(samples) for _ in range(int(train_r * len(samples)))],
        [choice(samples) for _ in range(int(validation_r * len(samples)))],
        [choice(samples) for _ in range(int(test_r * len(samples)))]
    )