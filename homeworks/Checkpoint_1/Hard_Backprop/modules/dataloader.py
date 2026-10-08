import numpy as np


class DataLoader(object):
    def __init__(self, X, y, batch_size=1, shuffle=False):
        assert X.shape[0] == y.shape[0]
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.batch_id = 0

    def __len__(self) -> int:
        return int(np.ceil(self.X.shape[0] / self.batch_size))

    def num_samples(self) -> int:
        return self.X.shape[0]

    def __iter__(self):
        if self.shuffle:
            perm = np.random.permutation(self.X.shape[0])
            self.X = self.X[perm]
            self.y = self.y[perm]
        self.batch_id = 0
        return self

    def __next__(self):
        if self.batch_id >= len(self):
            raise StopIteration
        start = self.batch_id * self.batch_size
        end = min(start + self.batch_size, self.X.shape[0])
        self.batch_id += 1
        return self.X[start:end], self.y[start:end]
