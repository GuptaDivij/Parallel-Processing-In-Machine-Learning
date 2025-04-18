import numpy as np
from collections import Counter
from multiprocessing import Pool, cpu_count

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2)**2))

def _predict_single(args):
    x, X_train, y_train, k = args
    distances = [euclidean_distance(x, x_train) for x_train in X_train]
    k_indices = np.argsort(distances)[:k]
    k_labels = [y_train[i] for i in k_indices]
    return Counter(k_labels).most_common(1)[0][0]

class ParallelKNN:
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X, processes=None):
        if processes is None:
            processes = cpu_count()

        args = [(x, self.X_train, self.y_train, self.k) for x in X]
        with Pool(processes=processes) as pool:
            return np.array(pool.map(_predict_single, args))

    def score(self, X, y, processes=None):
        y_pred = self.predict(X, processes=processes)
        return np.mean(y_pred == y)
