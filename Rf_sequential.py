import numpy as np
from collections import Counter
from sklearn.tree import DecisionTreeClassifier

class SequentialRandomForest:
    def __init__(self, n_estimators=10, max_depth=None, random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.trees = []
        self.bootstrap_indices = []

    def fit(self, X, y):
        self.trees = []
        rng = np.random.RandomState(self.random_state)
        n_samples = X.shape[0]

        for i in range(self.n_estimators):
            indices = rng.choice(n_samples, n_samples, replace=True)
            self.bootstrap_indices.append(indices)
            X_sample = X[indices]
            y_sample = y[indices]
            tree = DecisionTreeClassifier(max_depth=self.max_depth, random_state=rng.randint(0, 100000))
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        y_pred = [Counter(tree_preds[:, i]).most_common(1)[0][0] for i in range(X.shape[0])]
        return np.array(y_pred)

    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
