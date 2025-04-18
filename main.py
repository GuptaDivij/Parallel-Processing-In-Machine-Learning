import time
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from Knn_sequential import KNN as SequentialKNN
from Knn_parallel import ParallelKNN
from sklearn.neighbors import KNeighborsClassifier

from Rf_sequential import SequentialRandomForest
from Rf_parallel import ParallelRandomForest
from sklearn.ensemble import RandomForestClassifier

# ---------- KNN ----------
def run_sequential(X_train, X_test, y_train, y_test, k=3):
    model = SequentialKNN(k=k)
    model.fit(X_train, y_train)
    start = time.time()
    y_pred = model.predict(X_test)
    end = time.time()
    acc = accuracy_score(y_test, y_pred)
    return acc, end - start

def run_parallel(X_train, X_test, y_train, y_test, k=3):
    model = ParallelKNN(k=k)
    model.fit(X_train, y_train)
    start = time.time()
    y_pred = model.predict(X_test)
    end = time.time()
    acc = accuracy_score(y_test, y_pred)
    return acc, end - start

def run_sklearn(X_train, X_test, y_train, y_test, k=3):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    start = time.time()
    y_pred = model.predict(X_test)
    end = time.time()
    acc = accuracy_score(y_test, y_pred)
    return acc, end - start

# ---------- RF ----------
def run_rf_sequential(X_train, X_test, y_train, y_test, n_estimators=10):
    model = SequentialRandomForest(n_estimators=n_estimators)
    model.fit(X_train, y_train)
    start = time.time()
    y_pred = model.predict(X_test)
    end = time.time()
    acc = accuracy_score(y_test, y_pred)
    return acc, end - start

def run_rf_parallel(X_train, X_test, y_train, y_test, n_estimators=10):
    model = ParallelRandomForest(n_estimators=n_estimators)
    model.fit(X_train, y_train)
    start = time.time()
    y_pred = model.predict(X_test)
    end = time.time()
    acc = accuracy_score(y_test, y_pred)
    return acc, end - start

def run_rf_sklearn(X_train, X_test, y_train, y_test, n_estimators=100):
    model = RandomForestClassifier(n_estimators=n_estimators, n_jobs=-1)
    model.fit(X_train, y_train)
    start = time.time()
    y_pred = model.predict(X_test)
    end = time.time()
    acc = accuracy_score(y_test, y_pred)
    return acc, end - start

# ---------- Main ----------
def main():
    X, y = make_classification(n_samples=10_000, n_features=20, n_informative=15,
                               n_redundant=5, n_classes=2, random_state=42)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    k = 3
    n_estimators = 10

    print("----- KNN Benchmarks -----")
    acc_seq, time_seq = run_sequential(X_train, X_test, y_train, y_test, k)
    print(f"Sequential KNN -> Accuracy: {acc_seq:.4f}, Time: {time_seq:.4f}s")

    acc_par, time_par = run_parallel(X_train, X_test, y_train, y_test, k)
    print(f"Parallel KNN   -> Accuracy: {acc_par:.4f}, Time: {time_par:.4f}s")

    acc_sk, time_sk = run_sklearn(X_train, X_test, y_train, y_test, k)
    print(f"Sklearn KNN    -> Accuracy: {acc_sk:.4f}, Time: {time_sk:.4f}s\n")

    print("----- Random Forest Benchmarks -----")
    acc_rfs, time_rfs = run_rf_sequential(X_train, X_test, y_train, y_test, n_estimators)
    print(f"Sequential RF  -> Accuracy: {acc_rfs:.4f}, Time: {time_rfs:.4f}s")

    acc_rfp, time_rfp = run_rf_parallel(X_train, X_test, y_train, y_test, n_estimators)
    print(f"Parallel RF    -> Accuracy: {acc_rfp:.4f}, Time: {time_rfp:.4f}s")

    acc_rfk, time_rfk = run_rf_sklearn(X_train, X_test, y_train, y_test, n_estimators=100)
    print(f"Sklearn RF     -> Accuracy: {acc_rfk:.4f}, Time: {time_rfk:.4f}s")

if __name__ == "__main__":
    main()
