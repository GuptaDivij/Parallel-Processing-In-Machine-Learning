# **Parallel vs Sequential ML Algorithms**

This project benchmarks **K-Nearest Neighbors (KNN)** and **Random Forest (RF)** classifiers in **sequential**, **parallel**, and **Scikit-learn** implementations using a synthetic dataset.

---

## 📁 **Project Structure**

1. `Knn_sequential.py` – Sequential KNN implementation  
2. `Knn_parallel.py` – Parallel KNN using multiprocessing  
3. `Rf_sequential.py` – Sequential Random Forest implementation  
4. `Rf_parallel.py` – Parallel Random Forest using multiprocessing  
5. `main.py` – Runs benchmarks and prints accuracy and timing

---

## ⚙️ **Features**

- Custom implementations of KNN and Random Forest (sequential + parallel)  
- Benchmarks include execution time and accuracy  
- Comparisons against Scikit-learn models as baselines  

---

## 📊 **Results**

### 1) **Small Dataset (n = 1,000)**

KNN Benchmarks:
Sequential KNN → Accuracy: 0.8933, Time: 0.4600s
Parallel KNN → Accuracy: 0.8933, Time: 1.2688s
Sklearn KNN → Accuracy: 0.8933, Time: 0.0393s

Random Forest Benchmarks:
Sequential RF → Accuracy: 0.8467, Time: 0.0012s
Parallel RF → Accuracy: 0.8500, Time: 0.0013s
Sklearn RF → Accuracy: 0.9067, Time: 0.0138s


> Parallel versions introduce overhead for small datasets, making them slower than sequential. Scikit-learn models are extremely fast and accurate even at small scale.

---

### 2) **Medium Dataset (n = 10,000)**

KNN Benchmarks:
Sequential KNN → Accuracy: 0.9560, Time: 47.8503s
Parallel KNN → Accuracy: 0.9560, Time: 10.4674s
Sklearn KNN → Accuracy: 0.9560, Time: 0.1006s

Random Forest Benchmarks:
Sequential RF → Accuracy: 0.9110, Time: 0.0084s
Parallel RF → Accuracy: 0.9110, Time: 0.0086s
Sklearn RF → Accuracy: 0.9433, Time: 0.0138s


> Parallel KNN is ~5× faster than sequential with no loss in accuracy. Random Forest performs efficiently in all forms, with Scikit-learn leading slightly in both speed and accuracy.

---

### 3) **Large Dataset (n = 100,000)**

KNN Benchmarks:
Sklearn KNN → Accuracy: 0.9770, Time: 1.0403s

Random Forest Benchmarks:
Sequential RF → Accuracy: 0.9504, Time: 0.0881s
Parallel RF → Accuracy: 0.9495, Time: 0.0889s
Sklearn RF → Accuracy: 0.9725, Time: 0.0651s

Custom KNN implementations became unusable due to excessive runtime and resource consumption. Scikit-learn's KNN executed smoothly and accurately. Random Forest handled the data well across all versions, but Scikit-learn was still the most efficient.

### 4) ** Very Large Dataset (n = 1,000,000)**

Random Forest Benchmarks:
Sequential RF → Accuracy: 0.9692, Time: 1.0405s
Parallel RF → Accuracy: 0.9694, Time: 1.1462s
Sklearn RF → Accuracy: 0.9824, Time: 0.8554s


> Even at this scale, Random Forest models trained and predicted efficiently. However, custom parallelization didn’t show a significant advantage due to the small relative cost of training individual trees. Scikit-learn remained the fastest and most accurate.

---

## 🔍 **Key Findings from Benchmarking**

- **Small Datasets (n = 1,000)**  
  Parallel versions introduce more overhead than benefit. Scikit-learn outperforms custom models in speed with the same accuracy.

- **Medium Datasets (n = 10,000)**  
  Parallel KNN shows a significant speedup (~5x faster) over sequential. All RF variants perform efficiently, with Scikit-learn still slightly ahead.

- **Large Datasets (n = 100,000)**  
  Custom KNN becomes impractical — either hangs or overloads system. Scikit-learn models remain smooth and accurate. RF continues to scale well.

- **Very Large Datasets (n = 1,000,000)**  
  Custom Random Forest models still perform well, but Python multiprocessing adds little benefit. Scikit-learn again offers the best performance overall.

---

## 🏁 **Conclusion**

Scikit-learn models are the most **efficient**, **accurate**, and **scalable** due to their underlying C/C++ optimizations.  
Custom implementations are valuable for learning and small-scale experiments, but show clear limitations with larger datasets, especially for KNN. Parallelization helps with medium-scale data, but is not always beneficial and can introduce unnecessary overhead.
