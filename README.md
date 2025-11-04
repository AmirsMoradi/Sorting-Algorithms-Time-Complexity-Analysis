# Time Complexity Analysis of Sorting Algorithms

This project provides **full implementation and analysis** of several classic sorting algorithms in Python.

The project measures **running time**, **number of comparisons**, and **number of permutations** for each algorithm on different input types.

--

## 🚀 Overview

Goals of this project:

* Compare **theoretical and practical** time complexity of popular sorting algorithms.

* Evaluate their behavior on:

* Sorted arrays

* Reverse-sorted arrays

* Random arrays
* Visualize the results with **charts** and **tables**.

---

---

## 📊 Features

* Precise timing with `time.perf_counter()`
* Counting operations (comparisons, shifts)
* Visual comparison using **Matplotlib**
* Handles three types of input:

* Pre-sorted
* Reverse-sorted
* Random

---

## 🧠 Theoretical background

This project includes a short theoretical explanation (big O notation) inside the code comments to help understand why certain algorithms perform better in certain situations.

Example:

```python
# Bubble sort
# Time complexity:
# Best: O(n) when array is already sorted
# Average/worst: O(n^2)
# Space: O(1)
```

---

## 📈 Example output

### 1. Console summary

After running the program, a table with details is printed:

```
Bubble sort: time=0.032 seconds, comparisons=4950, shifts=2475
Quick sort: time=0.001 seconds, comparisons=634, shifts=312
...
```

### 2. Charts

The script automatically generates comparison charts like the following:

* **Runtime vs. Algorithms**
* **Number of Comparisons vs. Algorithms**
* **Number of Shifts vs. Algorithms**

---

## 🧪 How to run

### Requirements

Make sure you have Python 3.8+ and the following libraries installed:

```bash
pip install matplotlib numpy
```

### Run the script

```bash
python algo.py
```

---

## 📘 File structure

```
📂 Sorting-Algorithms-Analysis/
├── algo.py # Main code file
├── README.md # This file
└── plots/ # Auto-generated result plots (optional)
```

---

## 🧮 Summary of results

* **O(n²)** algorithms (Bubble, Insertion, Selection) perform poorly on large data or reverse sorting. data.

* **O(n log n)** algorithms (Merge, Quick, Heap) are significantly faster, especially for random inputs.

* **Quicksort** has the best performance on average, while **Merge Sort** guarantees stable O(n log n) performance.

--

## 🏁 Conclusion

This project demonstrates:

* How sorting algorithms differ in terms of efficiency.

* Why algorithmic complexity matters in real-world data processing.

* The importance of testing algorithms under multiple input conditions.

--

## 👨‍💻 Author's Notes

The code includes handwritten comments (in Persian) for educational purposes that explain the reasoning behind each implementation and observation.

If you found this article useful, please ⭐ check out the repository!

--
