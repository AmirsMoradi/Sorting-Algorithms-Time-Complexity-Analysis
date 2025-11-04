import random
import time
import pandas as pd
import matplotlib.pyplot as plt
import os

# --------------------------
# ابزار شمارش عملیات الگوریتم
# --------------------------
class OpCounter:
    def __init__(self):
        self.comparisons = 0
        self.swaps = 0
        self.moves = 0  # برای الگوریتم‌های غیر مقایسه‌ای

    def reset(self):
        self.comparisons = 0
        self.swaps = 0
        self.moves = 0

    def total_ops(self):
        # معیار ساده برای جمع عملیات‌ها (جمع مقایسه + جابجایی + حرکت)
        return self.comparisons + self.swaps + self.moves


# --------------------------
# Sorting Algorithms
# --------------------------

def bubble_sort(arr, counter: OpCounter):
    a = arr.copy()
    n = len(a)
    for i in range(n):
        swapped = False
        for j in range(0, n - i - 1):
            counter.comparisons += 1
            if a[j] > a[j + 1]:
                a[j], a[j + 1] = a[j + 1], a[j]
                counter.swaps += 1
                swapped = True
        if not swapped:
            break
    return a


def selection_sort(arr, counter: OpCounter):
    a = arr.copy()
    n = len(a)
    for i in range(n):
        min_idx = i
        for j in range(i + 1, n):
            counter.comparisons += 1
            if a[j] < a[min_idx]:
                min_idx = j
        if min_idx != i:
            a[i], a[min_idx] = a[min_idx], a[i]
            counter.swaps += 1
    return a


def insertion_sort(arr, counter: OpCounter):
    a = arr.copy()
    for i in range(1, len(a)):
        key = a[i]
        j = i - 1
        counter.moves += 1  # خوندن key
        while j >= 0:
            counter.comparisons += 1
            if a[j] > key:
                a[j + 1] = a[j]
                counter.moves += 1
                j -= 1
            else:
                break
        a[j + 1] = key
        counter.moves += 1
    return a


def merge_sort(arr, counter: OpCounter):
    def _merge(left, right):
        result = []
        i = j = 0
        while i < len(left) and j < len(right):
            counter.comparisons += 1
            if left[i] <= right[j]:
                result.append(left[i])
                counter.moves += 1
                i += 1
            else:
                result.append(right[j])
                counter.moves += 1
                j += 1
        result.extend(left[i:])
        result.extend(right[j:])
        counter.moves += len(left) - i + len(right) - j
        return result

    def _ms(a):
        if len(a) <= 1:
            return a
        mid = len(a) // 2
        l = _ms(a[:mid])
        r = _ms(a[mid:])
        return _merge(l, r)

    return _ms(arr.copy())


def quick_sort(arr, counter: OpCounter):
    def _qs(a):
        if len(a) <= 1:
            return a
        pivot = a[len(a) // 2]
        left, mid, right = [], [], []
        for x in a:
            counter.comparisons += 1
            if x < pivot:
                left.append(x)
                counter.moves += 1
            elif x == pivot:
                mid.append(x)
                counter.moves += 1
            else:
                right.append(x)
                counter.moves += 1
        return _qs(left) + mid + _qs(right)

    return _qs(arr.copy())


def heapify(a, n, i, counter: OpCounter):
    largest = i
    l = 2 * i + 1
    r = 2 * i + 2
    if l < n:
        counter.comparisons += 1
        if a[l] > a[largest]:
            largest = l
    if r < n:
        counter.comparisons += 1
        if a[r] > a[largest]:
            largest = r
    if largest != i:
        a[i], a[largest] = a[largest], a[i]
        counter.swaps += 1
        heapify(a, n, largest, counter)


def heap_sort(arr, counter: OpCounter):
    a = arr.copy()
    n = len(a)
    for i in range(n // 2 - 1, -1, -1):
        heapify(a, n, i, counter)
    for i in range(n - 1, 0, -1):
        a[0], a[i] = a[i], a[0]
        counter.swaps += 1
        heapify(a, i, 0, counter)
    return a


def counting_sort(arr, counter: OpCounter):
    a = arr.copy()
    if not a:
        return a
    min_val = min(a)
    max_val = max(a)
    range_len = max_val - min_val + 1
    count = [0] * range_len
    for num in a:
        count[num - min_val] += 1
        counter.moves += 1
    out = []
    for i, c in enumerate(count):
        if c:
            out.extend([i + min_val] * c)
            counter.moves += c
    return out


def radix_sort(arr, counter: OpCounter):
    a = arr.copy()
    if not a:
        return a
    negatives = [x for x in a if x < 0]
    nonneg = [x for x in a if x >= 0]

    def _radix_positive(lst):
        if not lst:
            return []
        max_val = max(lst)
        exp = 1
        out = lst[:]
        while max_val // exp > 0:
            n = len(out)
            output = [0] * n
            count = [0] * 10
            for i in range(n):
                index = (out[i] // exp) % 10
                count[index] += 1
                counter.moves += 1
            for i in range(1, 10):
                count[i] += count[i - 1]
                counter.moves += 1
            for i in range(n - 1, -1, -1):
                index = (out[i] // exp) % 10
                output[count[index] - 1] = out[i]
                count[index] -= 1
                counter.moves += 1
            out = output
            exp *= 10
        return out

    sorted_nonneg = _radix_positive(nonneg)
    neg_abs_sorted = _radix_positive([abs(x) for x in negatives])
    neg_sorted = [-x for x in reversed(neg_abs_sorted)]
    return neg_sorted + sorted_nonneg


def bucket_sort(arr, counter: OpCounter, bucket_size=10):
    a = arr.copy()
    if not a:
        return a
    min_val, max_val = min(a), max(a)
    bucket_count = (max_val - min_val) // bucket_size + 1
    buckets = [[] for _ in range(bucket_count)]
    for num in a:
        idx = (num - min_val) // bucket_size
        buckets[idx].append(num)
        counter.moves += 1
    result = []
    for b in buckets:
        if b:
            tmp_counter = OpCounter()
            sorted_b = insertion_sort(b, tmp_counter)
            result.extend(sorted_b)
            counter.comparisons += tmp_counter.comparisons
            counter.swaps += tmp_counter.swaps
            counter.moves += tmp_counter.moves
    return result


# --------------------------
# Algorithms config
# --------------------------
ALGORITHMS = {
    "Bubble": bubble_sort,
    "Selection": selection_sort,
    "Insertion": insertion_sort,
    "Merge": merge_sort,
    "Quick": quick_sort,
    "Heap": heap_sort,
    "Counting": counting_sort,
    "Radix": radix_sort,
    "Bucket": bucket_sort
}

SIZES = [100, 500, 1000, 2000]
INPUT_TYPES = ["random", "sorted", "reversed"]

# --------------------------
# Benchmark runner
# --------------------------
def run_benchmarks(repeats=3):
    records = []
    for n in SIZES:
        print(f"\n=== n = {n} ===")
        for input_type in INPUT_TYPES:
            print(f"-- Input: {input_type} --")
            for alg_name, alg_func in ALGORITHMS.items():
                times, comps, swaps, moves = [], [], [], []
                for r in range(repeats):
                    if input_type == "random":
                        data = [random.randint(-5000, 5000) for _ in range(n)]
                    elif input_type == "sorted":
                        data = list(range(n))
                    elif input_type == "reversed":
                        data = list(range(n, 0, -1))
                    else:
                        data = [random.randint(0, 10000) for _ in range(n)]

                    counter = OpCounter()
                    start = time.time()
                    alg_func(data, counter)
                    end = time.time()
                    times.append(end - start)
                    comps.append(counter.comparisons)
                    swaps.append(counter.swaps)
                    moves.append(counter.moves)

                rec = {
                    "algorithm": alg_name,
                    "n": n,
                    "input_type": input_type,
                    "time_sec_mean": sum(times) / len(times),
                    "comparisons_mean": sum(comps) // len(comps),
                    "swaps_mean": sum(swaps) // len(swaps),
                    "moves_mean": sum(moves) // len(moves),
                    "total_ops": (sum(comps) + sum(swaps) + sum(moves)) // len(comps)
                }
                records.append(rec)
                print(f"{alg_name:9} | time: {rec['time_sec_mean']:.4f}s | comps: {rec['comparisons_mean']:8} | swaps: {rec['swaps_mean']:7} | moves: {rec['moves_mean']:8}")
    return pd.DataFrame.from_records(records)


# --------------------------
# Theoretical Big-O table
# --------------------------
def print_big_o_table():
    table = [
        ["Bubble", "O(n)", "O(n²)", "O(n²)", "O(1)"],
        ["Selection", "O(n²)", "O(n²)", "O(n²)", "O(1)"],
        ["Insertion", "O(n)", "O(n²)", "O(n²)", "O(1)"],
        ["Merge", "O(n log n)", "O(n log n)", "O(n log n)", "O(n)"],
        ["Quick", "O(n log n)", "O(n log n)", "O(n²)", "O(log n)"],
        ["Heap", "O(n log n)", "O(n log n)", "O(n log n)", "O(1)"],
        ["Counting", "O(n + k)", "O(n + k)", "O(n + k)", "O(k)"],
        ["Radix", "O(nk)", "O(nk)", "O(nk)", "O(n + k)"],
        ["Bucket", "O(n + k)", "O(n + k)", "O(n²)", "O(n)"]
    ]
    df = pd.DataFrame(table, columns=["Algorithm", "Best", "Average", "Worst", "Space"])
    print("\n=== Theoretical Big-O Table ===")
    print(df.to_string(index=False))


# --------------------------
# Plotting & Saving
# --------------------------
def plot_results(df):
    os.makedirs("plots", exist_ok=True)

    for input_type in INPUT_TYPES:
        subset = df[df["input_type"] == input_type]
        plt.figure(figsize=(12, 6))
        for alg in subset["algorithm"].unique():
            s = subset[subset["algorithm"] == alg].sort_values("n")
            plt.plot(s["n"], s["time_sec_mean"], marker='o', label=alg)
        plt.title(f"Average Execution Time — Input: {input_type}")
        plt.xlabel("n (array size)")
        plt.ylabel("Time (seconds)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"plots/time_{input_type}.png", dpi=300)
        plt.show()

    subset = df[df["input_type"] == "random"]
    plt.figure(figsize=(12, 6))
    for alg in subset["algorithm"].unique():
        s = subset[subset["algorithm"] == alg].sort_values("n")
        plt.plot(s["n"], s["total_ops"], marker='o', label=alg)
    plt.title("Total Operations (comparisons + swaps + moves) — Random Input")
    plt.xlabel("n (array size)")
    plt.ylabel("Total operations")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("plots/total_ops_random.png", dpi=300)
    plt.show()


# --------------------------
# Main Execution
# --------------------------
def main():
    print("Starting Sorting Algorithms Benchmark...\n")
    df = run_benchmarks(repeats=3)
    df_sorted = df.sort_values(["n", "input_type", "time_sec_mean"])
    pd.set_option("display.max_rows", None)
    print("\n\n=== Sample of Results ===")
    print(df_sorted.head(50).to_string(index=False))

    os.makedirs("results", exist_ok=True)
    df_sorted.to_csv("results/sorting_benchmarks_results.csv", index=False)
    print("\nResults saved to results/sorting_benchmarks_results.csv")

    print_big_o_table()
    plot_results(df_sorted)


if __name__ == "__main__":
    main()
