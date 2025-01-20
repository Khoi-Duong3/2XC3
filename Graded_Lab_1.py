import random
import time
import timeit 
import matplotlib.pyplot as plt
import numpy as np
import math
from typing import List

# Utitilty functions - some are implemented, others you must implement yourself.

# function to plot the bar graph and average runtimes of N trials
# Please note that this function only plots the graph and does not save it
# To save the graphs you must use plot.save(). Refer to matplotlib documentation
def draw_plot(run_arr, mean, title):
    x = np.arange(0, len(run_arr),1)
    fig=plt.figure(figsize=(20,8))
    plt.bar(x, run_arr, color="blue", alpha=0.7, label="Run times")
    plt.axhline(mean,color="red",linestyle="--",label="Avg")
    plt.ticklabel_format(style='plain', axis='y')
    plt.xlabel("Iterations")
    plt.ylabel("Run time in ms (milisections) order of 1e-3 seconds")
    plt.title(title)
    plt.tight_layout()
    plt.show()

# function to generate random list 
# @args : length = number of items 
#       : max_value maximum value
def create_random_list(length, max_value, item=None, item_index=None):
    random_list = [random.randint(0,max_value) for i in range(length)]
    if item!= None:
        random_list.insert(item_index,item)

    return random_list

# function to generate reversed list of a given size and with a given maximum value
def create_reverse_list(length, max_value, item=None, item_index=None):
    reversed_list = []

    #include your code here
    for i in range (length):
        random_num = random.randint(1, max_value)
        reversed_list.append(random_num)

    reversed_list.sort()
    reversed_list.reverse()
    return reversed_list

# function to generate near sorted list of a given size and with a given maximum value
def create_near_sorted_list(length, max_value, item=None, item_index=None):
    near_sorted_list = []

    #include your code here
    for i in range ((max_value - length), max_value):
        near_sorted_list.append(i)
    
    for j in range (len(near_sorted_list)/8):
        random_item = random.randint(0, len(near_sorted_list)-1)
        near_sorted_list[j], near_sorted_list[random_item] = near_sorted_list[random_item], near_sorted_list[j]

    return near_sorted_list

# function to generate near sorted list of a given size and with a given maximum value
def reduced_unique_list(length, max_value, item=None, item_index=None):

    #include your code here

    reduced_list = []
    random_list = []
    exist = set()
    for _ in range (length):
        random_list.append(random.randint(0, max_value))
    
    for num in random_list:
        if num not in exist:
            reduced_list.append(num)
            exist.add(num)

    return reduced_list

def experiment_D_plot (run_times, mean_times, title, filename):
    lengths = [500,5000,10000,20000,50000]
    x_labels = [f"{l}" for l in lengths]

    iterations = 80
    x = np.arange(iterations)

    fig, ax = plt.subplots(figsize=(20,8))

    for i, size in enumerate(lengths):
        heights = [run_times[size][j] for j in range(iterations)]
        ax.bar(x + i * 0.15, heights, width=0.15, label=f"size{size}")

    for i, size in enumerate(lengths):
        ax.axhline(mean_times[size], color=f"C{i}", linestyle="--", label=f"Mean (Size {size})")

    ax.set_xticks(x + 0.3)
    ax.set_xticklabels([f"Iteration {i}" for i in range(iterations)])
    plt.ticklabel_format(style='plain', axis='y')
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Run time in ms")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()

    # Save the plot before showing
    plt.savefig(filename)
    plt.show()


# Implementation of sorting algorithms
class BubbleSort:
    def __init__(self, items_to_sort):
        self.items = items_to_sort
        self.sorted_items=[]

       ### your implementation for bubble sort goes here 

        self.sorted_items = self.items[:]

        def bubbleSort(items_to_sort):

            for i in range (len(items_to_sort)):
                for j in range (0, len(items_to_sort) - i - 1):
                    if items_to_sort[j] > items_to_sort[j + 1]:
                        items_to_sort[j], items_to_sort[j + 1] = items_to_sort[j + 1], items_to_sort[j]

            return
        
        bubbleSort(self.sorted_items)

    def get_sorted(self,):
        return self.sorted_items
    
class InsertionSort:
    def __init__(self, items_to_sort):
        self.items = items_to_sort
        self.sorted_items=[]

        self.sorted_items = self.items[:]

       ### your implementation for insertion sort goes here
        def insertionSort(items_to_sort):
            for i in range (1, len(items_to_sort)):
                j = i
                while j > 0 and (items_to_sort[j-1] > items_to_sort[j]):
                    items_to_sort[j-1], items_to_sort[j] = items_to_sort[j], items_to_sort[j-1]
                    j = j - 1
            return
        
        insertionSort(self.sorted_items)

    def get_sorted(self,):
        return self.sorted_items
    
class SelectionSort:
    def __init__(self, items_to_sort):
        self.items = items_to_sort
        self.sorted_items=[]


        self.sorted_items = self.items[:]

       ### your implementation for selection sort goes here
        def selectionSort(items_to_sort):

            for i in range (len(items_to_sort) - 1):
                min = i
                for j in range (i + 1, len(items_to_sort)):
                    if items_to_sort[j] < items_to_sort[min]:
                        min = j
                items_to_sort[i], items_to_sort[min] = items_to_sort[min], items_to_sort[i]

            return
        
        selectionSort(self.sorted_items)

    def get_sorted(self,):
        return self.sorted_items
    
class MergeSort:
    def __init__(self, items_to_sort):
    
        self.items = items_to_sort
        self.sorted_items = self.mergeSort(self.items)

       ### your implementation for selection sort goes here

    def merge(self, arr1, arr2):
        aux = []
        l = 0
        r = 0

        while l < len(arr1) and r < len(arr2):
            if arr1[l] > arr2[r]:
                aux.append(arr2[r])
                r += 1
            else:
                aux.append(arr1[l])
                l += 1

        while l < len(arr1):
            aux.append(arr1[l])
            l += 1
            
        while r < len(arr2):
            aux.append(arr2[r])
            r += 1

        return aux
        
    def mergeSort(self, arr):
        if (len(arr) <= 1):
            return arr
            
        mid = len(arr) // 2
        arr1 = arr[:mid]
        arr2 = arr[mid:] 

        sortArr1 = self.mergeSort(arr1)
        sortArr2 = self.mergeSort(arr2)

        return self.merge(sortArr1, sortArr2)
        

    def get_sorted(self,):
        return self.sorted_items
    
class QuickSort:
    def __init__(self, items_to_sort):
        start = 0
        end = len(items_to_sort) - 1
        self.items = items_to_sort
        self.sorted_items= self.quickSort(self.items, start, end)

       ### your implementation for selection sort goes here 

    def partition(self, items_to_sort, start, end):

        pivot = items_to_sort[end]
        j = start
        i = start - 1

        for j in range (start, end):
            if items_to_sort[j] < pivot:
                i += 1
                temp = items_to_sort[i]
                items_to_sort[i] = items_to_sort[j]
                items_to_sort[j] = temp
            
        i += 1
        temp = items_to_sort[i]
        items_to_sort[i] = items_to_sort[end]
        items_to_sort[end] = temp

        return i
        
    def quickSort(self, arr, start, end):
        if start >= end:
            return arr
        pivot = self.partition(arr, start, end)
        self.quickSort(arr,start, pivot - 1)
        self.quickSort(arr, pivot + 1, end)

        return arr


    def get_sorted(self,):
        return self.sorted_items

# test all algorithm implementations
test_case = [10,9,8,7,6,5,4,3,2,1]   
print("Test case array input: ",test_case)

#example run for QuickSort
bubble_sort = BubbleSort(test_case)
insertion_sort = InsertionSort(test_case)
selection_sort = SelectionSort(test_case)
merge_sort = MergeSort(test_case)
quick_sort = QuickSort(test_case)
print("After sorting by Bubble Sort:", bubble_sort.get_sorted())
print("After sorting by Insersion Sort:", insertion_sort.get_sorted())
print("After sorting by Selection Sort:", selection_sort.get_sorted())
print("After sorting by Merge Sort:", merge_sort.get_sorted())
print("After sorting by Quick Sort: ",quick_sort.get_sorted())

# run all algorithms
def experiment_A():
    
    # Insert your code for experiment A design here 

    random_list = create_random_list(10000,100000)
    N = 80
    bubble_run_times = []
    insertion_run_times = []
    selection_run_times = []
    quick_run_times = []
    merge_run_times = []
    for _ in range (N):
        start_time = timeit.default_timer()

        _ = BubbleSort(random_list[:])

        end_time = timeit.default_timer()

        run_time = (end_time - start_time) * 1e3
        bubble_run_times.append(run_time)
    
    for _ in range (N):
        start_time = timeit.default_timer()

        _ = InsertionSort(random_list[:])

        end_time = timeit.default_timer()

        run_time = (end_time - start_time) * 1e3
        insertion_run_times.append(run_time)
    
    for _ in range (N):
        start_time = timeit.default_timer()

        _ = SelectionSort(random_list[:])

        end_time = timeit.default_timer()

        run_time = (end_time - start_time) * 1e3
        selection_run_times.append(run_time)
    
    for _ in range (N):
        start_time = timeit.default_timer()

        _ = QuickSort(random_list[:])

        end_time = timeit.default_timer()

        run_time = (end_time - start_time) * 1e3
        quick_run_times.append(run_time)
    
    for _ in range (N):
        start_time = timeit.default_timer()

        _ = MergeSort(random_list[:])

        end_time = timeit.default_timer()

        run_time = (end_time - start_time) * 1e3
        merge_run_times.append(run_time)

    print(bubble_run_times)
    print(insertion_run_times)
    print(selection_run_times)
    print(quick_run_times)
    print(merge_run_times)

    bubble_mean = np.mean(bubble_run_times)
    insertion_mean = np.mean(insertion_run_times)
    selection_mean = np.mean(selection_run_times)
    quick_mean = np.mean(quick_run_times)
    merge_mean = np.mean(merge_run_times)

    draw_plot(bubble_run_times, bubble_mean, "Experiment A: BubbleSort")
    draw_plot(insertion_run_times, insertion_mean, "Experiement A: InsertionSort")
    draw_plot(selection_run_times, selection_mean, "Experiment A: SelectionSort")
    draw_plot(quick_run_times, quick_mean, "Experiment A: QuickSort")
    draw_plot(merge_run_times, merge_mean, "Experiment A: MergeSort")

    return 0

def experiment_B():
    
    # Insert your code for experiment B design here 

    near_sorted_list = create_near_sorted_list(5000,100000)
    N = 100

    bubble_run_times = []
    insertion_run_times = []
    selection_run_times = []
    quick_run_times = []
    merge_run_times = []

    for _ in range (N):
        start_time = timeit.defaul_timer()

        _ = BubbleSort(near_sorted_list[:])

        end_time = timeit.default_timer()
        run_time = (end_time - start_time) * 1e3
        bubble_run_times.append(run_time)

    for _ in range (N):
        start_time = timeit.defaul_timer()

        _ = InsertionSort(near_sorted_list[:])

        end_time = timeit.default_timer()
        run_time = (end_time - start_time) * 1e3
        insertion_run_times.append(run_time)
    
    for _ in range (N):
        start_time = timeit.defaul_timer()

        _ = SelectionSort(near_sorted_list[:])

        end_time = timeit.default_timer()
        run_time = (end_time - start_time) * 1e3
        selection_run_times.append(run_time)
    
    for _ in range (N):
        start_time = timeit.defaul_timer()

        _ = QuickSort(near_sorted_list[:])

        end_time = timeit.default_timer()
        run_time = (end_time - start_time) * 1e3
        quick_run_times.append(run_time)
    
    for _ in range (N):
        start_time = timeit.defaul_timer()

        _ = MergeSort(near_sorted_list[:])

        end_time = timeit.default_timer()
        run_time = (end_time - start_time) * 1e3
        merge_run_times.append(run_time)

    print(bubble_run_times)
    print(insertion_run_times)
    print(selection_run_times)
    print(quick_run_times)
    print(merge_run_times)

    bubble_mean = np.mean(bubble_run_times)
    insertion_mean = np.mean(insertion_run_times)
    selection_mean = np.mean(selection_run_times)
    quick_mean = np.mean(quick_run_times)
    merge_mean = np.mean(merge_run_times)

    draw_plot(bubble_run_times, bubble_mean, "Experiment B: BubbleSort")
    draw_plot(insertion_run_times, insertion_mean, "Experiment B: InsertionSort")
    draw_plot(selection_run_times, selection_mean, "Experiment B: SelectionSort")
    draw_plot(quick_run_times, quick_mean, "Experiment B: QuickSort")
    draw_plot(merge_run_times, merge_mean, "Experiment B: MergeSort")

    return 0

def experiment_C():
    
    # Insert your code for experiment C design here 
    reversed_list = create_reverse_list(10000, 100000)
    N = 100

    bubble_run_times = []
    insertion_run_times = []
    selection_run_times = []
    quick_run_times = []
    merge_run_times = []

    for _ in range (N):
        start_time = timeit.default_timer()

        _ = BubbleSort(reversed_list[:])

        end_time = timeit.default_timer()
        run_time = (end_time - start_time) * 1e3
        bubble_run_times.append(run_time)
    
    for _ in range (N):
        start_time = timeit.default_timer()

        _ = InsertionSort(reversed_list[:])

        end_time = timeit.default_timer()
        run_time = (end_time - start_time) * 1e3
        insertion_run_times.append(run_time)

    for _ in range (N):
        start_time = timeit.default_timer()

        _ = SelectionSort(reversed_list[:])

        end_time = timeit.default_timer()
        run_time = (end_time - start_time) * 1e3
        selection_run_times.append(run_time)

    for _ in range (N):
        start_time = timeit.default_timer()

        _ = QuickSort(reversed_list[:])

        end_time = timeit.default_timer()
        run_time = (end_time - start_time) * 1e3
        quick_run_times.append(run_time)

    for _ in range (N):
        start_time = timeit.default_timer()

        _ = MergeSort(reversed_list[:])

        end_time = timeit.default_timer()
        run_time = (end_time - start_time) * 1e3
        merge_run_times.append(run_time)

    print(bubble_run_times)
    print(insertion_run_times)
    print(selection_run_times)
    print(quick_run_times)
    print(merge_run_times)

    bubble_mean = np.mean(bubble_run_times)
    insertion_mean = np.mean(insertion_run_times)
    selection_mean = np.mean(selection_run_times)
    quick_mean = np.mean(quick_run_times)
    merge_mean = np.mean(merge_run_times)

    draw_plot(bubble_run_times, bubble_mean, "Experiment C: BubbleSort")
    draw_plot(insertion_run_times, insertion_mean, "Experiment C: InsertionSort")
    draw_plot(selection_run_times, selection_mean, "Experiment C: SelectionSort")
    draw_plot(quick_run_times, quick_mean, "Experiment C: QuickSort")
    draw_plot(merge_run_times, merge_mean, "Experiment C: MergeSort")

    return 0

def experiment_D():
    
    # Insert your code for experiment D design here
    random_500 = create_random_list(500, 100000)
    random_5000 = create_random_list(5000, 100000)
    random_10000 =  create_random_list(10000, 100000)
    random_20000 = create_random_list(20000, 100000)
    random_50000 = create_random_list(50000, 100000)
    N = 80

    bubble_run_times_500 = []
    bubble_run_times_5000 = []
    bubble_run_times_10000 = []
    bubble_run_times_20000 = []
    bubble_run_times_50000 = []

    insertion_run_times_500 = []
    insertion_run_times_5000 = []
    insertion_run_times_10000 = []
    insertion_run_times_20000 = []
    insertion_run_times_50000 = []

    selection_run_times_500 = []
    selection_run_times_5000 = []
    selection_run_times_10000 = []
    selection_run_times_20000 = []
    selection_run_times_50000 = []

    quick_run_times_500 = []
    quick_run_times_5000 = []
    quick_run_times_10000 = []
    quick_run_times_20000 = []
    quick_run_times_50000 = []

    merge_run_times_500 = []
    merge_run_times_5000 = []
    merge_run_times_10000 = []
    merge_run_times_20000 = []
    merge_run_times_50000 = []

    bubble_start = timeit.default_timer()
    for i in range (N):
        start_time500 = timeit.default_timer()
        _ = BubbleSort(random_500[:])
        end_time500 = timeit.default_timer()
        run_time_500 = (end_time500 - start_time500) * 1e3
        bubble_run_times_500.append(run_time_500)

        print("BubbleSort length 500 iteration: ", i)

        start_time5000 = timeit.default_timer()
        _ = BubbleSort(random_5000[:])
        end_time5000 = timeit.default_timer()
        run_time_5000 = (end_time5000 - start_time5000) * 1e3
        bubble_run_times_5000.append(run_time_5000)

        print("BubbleSort length 5000 iteration: ", i)
        
        start_time10000 = timeit.default_timer()
        _ = BubbleSort(random_10000[:])
        end_time10000 = timeit.default_timer()
        run_time_10000 = (end_time10000 - start_time10000) * 1e3
        bubble_run_times_10000.append(run_time_10000)

        print("BubbleSort length 10000 iteration: ", i)

        start_time20000 = timeit.default_timer()
        _ = BubbleSort(random_20000[:])
        end_time20000 = timeit.default_timer()
        run_time_20000 = (end_time20000 - start_time20000) * 1e3
        bubble_run_times_20000.append(run_time_20000)

        print("BubbleSort length 20000 iteration: ", i)

        start_time50000 = timeit.default_timer()
        _ = BubbleSort(random_500[:])
        end_time50000 = timeit.default_timer()
        run_time_50000 = (end_time50000 - start_time50000) * 1e3
        bubble_run_times_50000.append(run_time_50000)

        print("BubbleSort length 50000 iteration: ", i)
    
    bubble_end = timeit.default_timer()
    bubble_time = (bubble_end - bubble_start)
    
    insertion_start = timeit.default()
    for i in range (N):
        start_time500 = timeit.default_timer()
        _ = InsertionSort(random_500[:])
        end_time500 = timeit.default_timer()
        run_time_500 = (end_time500 - start_time500) * 1e3
        insertion_run_times_500.append(run_time_500)

        print("InsertionSort length 500 iteration: ", i)

        start_time5000 = timeit.default_timer()
        _ = InsertionSort(random_5000[:])
        end_time5000 = timeit.default_timer()
        run_time_5000 = (end_time5000 - start_time5000) * 1e3
        insertion_run_times_5000.append(run_time_5000)

        print("InsertionSort length 5000 iteration: ", i)
        
        start_time10000 = timeit.default_timer()
        _ = InsertionSort(random_10000[:])
        end_time10000 = timeit.default_timer()
        run_time_10000 = (end_time10000 - start_time10000) * 1e3
        insertion_run_times_10000.append(run_time_10000)

        print("InsertionSort length 10000 iteration: ", i)

        start_time20000 = timeit.default_timer()
        _ = InsertionSort(random_20000[:])
        end_time20000 = timeit.default_timer()
        run_time_20000 = (end_time20000 - start_time20000) * 1e3
        insertion_run_times_20000.append(run_time_20000)

        print("InsertionSort length 20000 iteration: ", i)

        start_time50000 = timeit.default_timer()
        _ = InsertionSort(random_50000[:])
        end_time50000 = timeit.default_timer()
        run_time_50000 = (end_time50000 - start_time50000) * 1e3
        insertion_run_times_50000.append(run_time_50000)

        print("InsertionSort length 50000 iteration: ", i)
    insertion_end = timeit.default_timer()
    insertion_time = (insertion_end - insertion_start)

    selection_start = timeit.default_timer()
    for i in range (N):
        start_time500 = timeit.default_timer()
        _ = SelectionSort(random_500[:])
        end_time500 = timeit.default_timer()
        run_time_500 = (end_time500 - start_time500) * 1e3
        selection_run_times_500.append(run_time_500)

        print("SelectionSort length 500 iteration: ", i)

        start_time5000 = timeit.default_timer()
        _ = SelectionSort(random_5000[:])
        end_time5000 = timeit.default_timer()
        run_time_5000 = (end_time5000 - start_time5000) * 1e3
        selection_run_times_5000.append(run_time_5000)

        print("SelectionSort length 5000 iteration: ", i)
        
        start_time10000 = timeit.default_timer()
        _ = SelectionSort(random_10000[:])
        end_time10000 = timeit.default_timer()
        run_time_10000 = (end_time10000 - start_time10000) * 1e3
        selection_run_times_10000.append(run_time_10000)

        print("SelectionSort length 10000 iteration: ", i)

        start_time20000 = timeit.default_timer()
        _ = SelectionSort(random_20000[:])
        end_time20000 = timeit.default_timer()
        run_time_20000 = (end_time20000 - start_time20000) * 1e3
        selection_run_times_20000.append(run_time_20000)

        print("SelectionSort length 20000 iteration: ", i)

        start_time50000 = timeit.default_timer()
        _ = SelectionSort(random_500[:])
        end_time50000 = timeit.default_timer()
        run_time_50000 = (end_time50000 - start_time50000) * 1e3
        selection_run_times_50000.append(run_time_50000)

        print("SelectionSort length 50000 iteration: ", i)
    selection_end = timeit.default_timer()
    selection_time = (selection_end - selection_start)

    quick_start = timeit.default_timer()
    for i in range (N):
        start_time500 = timeit.default_timer()
        _ = QuickSort(random_500[:])
        end_time500 = timeit.default_timer()
        run_time_500 = (end_time500 - start_time500) * 1e3
        quick_run_times_500.append(run_time_500)

        print("QuickSort length 500 iteration: ", i)

        start_time5000 = timeit.default_timer()
        _ = QuickSort(random_5000[:])
        end_time5000 = timeit.default_timer()
        run_time_5000 = (end_time5000 - start_time5000) * 1e3
        quick_run_times_5000.append(run_time_5000)

        print("QuickSort length 5000 iteration: ", i)
        
        start_time10000 = timeit.default_timer()
        _ = QuickSort(random_10000[:])
        end_time10000 = timeit.default_timer()
        run_time_10000 = (end_time10000 - start_time10000) * 1e3
        quick_run_times_10000.append(run_time_10000)

        print("QuickSort length 10000 iteration: ", i)

        start_time20000 = timeit.default_timer()
        _ = QuickSort(random_20000[:])
        end_time20000 = timeit.default_timer()
        run_time_20000 = (end_time20000 - start_time20000) * 1e3
        quick_run_times_20000.append(run_time_20000)

        print("QuickSort length 20000 iteration: ", i)

        start_time50000 = timeit.default_timer()
        _ = QuickSort(random_500[:])
        end_time50000 = timeit.default_timer()
        run_time_50000 = (end_time50000 - start_time50000) * 1e3
        quick_run_times_50000.append(run_time_50000)

        print("QuickSort length 50000 iteration: ", i)
    quick_end = timeit.default_timer()
    quick_time = (quick_end - quick_start)

    merge_start = timeit.default_timer()
    for i in range (N):
        start_time500 = timeit.default_timer()
        _ = MergeSort(random_500[:])
        end_time500 = timeit.default_timer()
        run_time_500 = (end_time500 - start_time500) * 1e3
        merge_run_times_500.append(run_time_500)

        print("MergeSort length 500 iteration: ", i)

        start_time5000 = timeit.default_timer()
        _ = MergeSort(random_5000[:])
        end_time5000 = timeit.default_timer()
        run_time_5000 = (end_time5000 - start_time5000) * 1e3
        merge_run_times_5000.append(run_time_5000)

        print("MergeSort length 5000 iteration: ", i)
        
        start_time10000 = timeit.default_timer()
        _ = MergeSort(random_10000[:])
        end_time10000 = timeit.default_timer()
        run_time_10000 = (end_time10000 - start_time10000) * 1e3
        merge_run_times_10000.append(run_time_10000)

        print("MergeSort length 10000 iteration: ", i)

        start_time20000 = timeit.default_timer()
        _ = MergeSort(random_20000[:])
        end_time20000 = timeit.default_timer()
        run_time_20000 = (end_time20000 - start_time20000) * 1e3
        merge_run_times_20000.append(run_time_20000)

        print("MergeSort length 20000 iteration: ", i)

        start_time50000 = timeit.default_timer()
        _ = MergeSort(random_500[:])
        end_time50000 = timeit.default_timer()
        run_time_50000 = (end_time50000 - start_time50000) * 1e3
        merge_run_times_50000.append(run_time_50000)

        print("MergeSort length 50000 iteration: ", i)
    merge_end = timeit.default_timer()
    merge_time = (merge_end - merge_start)

    bubble_run_times = {
        500: bubble_run_times_500,
        5000: bubble_run_times_5000,
        10000: bubble_run_times_10000,
        20000: bubble_run_times_20000,
        50000: bubble_run_times_50000
    }

    insertion_run_times = {
        500: insertion_run_times_500,
        5000: insertion_run_times_5000,
        10000: insertion_run_times_10000,
        20000: insertion_run_times_20000,
        50000: insertion_run_times_50000
    }

    selection_run_times = {
        500: selection_run_times_500,
        5000: selection_run_times_5000,
        10000: selection_run_times_10000,
        20000: selection_run_times_20000,
        50000: selection_run_times_50000
    }

    quick_run_times = {
        500: quick_run_times_500,
        5000: quick_run_times_5000,
        10000: quick_run_times_10000,
        20000: quick_run_times_20000,
        50000: quick_run_times_50000
    }

    merge_run_times = {
        500: merge_run_times_500,
        5000: merge_run_times_5000,
        10000: merge_run_times_10000,
        20000: merge_run_times_20000,
        50000: merge_run_times_50000
    }

    bubble_mean_500 = np.mean(bubble_run_times_500)
    bubble_mean_5000 = np.mean(bubble_run_times_5000)
    bubble_mean_10000 = np.mean(bubble_run_times_10000)
    bubble_mean_20000 = np.mean(bubble_run_times_20000)
    bubble_mean_50000 = np.mean(bubble_run_times_50000)

    insertion_mean_500 = np.mean(insertion_run_times_500)
    insertion_mean_5000 = np.mean(insertion_run_times_5000)
    insertion_mean_10000 = np.mean(insertion_run_times_10000)
    insertion_mean_20000 = np.mean(insertion_run_times_20000)
    insertion_mean_50000 = np.mean(insertion_run_times_50000)

    selection_mean_500 = np.mean(selection_run_times_500)
    selection_mean_5000 = np.mean(selection_run_times_5000)
    selection_mean_10000 = np.mean(selection_run_times_10000)
    selection_mean_20000 = np.mean(selection_run_times_20000)
    selection_mean_50000 = np.mean(selection_run_times_50000)

    quick_mean_500 = np.mean(quick_run_times_500)
    quick_mean_5000 = np.mean(quick_run_times_5000)
    quick_mean_10000 = np.mean(quick_run_times_10000)
    quick_mean_20000 = np.mean(quick_run_times_20000)
    quick_mean_50000 = np.mean(quick_run_times_50000)

    merge_mean_500 = np.mean(merge_run_times_500)
    merge_mean_5000 = np.mean(merge_run_times_5000)
    merge_mean_10000 = np.mean(merge_run_times_10000)
    merge_mean_20000 = np.mean(merge_run_times_20000)
    merge_mean_50000 = np.mean(merge_run_times_50000)

    bubble_mean_times = {
        500: bubble_mean_500,
        5000: bubble_mean_5000,
        10000: bubble_mean_10000,
        20000: bubble_mean_20000,
        50000: bubble_mean_50000
    }

    selection_mean_times = {
        500: selection_mean_500,
        5000: selection_mean_5000,
        10000: selection_mean_10000,
        20000: selection_mean_20000,
        50000: selection_mean_50000
    }

    insertion_mean_times = {
        500: insertion_mean_500,
        5000: insertion_mean_5000,
        10000: insertion_mean_10000,
        20000: insertion_mean_20000,
        50000: insertion_mean_50000
    }

    quick_mean_times = {
        500: quick_mean_500,
        5000: quick_mean_5000,
        10000: quick_mean_10000,
        20000: quick_mean_20000,
        50000: quick_mean_50000
    }

    merge_mean_times = {
        500: merge_mean_500,
        5000: merge_mean_5000,
        10000: merge_mean_10000,
        20000: merge_mean_20000,
        50000: merge_mean_50000
    }

    print("BubbleSort took: ", bubble_time, " s")
    print("InsertionSort took: ", insertion_time, " s")
    print("SelectionSort took: ", selection_time, " s")
    print("QuickSort took: ", quick_time, " s")
    print("MergeSort took: ", merge_time, " s")

    experiment_D_plot(bubble_run_times, bubble_mean_times, "Experiment D: BubbleSort", "bubble_experiment_D.png")
    experiment_D_plot(insertion_run_times, insertion_mean_times, "Experiment D: InsertionSort", "insertion_experiment_D.png")
    experiment_D_plot(selection_run_times, selection_mean_times, "Experiment D: SelectionSort", "selection_experiment_D.png")
    experiment_D_plot(quick_run_times, quick_mean_times, "Experiment D: QuickSort", "quick_experiment_D.png")
    experiment_D_plot(merge_run_times, merge_mean_times, "Experiment D: MergeSort", "merge_experiment_D.png")


    return 0

def experiment_E():
    
    # Insert your code for experiment E design here 
    reduced_list = reduced_unique_list(5000, 100000)
    N = 100
    
    bubble_run_times = []
    insertion_run_times = []
    selection_run_times = []
    quick_run_times = []
    merge_run_times = []

    
    for _ in range (N):
        start_time = timeit.default_timer()

        _ = BubbleSort(reduced_list[:])

        end_time = timeit.default_timer()
        run_time = (end_time - start_time) * 1e3
        bubble_run_times.append(run_time)

    for _ in range (N):
        start_time = timeit.default_timer()

        _ = InsertionSort(reduced_list[:])

        end_time = timeit.default_timer()
        run_time = (end_time - start_time) * 1e3
        insertion_run_times.append(run_time)

    for _ in range (N):
        start_time = timeit.default_timer()

        _ = SelectionSort(reduced_list[:])

        end_time = timeit.default_timer()
        run_time = (end_time - start_time) * 1e3
        selection_run_times.append(run_time)

    for _ in range (N):
        start_time = timeit.default_timer()

        _ = QuickSort(reduced_list[:])

        end_time = timeit.default_timer()
        run_time = (end_time - start_time) * 1e3
        quick_run_times.append(run_time)

    for _ in range (N):
        start_time = timeit.default_timer()

        _ = MergeSort(reduced_list[:])

        end_time = timeit.default_timer()
        run_time = (end_time - start_time) * 1e3
        merge_run_times.append(run_time)

    print(bubble_run_times)
    print(insertion_run_times)
    print(selection_run_times)
    print(quick_run_times)
    print(merge_run_times)

    bubble_mean = np.mean(bubble_run_times)
    insertion_mean = np.mean(insertion_run_times)
    selection_mean = np.mean(selection_run_times)
    quick_mean = np.mean(quick_run_times)
    merge_mean = np.mean(merge_run_times)

    draw_plot(bubble_run_times, bubble_mean, "Experiment E: BubbleSort")
    draw_plot(insertion_run_times, insertion_mean, "Experiment E: InsertionSort")
    draw_plot(selection_run_times, selection_mean, "Experiment E: SelectionSort")
    draw_plot(quick_run_times, quick_mean, "Experiment E: QuickSort")
    draw_plot(merge_run_times, merge_mean, "Experiment E: MergeSort")

    return 0

# call each experiment
#experiment_A()
#experiment_B()
#experiment_C()
experiment_D()
#experiment_E()


    
