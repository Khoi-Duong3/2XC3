import random
import time
import timeit 
import matplotlib.pyplot as plt
import numpy as np
import math
from typing import List
import sys

sys.setrecursionlimit(15000)

# Utitilty functions - some are implemented, others you must implement yourself.

# function to plot the bar graph and average runtimes of N trials
# Please note that this function only plots the graph and does not save it
# To save the graphs you must use plot.save(). Refer to matplotlib documentation
def draw_plot(run_arr, mean, title, filename):
    x = np.arange(1, len(run_arr)+1,1)
    fig=plt.figure(figsize=(20,8))
    plt.bar(x, run_arr, color="blue", alpha=0.7, label="Run times")
    plt.axhline(mean,color="red",linestyle="--",label="Avg")
    plt.ticklabel_format(style='plain', axis='y')
    plt.xlabel("Iterations")
    plt.ylabel("Run time in ms (milliseconds) order of 1e-3 seconds")
    plt.title(title)
    plt.legend([f"Average: {mean:.3f} ms"])
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()

def experiment_d_plot(mean_values, title, filename):
    import matplotlib.pyplot as plt
    import numpy as np

    x_labels = [50, 500, 1000, 2000, 5000]
    x = np.arange(len(x_labels))  # x positions for bars

    plt.figure(figsize=(12, 8))
    bars = plt.bar(x, mean_values, color="blue", alpha=0.7)

    # Add values above each bar
    for bar in bars:
        height = bar.get_height()  # Get the height of the bar
        plt.text(
            bar.get_x() + bar.get_width() / 2,  # X-coordinate 
            height + 0.02 * max(mean_values),     # Y-coordinate 
            f"{height:.3f}",                      # Format to 2 decimal places
            ha="center",                        # Align text to center
            va="bottom"                         # Align text at the bottom of the text box
        )

    plt.xticks(x, x_labels)  # Set x-axis labels
    plt.xlabel("Iterations")
    plt.ylabel("Mean Run Time (ms)")
    plt.title(title)

    plt.tight_layout()
    plt.savefig(filename)
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

    reversed_list.sort()    # Used .sort() here because it is much faster than our elementary sorts
    reversed_list.reverse()
    return reversed_list

# function to generate near sorted list of a given size and with a given maximum value
def create_near_sorted_list(length, max_value, item=None, item_index=None):
    near_sorted_list = []

    #include your code here
    for i in range ((max_value - length), max_value):
        near_sorted_list.append(i)
    
    for j in range (len(near_sorted_list)//8):      # Since we want a 75% sorted list we just need to do 1/8 inversions 
        random_item = random.randint(0, len(near_sorted_list)-1)
        near_sorted_list[j], near_sorted_list[random_item] = near_sorted_list[random_item], near_sorted_list[j]     # This is basically just swapping with random items

    return near_sorted_list

# function to generate near sorted list of a given size and with a given maximum value
def reduced_unique_list(length, max_value, item=None, item_index=None):

    #include your code here

    reduced_list = []
    random_list = []
    exist = set()   # Storing our seen items in a set so we know for sure that there is only 1 item in the set when adding new items into it
    for _ in range (length):
        random_list.append(random.randint(0, max_value))    # Generating the random list
    
    for num in random_list:
        if num not in exist:    # Checking if the item is in the set of not, if it is then we don't add it to the reduced list
            reduced_list.append(num)
            exist.add(num)

    return reduced_list     

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
                        items_to_sort[j], items_to_sort[j + 1] = items_to_sort[j + 1], items_to_sort[j]    # Traversing through the list and swapping, very basic 

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
            for i in range(1, len(items_to_sort)):
                key = items_to_sort[i]  # Save the current item/number so that we can insert it into the right place once we find that place
                j = i - 1
                while j >= 0 and items_to_sort[j] > key:    # Loop to keep checking whether the key should be in this position or not
                    items_to_sort[j + 1] = items_to_sort[j]  # Shift elements
                    j -= 1
                items_to_sort[j + 1] = key  # Inserting the key into its correct position
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
                min = i     # Keep track of the location of the min value
                for j in range (i + 1, len(items_to_sort)):
                    if items_to_sort[j] < items_to_sort[min]:   # Update the min value if we find a smaller value
                        min = j
                items_to_sort[i], items_to_sort[min] = items_to_sort[min], items_to_sort[i]     # Swap the min item into the sorted section of the list

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

        while l < len(arr1) and r < len(arr2):  # Merge function takes in two of the auxillery arrays then compares the items and places the smaller
            if arr1[l] > arr2[r]:               # one into the new aux array. We repeat this until one of the arrays is empty
                aux.append(arr2[r])
                r += 1
            else:
                aux.append(arr1[l])
                l += 1

        while l < len(arr1):    # Then we check which array still has items then we add the remaining items into our new aux array
            aux.append(arr1[l])
            l += 1
            
        while r < len(arr2):
            aux.append(arr2[r])
            r += 1

        return aux
        
    def mergeSort(self, arr):   # We recursively call this function to split the original arrays an aux arrays in half everytime until we are
        if (len(arr) <= 1):     # left with singleton arrays which is when the merge functions will begin to merge the aux arrays together
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

    def partition(self, items_to_sort, start, end): # To partition the items I chose to partition on the last item, I wrote this before during 2C03 as practice 
                                                    # so I did not look at the given pseudocode for this implementation, it may pivot around a different item compared to the pseudocode
                                                    # If the item is smaller than the pivot then we perform a swap if not we just increment by one item, then at the end we put the pivot
                                                    # into the right place.
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
        
    def quickSort(self, arr, start, end):   # We recursively call the function on new pivots to partition the lists by halfs and we keep on halving it until we reach one.
                                            # Once we've reached one then we know we've finished partitioning and the list is sorted.
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

    # I could've made a run experiment function but when I originally wrote this I wanted to leave it running while I was away on campus so
    # it ended up having 5 for loops for loops per experiment to measure the run times and call the sorts.
    # This applies to all the experiment, I just append all the run times into an array then calculated the mean using numpy before giving those values into the plot function
    # to the draw the bar charts
    N = 80
    bubble_run_times = []
    insertion_run_times = []
    selection_run_times = []
    quick_run_times = []
    merge_run_times = []
    for _ in range (N):
        random_list = create_random_list(10000,100000)
        start_time = timeit.default_timer()

        _ = BubbleSort(random_list[:])

        end_time = timeit.default_timer()

        run_time = (end_time - start_time) * 1e3
        bubble_run_times.append(run_time)
    
    for _ in range (N):
        random_list = create_random_list(10000,100000)
        start_time = timeit.default_timer()

        _ = InsertionSort(random_list[:])

        end_time = timeit.default_timer()

        run_time = (end_time - start_time) * 1e3
        insertion_run_times.append(run_time)
    
    for _ in range (N):
        random_list = create_random_list(10000,100000)
        start_time = timeit.default_timer()

        _ = SelectionSort(random_list[:])

        end_time = timeit.default_timer()

        run_time = (end_time - start_time) * 1e3
        selection_run_times.append(run_time)
    
    for _ in range (N):
        random_list = create_random_list(10000,100000)
        start_time = timeit.default_timer()

        _ = QuickSort(random_list[:])

        end_time = timeit.default_timer()

        run_time = (end_time - start_time) * 1e3
        quick_run_times.append(run_time)
    
    for _ in range (N):
        random_list = create_random_list(10000,100000)
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

    draw_plot(bubble_run_times, bubble_mean, "Experiment A: BubbleSort", "bubble_experiment_a.png")
    draw_plot(insertion_run_times, insertion_mean, "Experiement A: InsertionSort", "insertion_experiment_a.png")
    draw_plot(selection_run_times, selection_mean, "Experiment A: SelectionSort", "selection_experiment_a.png")
    draw_plot(quick_run_times, quick_mean, "Experiment A: QuickSort", "quick_experiment_a.png")
    draw_plot(merge_run_times, merge_mean, "Experiment A: MergeSort", "merge_experiment_a.png")

    return 0

def experiment_B():
    
    # Insert your code for experiment B design here 
    N = 100

    bubble_run_times = []
    insertion_run_times = []
    selection_run_times = []
    quick_run_times = []
    merge_run_times = []

    for _ in range (N):
        near_sorted_list = create_near_sorted_list(5000,100000)
        start_time = timeit.default_timer()

        _ = BubbleSort(near_sorted_list[:])

        end_time = timeit.default_timer()
        run_time = (end_time - start_time) * 1e3
        bubble_run_times.append(run_time)

    for _ in range (N):
        near_sorted_list = create_near_sorted_list(5000,100000)
        start_time = timeit.default_timer()

        _ = InsertionSort(near_sorted_list[:])

        end_time = timeit.default_timer()
        run_time = (end_time - start_time) * 1e3
        insertion_run_times.append(run_time)
    
    for _ in range (N):
        near_sorted_list = create_near_sorted_list(5000,100000)
        start_time = timeit.default_timer()

        _ = SelectionSort(near_sorted_list[:])

        end_time = timeit.default_timer()
        run_time = (end_time - start_time) * 1e3
        selection_run_times.append(run_time)
    
    for _ in range (N):
        near_sorted_list = create_near_sorted_list(5000,100000)
        start_time = timeit.default_timer()

        _ = QuickSort(near_sorted_list[:])

        end_time = timeit.default_timer()
        run_time = (end_time - start_time) * 1e3
        quick_run_times.append(run_time)
    
    for _ in range (N):
        near_sorted_list = create_near_sorted_list(5000,100000)
        start_time = timeit.default_timer()

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

    draw_plot(bubble_run_times, bubble_mean, "Experiment B: BubbleSort", "bubble_experiment_b.png")
    draw_plot(insertion_run_times, insertion_mean, "Experiment B: InsertionSort", "insertion_experiment_b.png")
    draw_plot(selection_run_times, selection_mean, "Experiment B: SelectionSort", "selection_experiment_b.png")
    draw_plot(quick_run_times, quick_mean, "Experiment B: QuickSort", "quick_experiment_b.png")
    draw_plot(merge_run_times, merge_mean, "Experiment B: MergeSort", "merge_experiment_b.png")

    return 0

def experiment_C():
    
    # Insert your code for experiment C design here 
    
    N = 100

    bubble_run_times = []
    insertion_run_times = []
    selection_run_times = []
    quick_run_times = []
    merge_run_times = []

    for _ in range (N):
        reversed_list = create_reverse_list(10000, 100000)
        start_time = timeit.default_timer()

        _ = BubbleSort(reversed_list[:])

        end_time = timeit.default_timer()
        run_time = (end_time - start_time) * 1e3
        bubble_run_times.append(run_time)
    
    for _ in range (N):
        reversed_list = create_reverse_list(10000, 100000)
        start_time = timeit.default_timer()

        _ = InsertionSort(reversed_list[:])

        end_time = timeit.default_timer()
        run_time = (end_time - start_time) * 1e3
        insertion_run_times.append(run_time)

    for _ in range (N):
        reversed_list = create_reverse_list(10000, 100000)
        start_time = timeit.default_timer()

        _ = SelectionSort(reversed_list[:])

        end_time = timeit.default_timer()
        run_time = (end_time - start_time) * 1e3
        selection_run_times.append(run_time)

    for _ in range (N):
        reversed_list = create_reverse_list(10000, 100000)
        start_time = timeit.default_timer()

        _ = QuickSort(reversed_list[:])

        end_time = timeit.default_timer()
        run_time = (end_time - start_time) * 1e3
        quick_run_times.append(run_time)

    for _ in range (N):
        reversed_list = create_reverse_list(10000, 100000)
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

    draw_plot(bubble_run_times, bubble_mean, "Experiment C: BubbleSort", "bubble_experiment_c.png")
    draw_plot(insertion_run_times, insertion_mean, "Experiment C: InsertionSort", "insertion_experiment_c.png")
    draw_plot(selection_run_times, selection_mean, "Experiment C: SelectionSort", "selection_experiment_c.png")
    draw_plot(quick_run_times, quick_mean, "Experiment C: QuickSort", "quick_experiment_c.png")
    draw_plot(merge_run_times, merge_mean, "Experiment C: MergeSort", "merge_experiment_c.png")

    return 0

def experiment_D():

    # Insert your code for experiment D design here

    # Experiment D was a bit different here since I had 25 for loops, very inefficient but I don't want to waste more time refactoring code that alreay works and I don't get additional marks for.
    # I made a separate draw function for D since I wanted to diaplay the mean times on top of the bars 
    N = 80

    bubble_run_times_50 = []
    bubble_run_times_500 = []
    bubble_run_times_1000 = []
    bubble_run_times_2000 = []
    bubble_run_times_5000 = []

    insertion_run_times_50 = []
    insertion_run_times_500 = []
    insertion_run_times_1000 = []
    insertion_run_times_2000 = []
    insertion_run_times_5000 = []

    selection_run_times_50 = []
    selection_run_times_500 = []
    selection_run_times_1000 = []
    selection_run_times_2000 = []
    selection_run_times_5000 = []

    quick_run_times_50 = []
    quick_run_times_500 = []
    quick_run_times_1000 = []
    quick_run_times_2000 = []
    quick_run_times_5000 = []

    merge_run_times_50 = []
    merge_run_times_500 = []
    merge_run_times_1000 = []
    merge_run_times_2000 = []
    merge_run_times_5000 = []

    bubble_start = timeit.default_timer()
    for i in range (N):
        random_50 = create_random_list(50, 100000)
        start_time50 = timeit.default_timer()
        _ = BubbleSort(random_50[:])
        end_time50 = timeit.default_timer()
        run_time_50 = (end_time50 - start_time50) * 1e3
        bubble_run_times_50.append(run_time_50)

        print("BubbleSort length 50 iteration: ", i+1)

        random_500 = create_random_list(500, 100000)
        start_time500 = timeit.default_timer()
        _ = BubbleSort(random_500[:])
        end_time500 = timeit.default_timer()
        run_time_500 = (end_time500 - start_time500) * 1e3
        bubble_run_times_500.append(run_time_500)

        print("BubbleSort length 500 iteration: ", i+1)
        
        random_1000 =  create_random_list(1000, 100000)
        start_time1000 = timeit.default_timer()
        _ = BubbleSort(random_1000[:])
        end_time1000 = timeit.default_timer()
        run_time_1000 = (end_time1000 - start_time1000) * 1e3
        bubble_run_times_1000.append(run_time_1000)

        print("BubbleSort length 1000 iteration: ", i+1)

        random_2000 = create_random_list(2000, 100000)
        start_time2000 = timeit.default_timer()
        _ = BubbleSort(random_2000[:])
        end_time2000 = timeit.default_timer()
        run_time_2000 = (end_time2000 - start_time2000) * 1e3
        bubble_run_times_2000.append(run_time_2000)

        print("BubbleSort length 2000 iteration: ", i+1)

        random_5000 = create_random_list(5000, 100000)
        start_time5000 = timeit.default_timer()
        _ = BubbleSort(random_5000[:])
        end_time5000 = timeit.default_timer()
        run_time_5000 = (end_time5000 - start_time5000) * 1e3
        bubble_run_times_5000.append(run_time_5000)

        print("BubbleSort length 5000 iteration: ", i+1)
    
    bubble_end = timeit.default_timer()
    bubble_time = (bubble_end - bubble_start)
    
    insertion_start = timeit.default_timer()
    for i in range (N):
        random_50 = create_random_list(50, 100000)
        start_time50 = timeit.default_timer()
        _ = InsertionSort(random_50[:])
        end_time50 = timeit.default_timer()
        run_time_50 = (end_time50 - start_time50) * 1e3
        insertion_run_times_50.append(run_time_50)

        print("InsertionSort length 50 iteration: ", i+1)

        random_500 = create_random_list(500, 100000)
        start_time500 = timeit.default_timer()
        _ = InsertionSort(random_500[:])
        end_time500 = timeit.default_timer()
        run_time_500 = (end_time500 - start_time500) * 1e3
        insertion_run_times_500.append(run_time_500)

        print("InsertionSort length 500 iteration: ", i+1)
        
        random_1000 =  create_random_list(1000, 100000)
        start_time1000 = timeit.default_timer()
        _ = InsertionSort(random_1000[:])
        end_time1000 = timeit.default_timer()
        run_time_1000 = (end_time1000 - start_time1000) * 1e3
        insertion_run_times_1000.append(run_time_1000)

        print("InsertionSort length 1000 iteration: ", i+1)

        random_2000 = create_random_list(2000, 100000)
        start_time2000 = timeit.default_timer()
        _ = InsertionSort(random_2000[:])
        end_time2000 = timeit.default_timer()
        run_time_2000 = (end_time2000 - start_time2000) * 1e3
        insertion_run_times_2000.append(run_time_2000)

        print("InsertionSort length 2000 iteration: ", i+1)

        random_5000 = create_random_list(5000, 100000)
        start_time5000 = timeit.default_timer()
        _ = InsertionSort(random_5000[:])
        end_time5000 = timeit.default_timer()
        run_time_5000 = (end_time5000 - start_time5000) * 1e3
        insertion_run_times_5000.append(run_time_5000)

        print("InsertionSort length 5000 iteration: ", i+1)
    insertion_end = timeit.default_timer()
    insertion_time = (insertion_end - insertion_start)

    selection_start = timeit.default_timer()
    for i in range (N):
        random_50 = create_random_list(50, 100000)
        start_time50 = timeit.default_timer()
        _ = SelectionSort(random_50[:])
        end_time50 = timeit.default_timer()
        run_time_50 = (end_time50 - start_time50) * 1e3
        selection_run_times_50.append(run_time_50)

        print("SelectionSort length 50 iteration: ", i+1)

        random_500 = create_random_list(500, 100000)
        start_time500 = timeit.default_timer()
        _ = SelectionSort(random_500[:])
        end_time500 = timeit.default_timer()
        run_time_500 = (end_time500 - start_time500) * 1e3
        selection_run_times_500.append(run_time_500)

        print("SelectionSort length 500 iteration: ", i+1)
        
        random_1000 =  create_random_list(1000, 100000)
        start_time1000 = timeit.default_timer()
        _ = SelectionSort(random_1000[:])
        end_time1000 = timeit.default_timer()
        run_time_1000 = (end_time1000 - start_time1000) * 1e3
        selection_run_times_1000.append(run_time_1000)

        print("SelectionSort length 1000 iteration: ", i+1)

        random_2000 = create_random_list(2000, 100000)
        start_time2000 = timeit.default_timer()
        _ = SelectionSort(random_2000[:])
        end_time2000 = timeit.default_timer()
        run_time_2000 = (end_time2000 - start_time2000) * 1e3
        selection_run_times_2000.append(run_time_2000)

        print("SelectionSort length 2000 iteration: ", i+1)

        random_5000 = create_random_list(5000, 100000)
        start_time5000 = timeit.default_timer()
        _ = SelectionSort(random_5000[:])
        end_time5000 = timeit.default_timer()
        run_time_5000 = (end_time5000 - start_time5000) * 1e3
        selection_run_times_5000.append(run_time_5000)

        print("SelectionSort length 5000 iteration: ", i+1)
    selection_end = timeit.default_timer()
    selection_time = (selection_end - selection_start)

    quick_start = timeit.default_timer()
    for i in range (N):
        random_50 = create_random_list(50, 100000)
        start_time50 = timeit.default_timer()
        _ = QuickSort(random_50[:])
        end_time50 = timeit.default_timer()
        run_time_50 = (end_time50 - start_time50) * 1e3
        quick_run_times_50.append(run_time_50)

        print("QuickSort length 50 iteration: ", i+1)

        random_500 = create_random_list(500, 100000)
        start_time500 = timeit.default_timer()
        _ = QuickSort(random_500[:])
        end_time500 = timeit.default_timer()
        run_time_500 = (end_time500 - start_time500) * 1e3
        quick_run_times_500.append(run_time_500)

        print("QuickSort length 500 iteration: ", i+1)
        
        random_1000 =  create_random_list(1000, 100000)
        start_time1000 = timeit.default_timer()
        _ = QuickSort(random_1000[:])
        end_time1000 = timeit.default_timer()
        run_time_1000 = (end_time1000 - start_time1000) * 1e3
        quick_run_times_1000.append(run_time_1000)

        print("QuickSort length 1000 iteration: ", i+1)

        random_2000 = create_random_list(2000, 100000)
        start_time2000 = timeit.default_timer()
        _ = QuickSort(random_2000[:])
        end_time2000 = timeit.default_timer()
        run_time_2000 = (end_time2000 - start_time2000) * 1e3
        quick_run_times_2000.append(run_time_2000)

        print("QuickSort length 2000 iteration: ", i+1)

        random_5000 = create_random_list(5000, 100000)
        start_time5000 = timeit.default_timer()
        _ = QuickSort(random_5000[:])
        end_time5000 = timeit.default_timer()
        run_time_5000 = (end_time5000 - start_time5000) * 1e3
        quick_run_times_5000.append(run_time_5000)

        print("QuickSort length 5000 iteration: ", i+1)
    quick_end = timeit.default_timer()
    quick_time = (quick_end - quick_start)

    merge_start = timeit.default_timer()
    for i in range (N):
        random_50 = create_random_list(50, 100000)
        start_time50 = timeit.default_timer()
        _ = MergeSort(random_50[:])
        end_time50 = timeit.default_timer()
        run_time_50 = (end_time50 - start_time50) * 1e3
        merge_run_times_50.append(run_time_50)

        print("MergeSort length 50 iteration: ", i+1)

        random_500 = create_random_list(500, 100000)
        start_time500 = timeit.default_timer()
        _ = MergeSort(random_500[:])
        end_time500 = timeit.default_timer()
        run_time_500 = (end_time500 - start_time500) * 1e3
        merge_run_times_500.append(run_time_500)

        print("MergeSort length 5000 iteration: ", i+1)
        
        random_1000 =  create_random_list(1000, 100000)
        start_time1000 = timeit.default_timer()
        _ = MergeSort(random_1000[:])
        end_time1000 = timeit.default_timer()
        run_time_1000 = (end_time1000 - start_time1000) * 1e3
        merge_run_times_1000.append(run_time_1000)

        print("MergeSort length 1000 iteration: ", i+1)

        random_2000 = create_random_list(2000, 100000)
        start_time2000 = timeit.default_timer()
        _ = MergeSort(random_2000[:])
        end_time2000 = timeit.default_timer()
        run_time_2000 = (end_time2000 - start_time2000) * 1e3
        merge_run_times_2000.append(run_time_2000)

        print("MergeSort length 2000 iteration: ", i+1)

        random_5000 = create_random_list(5000, 100000)
        start_time5000 = timeit.default_timer()
        _ = MergeSort(random_5000[:])
        end_time5000 = timeit.default_timer()
        run_time_5000 = (end_time5000 - start_time5000) * 1e3
        merge_run_times_5000.append(run_time_5000)

        print("MergeSort length 5000 iteration: ", i+1)
    merge_end = timeit.default_timer()
    merge_time = (merge_end - merge_start)

    bubble_mean_times = [np.mean(bubble_run_times_50), np.mean(bubble_run_times_500), np.mean(bubble_run_times_1000), np.mean(bubble_run_times_2000), np.mean(bubble_run_times_5000)]
    insertion_mean_times = [np.mean(insertion_run_times_50), np.mean(insertion_run_times_500), np.mean(insertion_run_times_1000), np.mean(insertion_run_times_2000), np.mean(insertion_run_times_5000)]
    selection_mean_times = [np.mean(selection_run_times_50), np.mean(selection_run_times_500), np.mean(selection_run_times_1000), np.mean(selection_run_times_2000), np.mean(selection_run_times_5000)]
    quick_mean_times = [np.mean(quick_run_times_50), np.mean(quick_run_times_500), np.mean(quick_run_times_1000), np.mean(quick_run_times_2000), np.mean(quick_run_times_5000)]
    merge_mean_times = [np.mean(merge_run_times_50), np.mean(merge_run_times_500), np.mean(merge_run_times_1000), np.mean(merge_run_times_2000), np.mean(merge_run_times_5000)]

    print("BubbleSort took: ", bubble_time, "s")
    print("InsertionSort took: ", insertion_time, "s")
    print("SelectionSort took: ", selection_time, "s")
    print("QuickSort took: ", quick_time, "s")
    print("MergeSort took: ", merge_time, "s")

    experiment_d_plot(bubble_mean_times, "Experiment D: BubbleSort", "bubble_experiment_d.png")
    experiment_d_plot(insertion_mean_times, "Experiment D: InsertionSort", "insertion_experiment_d.png")
    experiment_d_plot(selection_mean_times, "Experiment D: SelectionSort", "selection_experiment_d.png")
    experiment_d_plot(quick_mean_times, "Experiment D:QuickSort", "quick_experiment_d.png")
    experiment_d_plot(merge_mean_times, "Experiment D: MergeSort", "merge_experiment_d.png")

    return 0

def experiment_E():
    
    # Insert your code for experiment E design here 
    
    N = 100
    
    bubble_run_times = []
    insertion_run_times = []
    selection_run_times = []
    quick_run_times = []
    merge_run_times = []

    
    for _ in range (N):
        reduced_list = reduced_unique_list(5000, 100000)
        start_time = timeit.default_timer()

        _ = BubbleSort(reduced_list[:])

        end_time = timeit.default_timer()
        run_time = (end_time - start_time) * 1e3
        bubble_run_times.append(run_time)

    for _ in range (N):
        reduced_list = reduced_unique_list(5000, 100000)
        start_time = timeit.default_timer()

        _ = InsertionSort(reduced_list[:])

        end_time = timeit.default_timer()
        run_time = (end_time - start_time) * 1e3
        insertion_run_times.append(run_time)

    for _ in range (N):
        reduced_list = reduced_unique_list(5000, 100000)
        start_time = timeit.default_timer()

        _ = SelectionSort(reduced_list[:])

        end_time = timeit.default_timer()
        run_time = (end_time - start_time) * 1e3
        selection_run_times.append(run_time)

    for _ in range (N):
        reduced_list = reduced_unique_list(5000, 100000)
        start_time = timeit.default_timer()

        _ = QuickSort(reduced_list[:])

        end_time = timeit.default_timer()
        run_time = (end_time - start_time) * 1e3
        quick_run_times.append(run_time)

    for _ in range (N):
        reduced_list = reduced_unique_list(5000, 100000)
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

    draw_plot(bubble_run_times, bubble_mean, "Experiment E: BubbleSort", "bubble_experiment_e.png")
    draw_plot(insertion_run_times, insertion_mean, "Experiment E: InsertionSort", "insertion_experiment_e.png")
    draw_plot(selection_run_times, selection_mean, "Experiment E: SelectionSort", "selection_experiment_e.png")
    draw_plot(quick_run_times, quick_mean, "Experiment E: QuickSort", "quick_experiment_e.png")
    draw_plot(merge_run_times, merge_mean, "Experiment E: MergeSort", "merge_experiment_e.png")

    return 0

# call each experiment
#experiment_A()
#experiment_B()
#experiment_C()
#experiment_D()
experiment_E()


    
