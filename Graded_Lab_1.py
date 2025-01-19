

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
def draw_plot(run_arr, mean):
    x = np.arange(0, len(run_arr),1)
    fig=plt.figure(figsize=(20,8))
    plt.axhline(mean,color="red",linestyle="--",label="Avg")
    plt.xlabel("Iterations")
    plt.ylabel("Run time in ms order of 1e-6")
    plt.title("Run time for retrieval")
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
    for i in range (max_value,(max_value - length), -1):
        reversed_list.append(i)

    return reversed_list

# function to generate near sorted list of a given size and with a given maximum value
def create_near_sorted_list(length, max_value, item=None, item_index=None):
    near_sorted_list = []

    #include your code here
    for i in range ((max_value - length), max_value):
        near_sorted_list.append(i)
    
    for j in range (len(near_sorted_list)/8):
        random_item = random.randint(len(near_sorted_list))
        near_sorted_list[i], near_sorted_list[random_item] = near_sorted_list[random_item], near_sorted_list[i]

    return near_sorted_list

# function to generate near sorted list of a given size and with a given maximum value
def reduced_unique_list(length, max_value, item=None, item_index=None):
    reduced_list = []

    #include your code here

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

    return 0

def experiment_B():
    
    # Insert your code for experiment B design here 

    return 0

def experiment_C():
    
    # Insert your code for experiment C design here 

    return 0

def experiment_D():
    
    # Insert your code for experiment D design here 

    return 0

def experiment_E():
    
    # Insert your code for experiment E design here 

    return 0

# call each experiment
experiment_A()
experiment_B()
experiment_C()
experiment_D()
experiment_E()
    
