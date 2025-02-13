import random
import time
import timeit 
import matplotlib.pyplot as plt
import numpy as np
import copy

# Utility functions - some are implemented, others you must implement yourself.

# function to plot the bar graph and average runtimes of N trials
# Please note that this function only plots the graph and does not save it
# To save the graphs you must use plot.save(). Refer to matplotlib documentation
def draw_plot(run_arr, mean):
    x = np.arange(0, len(run_arr),1)
    fig=plt.figure(figsize=(20,8))
    plt.axhline(mean,color="red",linestyle="--",label="Avg")
    plt.bar(run_arr, color = 'blue')
    plt.xlabel("Iterations")
    plt.ylabel("Run time in ms order of 1e-6")
    plt.title("Run time for retrieval")
    plt.show()

# function to generate random graphs 
# @args : nodes = number of nodes 
#       : edges = number of edges
def create_random_graph(nodes, edges):
    graph = None

    # your implementation goes here

    return graph

# function to generate random list 
# @args : length = number of items 
#       : max_value maximum value
def create_random_list(length, max_value, item=None, item_index=None):
    # your implementation for goes here

    random_list = [random.randint(0,max_value) for i in range(length)]
    if item!= None:
        random_list.insert(item_index,item)

    return random_list

# hybrid sort
def hybrid_sort(L):
    # your implementation for part 1 goes here

    def recursive_binary_search(low, high, L , key):
      
        if low > high:
            return low
        else:
            mid = (low + high) // 2
            if L[mid] == key:
                return mid + 1
            elif L[mid] > key:
                recursive_binary_search(0, mid - 1, L, key)
            else:
                recursive_binary_search(mid + 1, high, L, key)     

    for i in range (1, len (L)):

        key = L[i]
        
        index = recursive_binary_search(0, i , L, key)

        L.insert(index, key)

    return

def insertion_sort(L):
    for i in range (1, len(L)):
        key = L[i]
        j = i - 1

        while j >= 0 and L[j] > L[i]:
            L[j + 1] = L[j]
            j -= 1

        L[j] = key
    
    return 0

def experiment_part_2():

    # your implementation for part 2 goes here
    insertion_times = []
    hybrid_times = []

    for _ in range (30):
        random_list = create_random_list(500, 5000)
        insertion_list = copy.deepcopy(random_list)
        hybrid_list = copy.deepcopy(random_list)

        start = timeit.default_timer()
        hybrid_sort(hybrid_list)
        end = timeit.default_timer()
        hybrid_times.append(end - start)

        start = timeit.default_timer()
        insertion_sort(insertion_list)
        end = timeit.default_timer()
        insertion_times.append(end - start)

    draw_plot(insertion_times, np.mean(insertion_times))
    draw_plot(hybrid_times, np.mean(hybrid_times))

    return 0

# binary search dynamic
class binary_search():
    def __init__(self):
        pass

    # your implementation for part 3 goes here
    # feel free to add arguments and helper functions that you need to add
    def search():
        return 0 
    
    def insert():
        return 0 
    
    def delete():
        return 0 
    
    def binary_search_dynamic():
        return 0
    
def experiment_part_4():

     # your implementation for part 4 goes here

    return 0

# binary search  implementation 1 against which you must compare the performance of Part 3 
def binary_search_1(item_list, to_find):
    lower=0
    upper=len(item_list)-1
    while lower < upper:
        mid = (lower+upper)//2
        if item_list[mid] == to_find:
            return True
        if item_list[mid] < to_find:
            lower = mid+1
        else:
            upper=mid
    return item_list[lower]==to_find

# binary search  implementation 2 against which you must compare the performance of Part 3 
def binary_search_2(item_list, to_find):
    lower=0
    upper=len(item_list)-1
    while lower <= upper:
        mid = (lower+upper)//2
        if item_list[mid] == to_find:
            return True
        if item_list[mid] < to_find:
            lower = mid+1
        else:
            upper=mid-1
    return item_list[lower]==to_find

# binary search  implementation 3 against which you must compare the performance of Part 3 
def binary_search_3(item_list, to_find):
    left=0
    right=len(item_list)-1
    while left != right:
        mid = (left+right)//2
        if item_list[mid] < to_find:
            left = mid+1
        elif item_list[mid] > to_find:
            right = mid
        else:
            return True
    return item_list[left]==to_find

# below is code template for part 5
class DirectedWeightedGraph:

    def __init__(self):
        self.adj = {}
        self.weights = {}

    def are_connected(self, node1, node2):
        for neighbour in self.adj[node1]:
            if neighbour == node2:
                return True
        return False

    def adjacent_nodes(self, node):
        return self.adj[node]

    def add_node(self, node):
        self.adj[node] = []

    def add_edge(self, node1, node2, weight):
        if node2 not in self.adj[node1]:
            self.adj[node1].append(node2)
        self.weights[(node1, node2)] = weight

    def w(self, node1, node2):
        if self.are_connected(node1, node2):
            return self.weights[(node1, node2)]

    def number_of_nodes(self):
        return len(self.adj)

def init_d(G):
    n = G.number_of_nodes()
    d = [[float("inf") for j in range(n)] for i in range(n)]
    for i in range(n):
        for j in range(n):
            if G.are_connected(i, j):
                d[i][j] = G.w(i, j)
        d[i][i] = 0
    return d

#Assumes G represents its nodes as integers 0,1,...,(n-1)
# this is the unknown algorithm you must reverse engineer for part 6
def unknown(G):
    n = G.number_of_nodes()
    d = init_d(G)
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if d[i][j] > d[i][k] + d[k][j]: 
                    d[i][j] = d[i][k] + d[k][j]
    return d

# below code template is for part 6
class Graph():
    def __init__(self):
        self.adj = {}
        self.weights = {}

    # you may use the above weighted graph class here and modify it if necessary.
    # aslo feel free to borrow any of the suitable graph class implementations discussed in class. 

class Heap():
    def __init__(self):
        pass

    # borrow this implementation from class

def prims(G):
    mst = []

    # borrow this implementation from class
    return mst

def krushkals(G):
    mst = []

    # your implementation for part 6 (krushkal's algorithm) goes here
    return mst

def experiment_part_6():

    # your implementation for part 6 (experiment to compare with prim's) goes here
    return 0
