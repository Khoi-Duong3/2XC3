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
def draw_plot(run_arr, mean, title, filename):
    x = np.arange(0, len(run_arr),1) + 1
    fig=plt.figure(figsize=(20,8))
    plt.axhline(mean,color="red",linestyle="--",label=f"Average = {mean}")
    plt.bar(x, run_arr, color = 'blue')
    plt.xlabel("Iterations")
    plt.ylabel("Run time in μs")
    plt.title(title)
    plt.legend()
    plt.savefig(filename)
    plt.show()

# function to generate random graphs 
# @args : nodes = number of nodes 
#       : edges = number of edges
def create_random_graph(nodes, edges, min_weight, max_weight):

    # your implementation goes here
    graph = Graph()

    for i in range(nodes):
        graph.adj[i] = []
    
    valid_edges = [(u,v) for u in range(nodes) for v in range(u + 1, nodes)]

    num_edges = min(len(valid_edges), edges)

    chosen_edges = random.sample(valid_edges, num_edges)

    for u,v in chosen_edges:
        weight = random.randint(min_weight, max_weight)
        graph.adj[u].append(v)
        graph.adj[v].append(u)
        graph.weights[(u,v)] = weight

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
        mid = (low + high) // 2
        if low >= high:
            if mid < 0:
                mid = 0
            elif mid > len(L) - 1:
                mid = len(L) - 1
        
            if L[mid] < key:
                return mid + 1
            else:
                return mid
        
        if L[mid] == key:
            return mid
        elif L[mid] > key:
            return recursive_binary_search(low, mid - 1, L, key)
        else:
            return recursive_binary_search(mid + 1, high, L, key)       

    for i in range (1, len (L)):
        
        index = recursive_binary_search(0, i-1, L, key = L[i])
        if i != index:
            L.insert(index, L.pop(i))

    return

def insertion_sort(L):
    for i in range (1, len(L)):
        key = L[i]
        j = i - 1

        while j >= 0 and L[j] > key:
            L[j + 1] = L[j]
            j -= 1

        L[j + 1] = key
    
    return 0

def experiment_part_2():

    # your implementation for part 2 goes here
    insertion_times = []
    hybrid_times = []

    
    
    # This is a test for the average case
    
    for _ in range (30):
        random_list = create_random_list(500, 5000)
        insertion_list = copy.deepcopy(random_list)
        hybrid_list = copy.deepcopy(random_list)

        start = timeit.default_timer()
        hybrid_sort(hybrid_list)
        end = timeit.default_timer()
        hybrid_times.append((end - start) * 1000000)

        start = timeit.default_timer()
        insertion_sort(insertion_list)
        end = timeit.default_timer()
        insertion_times.append((end - start) * 1000000)
    
    '''
    # This is a test for the worst case
   
    for _ in range (30):
        random_list = create_random_list(500, 5000)
        random_list.sort()
        random_list.reverse()
        insertion_list = copy.deepcopy(random_list)
        hybrid_list = copy.deepcopy(random_list)

        start = timeit.default_timer()
        hybrid_sort(hybrid_list)
        end = timeit.default_timer()
        hybrid_times.append((end - start) * 1000000)

        start = timeit.default_timer()
        insertion_sort(insertion_list)
        end = timeit.default_timer()
        insertion_times.append((end - start) * 1000000)
    '''
    '''
    # This is a test for the best case  
    for _ in range (30):
        random_list = create_random_list(500, 5000)
        random_list.sort()
        insertion_list = copy.deepcopy(random_list)
        hybrid_list = copy.deepcopy(random_list)

        start = timeit.default_timer()
        hybrid_sort(hybrid_list)
        end = timeit.default_timer()
        hybrid_times.append((end - start) * 1000000)

        start = timeit.default_timer()
        insertion_sort(insertion_list)
        end = timeit.default_timer()
        insertion_times.append((end - start) * 1000000)
    '''
    '''
    # This test is for the random case
    for _ in range (10):
        random_list = create_random_list(500, 5000)
        insertion_list = copy.deepcopy(random_list)
        hybrid_list = copy.deepcopy(random_list)

        start = timeit.default_timer()
        hybrid_sort(hybrid_list)
        end = timeit.default_timer()
        hybrid_times.append((end - start) * 1000000)

        start = timeit.default_timer()
        insertion_sort(insertion_list)
        end = timeit.default_timer()
        insertion_times.append((end - start) * 1000000)
    '''
    draw_plot(insertion_times, np.mean(insertion_times), "Insertion Sort Random Case", "insertion_sort_random.png")
    draw_plot(hybrid_times, np.mean(hybrid_times), "Hybrid Sort Random Case", "Hybrid_sort_random.png")

    return 0

# binary search dynamic
class DynamicArrays():
    def __init__(self):
        self.arrays = []
        self.elements = 0

    # your implementation for part 3 goes here
    # feel free to add arguments and helper functions that you need to add

    def binary_search(self, array, target):
        n = len(array)
        low = 0
        high = n - 1
        while high >= low:
            mid = (high + low) // 2
            if array[mid] == target:
                return mid
            elif array[mid] > target:
                high = mid - 1
            else:
                low = mid + 1     
        
        return -1
    
    # This function merges two sorted arrays in the same fashion as how the merge function works in merge sort 
    def merge_arrays(self, array1, array2):
        merged = []

        array1_pointer = 0
        array2_pointer = 0

        while array1_pointer < len(array1) and array2_pointer < len(array2):
            if array1[array1_pointer] < array2[array2_pointer]:
                merged.append(array1[array1_pointer])
                array1_pointer += 1

            elif array1[array1_pointer] > array2[array2_pointer]:
                merged.append(array2[array2_pointer])
                array2_pointer += 1
            
            else:
                merged.append(array1[array1_pointer])
                merged.append(array2[array2_pointer])
                array1_pointer += 1
                array2_pointer += 1
        
        while array1_pointer < len(array1):
            merged.append(array1[array1_pointer])
            array1_pointer += 1
        
        while array2_pointer < len(array2):
            merged.append(array2[array2_pointer])
            array2_pointer += 1

        return merged

    def search(self, target):
        for array in self.arrays:
            if array:
                if self.binary_search(array, target) != -1:
                    return True
        
        return False
    
    def insert(self, item):
        new_array = [item]
        i = 0

        while True:
            if i >= len(self.arrays):
                self.arrays.append(new_array)
                break
            if self.arrays[i] is None:
                self.arrays[i] = new_array
                break
            else:
                merged = self.merge_arrays(self.arrays[i], new_array)
                self.arrays[i] = None
                new_array = merged
                i += 1
        
        
        self.elements += 1
        
        return
    
    def delete(self, item):
        if not self.seach(item):
            return
        
        deleted = []
        for array in self.arrays:
            if array:
                i = 0
                while i < len(deleted):
                    deleted.append(array[i])
                    i += 1

        deleted.remove(item)
        self.arrays = []        # This line clears out the current dynamic array by reinitializing it
        self.elements = 0
        for element in deleted:
            self.insert(element)

        return 
    
    def get_random_num(self):
        if self.elements == 0:
            return None
        
        index = random.randint(0, self.elements - 1)

        for array in self.arrays:
            if array:
                if index < len(array):
                    return array[index]
                else:
                    index -= len(array)
        
        return None
    
    # WHY IS THIS NEEDED??? IDK WHAT TO DO WITH IT!!! 
    def binary_search_dynamic(self, target):
        return self.search(target)
    
def experiment_part_4():

     # your implementation for part 4 goes here
    one_times = []
    two_times = []
    three_times = []
    dynamic_times = []

    for _ in range (30):
        L = create_random_list(5000, 10000)
        dynamic_array = DynamicArrays()
        for i in range (len(L)):
            dynamic_array.insert(L[i])
        L.sort()
        target = random.choice(L)
        
        start = timeit.default_timer()
        binary_search_1(L, target)
        end = timeit.default_timer()
        one_times.append((end - start) * 1000000)

        start = timeit.default_timer()
        binary_search_2(L, target)
        end = timeit.default_timer()
        two_times.append((end - start) * 1000000)

        start = timeit.default_timer()
        binary_search_3(L, target)
        end = timeit.default_timer()
        three_times.append((end - start) * 1000000)

        start = timeit.default_timer()
        dynamic_array.binary_search_dynamic(target)
        end = timeit.default_timer()
        dynamic_times.append((end - start) * 1000000)

    draw_plot(one_times, np.mean(one_times), "Binary Search 1 performance", "binary1.png")
    draw_plot(two_times, np.mean(two_times), "Binary Search 2 performance", "binary2.png") 
    draw_plot(three_times, np.mean(three_times), "Binary Search 3 performance", "binary3.png")     
    draw_plot(dynamic_times, np.mean(dynamic_times), "Dynamic Array performance", "dynamic.png") 


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
# This unknown function is performing edge relaxation, whenever it finds a path through the itermediate node at k that is shorter than the path from node i to node j.
# It will update the path from i to j to be from i to k then from k to j. It finds the shortest path between every single pair of nodes in a graph. I believe this 
# algorithm is known as the Floyd Warshall algorithm
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
        self.heap = []
        self.size = 0
    # borrow this implementation from class

    def swim(self, index):
        parent_index = (index - 1) // 2
        if index > 0 and self.heap[index] < self.heap[parent_index]:
            self.heap[index], self.heap[parent_index] = self.heap[parent_index], self.heap[index]
            self.swim(parent_index)

        return

    def sink (self, index):
        left_child = 2 * index + 1
        right_child = 2 * index + 2
        smallest = index
        if left_child < self.size and self.heap[left_child] < self.heap[smallest]:
            smallest = left_child
        if right_child < self.size and self.heap[right_child] < self.heap[smallest]:
            smallest = right_child
        if smallest != index:
            self.heap[index], self.heap[smallest] = self.heap[smallest], self.heap[index]
            self.sink(smallest)

        return
    
    def insert(self, value):
        self.heap.append(value)
        self.size += 1
        self.swim(self.size - 1)

        return
    
    def swap(self, index1, index2):
        self.heap[index1], self.heap[index2] = self.heap[index2], self.heap[index1]

        return
    
    def delete(self):
        self.swap(0, self.size - 1)
        min_value = self.heap.pop()
        self.size -= 1
        self.sink(0)

        return min_value
    
    def heapify(self, array):
        self.heap = array
        self.size = len(array)
        for i in range(self.size // 2 - 1, -1, -1):
            self.sink(i)

        return

class UnionFind():
    def __init__(self, vertices):
        self.parent = {i: i for i in vertices}
        self.rank = {i: 0 for i in vertices}
    def find(self, vertex):
        if self.parent[vertex] != vertex:
            self.parent[vertex] = self.find(self.parent[vertex])
        return self.parent[vertex]
    
    def union(self, vertex1, vertex2):
        rootV1 = self.find(vertex1)
        rootV2 = self.find(vertex2)
        if rootV1 == rootV2:
            return False
        
        if self.rank[rootV1] < self.rank[rootV2]:
            self.parent[rootV1] = rootV2
        elif self.rank[rootV1] > self.rank[rootV2]:
            self.parent[rootV2] = rootV1
        else:
            self.parent[rootV2] = rootV1
            self.rank[rootV1] += 1


def prims(G, start = None):
    if not G:
        return []

    if start is None:
        start = list(G.adj.keys())[0]

    mst = []
    
    visited = set()
    visited.add(start)
    min_heap = Heap()
    def get_edge_weights(u, v):
        if (u,v) in G.weights:
            return G.weights[(u,v)]
        elif(v,u) in G.weights:
            return G.weights[(v,u)]
        else:
            return KeyError
    
    for neighbour in G.adj[start]:
        weight = get_edge_weights(start, neighbour)
        min_heap.insert((weight, start, neighbour))
    
    while min_heap.size > 0 and len(visited) < len(G.adj):
        weight, u, v = min_heap.delete()
        if v in visited:
            continue

        visited.add(v)
        mst.append((u,v, weight))

        for node in G.adj[v]:
            if node not in visited:
                weight = get_edge_weights(v, node)
                min_heap.insert((weight, v, node))

    # borrow this implementation from class
    return mst

def kruskals(G):
    if not G.adj:
        return []
    
    mst = []
    vertices = list(G.adj.keys())
    unionFind = UnionFind(vertices)
    
    edges = []
    
    for (u,v), weight in G.weights.items():
        if u == v:  # This avoids self loops
            continue
        if u < v:   # This avoids adding the same edge twice so we only take the edge where u is the smaller vertex
            edges.append((weight, u, v))
        
    min_heap = Heap()
    min_heap.heapify(edges)

    while min_heap.size > 0 and len(mst) < len(G.adj) - 1:
        weight, u, v = min_heap.delete()

        if unionFind.find(u) != unionFind.find(v):
            mst.append((u,v, weight))
            unionFind.union(u,v)

    # your implementation for part 6 (kruskal's algorithm) goes here
    return mst

def experiment_part_6():

    # your implementation for part 6 (experiment to compare with prim's) goes here
    prims_times = []
    kruskals_times = []
    for _ in range (50):
        graph = create_random_graph(7,20, 1, 10)
        prims_graph = copy.deepcopy(graph)
        kruskals_graph = copy.deepcopy(graph)

        start = timeit.default_timer()
        prims(prims_graph)
        end = timeit.default_timer()
        prims_times.append((end - start) * 1000000)

        start = timeit.default_timer()
        kruskals(kruskals_graph)
        end = timeit.default_timer()
        kruskals_times.append((end - start) * 1000000)

    draw_plot(prims_times, np.mean(prims_times), "Prim's Heap performance", "prims.png")
    draw_plot(kruskals_times, np.mean(kruskals_times), "Kruskal's Heap performance", "kruskals.png")
    return 0

experiment_part_2()
#experiment_part_4()
#experiment_part_6()