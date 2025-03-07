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

    # The same create random graph function in Lab 2. 
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

    #The same create random list function from Lab 1

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
            # We check here if we've gone out of range which we would then set mid to 0
            if mid < 0:
                mid = 0
            # Check here as well if mid is out of range past the length of the list then we set it back to the final item in the list
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
            # The insert method here performs insert for us by shifting elements to the right to make space for us to insert the item into the right place
            # We use L.pop(i) here to remove the item from the list at that index so that we don't end up with duplicate elements after the insert
            L.insert(index, L.pop(i))

    return

def insertion_sort(L):
    # Standard insertion sort from Lab 1
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
    '''
    # This is a test for the average case
    # The average case for these sorts would be random since the it avoids the worst case and best case for insertion sort.
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
    
    draw_plot(insertion_times, np.mean(insertion_times), "Insertion Sort Best Case", "insertion_sort_best.png")
    draw_plot(hybrid_times, np.mean(hybrid_times), "Hybrid Sort Best Case", "Hybrid_sort_best.png")

    return 0

# binary search dynamic
class DynamicArrays():
    def __init__(self):
        self.arrays = []
        self.elements = 0

    # your implementation for part 3 goes here
    # feel free to add arguments and helper functions that you need to add

    #Standard binary search implementation
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

    # Iterating through all the arrays in the dynamic array, then we perform binary search on each array if they are not empty
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
            # First we check if our index is larger than the number of arrays in the data structure
            # If it is then we append in the new_array which is a singleton array
            if i >= len(self.arrays):
                self.arrays.append(new_array)
                break
            # Check if at the current index it is None which we would then assign the singleton array at that index
            if self.arrays[i] is None:
                self.arrays[i] = new_array
                break
            else:
                # Otherwise we merge the two arrays with each other and set the new_array to be the merged array
                merged = self.merge_arrays(self.arrays[i], new_array)
                self.arrays[i] = None
                new_array = merged
                i += 1
        
        
        self.elements += 1
        
        return
    
    def delete(self, item):
        # Check if we are trying to delete an item that is not in our data structure
        if not self.search(item):
            return
        
        deleted = []
        # The approach for this is to add every single item from the data structure into a normal array
        for array in self.arrays:
            if array:
                i = 0
                while i < len(deleted):
                    deleted.append(array[i])
                    i += 1

        # Then we remove the item we want to delete from this array
        deleted.remove(item)
        self.arrays = []        # This line clears out the current dynamic array by reinitializing it
        self.elements = 0
        # Re-insert all of the items back into the data structure
        for element in deleted:
            self.insert(element)

        return 
    
    # Calls search to search in the dynamic array. I don't know what the difference between this and search.
    def binary_search_dynamic(self, target):
        return self.search(target)
    
def insert_normal(L, element):
    low = 0
    high = len(L)
    while low < high:
        mid = (low + high) // 2
        if L[mid] < element:
            low = mid + 1
        else:
            high = mid - 1
    
    index = low

    L.append(element)

    for i in range (len(L) - 1,index, -1):
        L[i] = L[i-1]

    return

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
    
    insert_normal_times = []
    dynamic_array_insert = []

    for _ in range (30):
        L = []
        start = timeit.default_timer()
        for _ in range(500):
            num = random.randint(0,5000)
            insert_normal(L, num)
        end = timeit.default_timer()
        insert_normal_times.append((end - start) * 1000000)

    for _ in range (30):
        Dynamic = DynamicArrays()
        start = timeit.default_timer()
        for _ in range(500):
            num = random.randint(0,5000)
            Dynamic.insert
        end = timeit.default_timer()
        dynamic_array_insert.append((end - start) * 1000000)


    draw_plot(insert_normal_times, np.mean(insert_normal_times), "Insert on normal array performance", "insert1.png")
    draw_plot(dynamic_array_insert, np.mean(dynamic_array_insert), "Insert on dynamic array performance", "dynamicinsert.png")    
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

    # Performs bubble up and recursively calls itself until a node can no longer swim up which maeans it's in the right place
    def swim(self, index):
        parent_index = (index - 1) // 2
        if index > 0 and self.heap[index] < self.heap[parent_index]:
            self.heap[index], self.heap[parent_index] = self.heap[parent_index], self.heap[index]
            self.swim(parent_index)

        return

    # Performs bubble down and recursively calls itself until the node is in the right place
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
    
    # Insert it at the bottom of the heap then we swim it up to its correct position
    def insert(self, value):
        self.heap.append(value)
        self.size += 1
        self.swim(self.size - 1)

        return
    
    # Simple swap helper function
    def swap(self, index1, index2):
        self.heap[index1], self.heap[index2] = self.heap[index2], self.heap[index1]

        return
    
    # Move the top node in the heap down to the bottom then we sink the new item down to the right place.
    # We also decrease the size of our heap after we pop the last item from our heap since the last item now is the item we want to delete
    def delete(self):
        self.swap(0, self.size - 1)
        min_value = self.heap.pop()
        self.size -= 1
        self.sink(0)

        return min_value
    
    # Heapify builds a heap from a normal list
    def heapify(self, array):
        self.heap = array
        self.size = len(array)
        for i in range(self.size // 2 - 1, -1, -1):
            self.sink(i)

        return

class UnionFind():
    def __init__(self, vertices):
        # Initially, all vertices' parent is itself
        self.parent = {i: i for i in vertices}
        self.rank = {i: 0 for i in vertices}

    # Recursively finds the parent of the parent of a node until it reaches the root
    def find(self, vertex):
        if self.parent[vertex] != vertex:
            self.parent[vertex] = self.find(self.parent[vertex])
        return self.parent[vertex]
    
    # Performs weight union where it adds the set/tree with the lower rank to the one with a higher rank by changing the parent of the root to the be the root
    # of the larger tree/set
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
    # Return nothing on an empty graph
    if not G:
        return []

    # If we are not specified a starting point/node/vertex we default to the first node in the graph
    if start is None:
        start = list(G.adj.keys())[0]

    mst = []
    
    visited = set()
    visited.add(start)
    min_heap = Heap()

    # This helper function finds the edge weights from our graph at a specific edge u,v which it then returns the weight
    def get_edge_weights(u, v):
        if (u,v) in G.weights:
            return G.weights[(u,v)]
        elif(v,u) in G.weights:
            return G.weights[(v,u)]
        else:
            return KeyError
    
    # Gather all of the weights of each edge that connects the current node to its neighbors and we add them to the heap
    for neighbour in G.adj[start]:
        weight = get_edge_weights(start, neighbour)
        min_heap.insert((weight, start, neighbour))
    
    # Then we go through the initial heaps that contain the edges that connect the starting point to the neighbor and their weights
    # We then remove the root of the heap then check to see if the destination has already been visited or not, if it has then we skip over it and check on the next edge that gets bubbled up
    # If it has not been visited then we add it to our visited set and mst then we collect all of the edges that is connected to this new node and we repeat until our heap is empty
    # or until our MST contains all nodes in the graph
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
    # Return empty if the graph is empty
    if not G.adj:
        return []
    
    mst = []
    vertices = list(G.adj.keys())
    unionFind = UnionFind(vertices) #Initialize our union find data structure to perform cuts between connect set of the mst and the rest of the graph
    
    edges = []
    
    # Adding all of the edges to the edges array so that we can perform heapify on it to turn it into a heap
    for (u,v), weight in G.weights.items():
        if u == v:  # This avoids self loops
            continue
        if u < v:   # This avoids adding the same edge twice so we only take the edge where u is the smaller vertex
            edges.append((weight, u, v))
        
    min_heap = Heap()
    min_heap.heapify(edges)

    # Looping until the heap is empty or the size of the mst contains all of the nodes in the graph
    while min_heap.size > 0 and len(mst) < len(G.adj) - 1:
        weight, u, v = min_heap.delete()    # Removing the edge from the heap

        # Only add the edge and node into the mst if and only if they have different roots which means one of them is in the mst and the other isn't
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