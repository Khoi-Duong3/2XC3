
import random
import time
import timeit 
import matplotlib.pyplot as plt
import numpy as np
from collections import deque
from typing import List

# Utility functions - some are implemented, others you must implement yourself.

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

# function to generate random graphs 
# @args : nodes = number of nodes 
#       : edges = number of edges
def create_random_graph(nodes, edges):

    # your implementation for Part 4 goes here
    
    graph = [[] for i in range (nodes)]
    max_number_edges = nodes * (nodes - 1) // 2

    if edges > max_number_edges:
        return -1

    added = set()

    while len(added) < edges:
        start = random.randrange(nodes)
        end = random.randrange(nodes)

        if start == end:
            continue
        
        edge = (min(start, end), max(start, end))

        if edge not in added:
            added.add(edge)
            graph[start].append(end)
            graph[end].append(start)

    return graph

# please select one representation from either Graph I or Graph II 
# for this assignment,
# please remove the representation you are not using

# graph implementation using hash map 

class GraphI:

    # using hash map
    def __init__(self, edges):
        self.graph = {}
        for x,y in edges:
            if x not in self.graph.keys():
                self.graph[x]=[]
            self.graph[x].append(y)

    def has_edge(self, src, dst):
        return dst in self.graph[src]

    def get_graph_size(self,):
        return len(self.graph)
    
    def get_graph(self,):
        return self.graph
    
    def has_cycle(self,):
        # your implementation for Part 3 goes here
        return False
    
    def is_connected(self,node1,node2):
        # your implementation for Part 3 goes here
        return False

# graph implementation using adjacency list   
class GraphII:
    # using adjacency list
    def __init__(self, nodes):
        self.graph = []
        # node numbered 0-1
        for node in range(nodes):
            self.graph.append([])
        
    def has_edge(self, src, dst):
        return src in self.graph[dst]
    
    def add_edge(self,src,dst):
        if not self.has_edge(src,dst):
            self.graph[src].append(dst)
            
            if src != dst:
                self.graph[dst].append(src)
    
    def get_graph(self,):
        return self.graph
    
    def has_cycle(self,):
        visited = set()
        # your implementation for Part 3 goes here
        def dfs(current, parent):
            visited.add(current)
            for node in self.graph[current]:
                if node not in visited:
                    if dfs(node, current):
                        return True
                elif node != parent:
                    return True
            return False
        
        for node in range (len(self.graph)):
            if node not in visited:
                if dfs(node, None):
                    return True
        return False
    
    def is_connected(self,node1,node2):
        # your implementation for Part 3 goes here
        if node1 == node2:
            return True

        visited = set([node1])
        q = deque([node1])

        while q:
            current = q.popleft()
            if current == node2:
                return True
            for node in self.graph[current]:
                if node not in visited:
                    visited.add(node)
                    q.append(node)

        return False
    
def BFS_2(graph,src,dst):
    path = []
    
    # Your implementation for Part 1 goes here

    if src == dst:
        return path.append(src)

    reached_end = False
    visited = set([src])

    parent = {src: None}

    q = deque([src])

    while q:
        current = q.popleft()

        if current == dst:
            reached_end = True
            break

        for node in graph.graph[current]:
            if node not in visited:
                visited.add(node)
                parent[node] = current
                q.append(node)
    
    if reached_end:
        node = dst
        while node is not None:
            path.append(node)
            node = parent[node]
        path.reverse()

    return path

def DFS_2(graph,src,dst):
    path = []
    # Your implementation for Part 1 goes here
    
    if src == dst:
        return path.append(src)

    visited = set()

    def dfs (node):
        visited.add(node)
        path.append(node)

        if node == dst:
            return True

        for neighbor in graph.graph[neighbor]:
            if neighbor not in visited:
                if dfs(neighbor):
                    return True
        path.pop()
        return False
    
    reached_end = dfs(src)

    if reached_end:
        return path
    else:
        return []


def BFS_3(graph,src):
    # Your implementation for Part 2 goes here

    predecessors = {}

    visited = set([src])
    q = deque([src])

    while q:
        current = q.popleft()
        for node in graph[current]:
            if node not in visited:
                visited.add(node)
                predecessors[current] = node
                q.append(node)
                
    return predecessors

def DFS_3(graph,src):
    # Your implementation for Part 2 goes here
    
    predecessors = {}
    visited = set()
    
    def dfs (current, parent):
        visited.add(current)

        if parent is not None:
            predecessors[current] = parent
        for neighbor in graph[current]:
            if neighbor not in visited:
                dfs(neighbor, current)
    
    dfs(src, None)

    return predecessors

#Utility functions to determine minimum vertex covers
def add_to_each(sets, element):
    copy = sets.copy()
    for set in copy:
        set.append(element)
    return copy

def power_set(set):
    if set == []:
        return [[]]
    return power_set(set[1:]) + add_to_each(power_set(set[1:]), set[0])

def is_vertex_cover(G, C):
    for u, neighbors in enumerate(G.graph):
        for v in neighbors:
            if u not in C and v not in C:
                return False
    return True

def MVC(G):
    nodes = [i for i in range(G.get_size())]
    subsets = power_set(nodes)
    min_cover = nodes
    for subset in subsets:
        if is_vertex_cover(G, subset):
            if len(subset) < len(min_cover):
                min_cover = subset
    return min_cover

def mvc_1(G):
    # Your implementation for part 6.a goes here
    C = set()

    while not is_vertex_cover(G, C):
        highest_degree = -99999
        highest_vertex = None
        for vertex in range(len(G.graph)):
            degree = len(G.graph[vertex])
            if degree > highest_degree and vertex not in C:
                highest_degree = degree
                highest_vertex = vertex
        
        C.add(highest_vertex)

        neighbors = list(G.graph[highest_vertex])

        for node in neighbors:
            G.graph[node].remove(highest_vertex)
        
        G.graph[highest_vertex].clear()
            
    return list(C)

def mvc_2(G):
    # Your implementation for part 6.b goes here
    C = set()
    size = len(G.graph)
    nodes = set(range(size))

    while not is_vertex_cover(G, C):
        remaining = list(nodes - C)

        if not remaining:
            break

        node = random.choice(remaining)
        C.add(node)

    return list(C)

def mvc_3(G):
    # Your implementation for part 6.c goes here
    C = set()
    e = set()

    for u, neighbors in enumerate(G.graph):
        for v in neighbors:
            edge = (min(u,v), max(u,v))
            e.add(edge)
    
    while not is_vertex_cover(G, C):
        u, v = random.choice(list(e))
        C.add(u)
        C.add(v)

        e = {edge for edge in e if u not in edge and v not in edge}

    print (list(C))

    return list(C)

def experiment_1():

    # your implementation for experiment in part 5 goes here
    count = 0
    for _ in range (100):
        graph = create_random_graph(1000,500) 

        adj_list = GraphII(1000)

        adj_list.graph = graph

        if adj_list.has_cycle():
            count += 1
    
    proportion = count/100 

    return round(proportion, 2)

def experiment_2():

    # your implementation for experiment in part 7 goes here

    edge_sizes = [1,5,10,15,20]
    total_optimal_sizes = []
    mvc1_sizes = []
    mvc2_sizes = []
    mvc3_sizes = []

    def run_experiment_2 ():
        for edges in edge_sizes:
            total_optimal = 0
            mvc1_total = 0
            mvc2_total = 0
            mvc3_total = 0
            for _ in range(100):
                graph = create_random_graph(6,edges)
                total_optimal += len(MVC(graph))
                mvc1_total += len(mvc_1(graph))
                mvc2_total += len(mvc_2(graph))
                mvc3_total += len(mvc_3(graph))
            
            total_optimal_sizes.append(total_optimal)
            mvc1_sizes.append(mvc1_total)
            mvc2_sizes.append(mvc2_total)
            mvc3_sizes.append(mvc3_total)
    
    run_experiment_2()
    
    expected_perforamnces_mvc1 = []
    expected_perforamnces_mvc2 = []
    expected_perforamnces_mvc3 = []
    
    for i in range (len(total_optimal_sizes)):
        expected_perforamnces_mvc1.append(mvc1_sizes[i]/total_optimal_sizes[i])
        expected_perforamnces_mvc2.append(mvc1_sizes[i]/total_optimal_sizes[i])
        expected_perforamnces_mvc3.append(mvc1_sizes[i]/total_optimal_sizes[i])
            
    return True

def experiment_3():

    # your implementation for any other 
    # supplemental experiments you need to run goes here (e.g For question 7.c)
    return True


def run_experiment_1(iterations):
    proportions = []
    for _ in range (iterations):
        res = experiment_1()
        proportions.append(res)
    
    print(proportions)

    return

run_experiment_1(100)
random_graph = create_random_graph(20,7)
random_adj = GraphII(20)
random_adj.graph = random_graph
mvc_3(random_adj)

# Please feel free to include other experiments that support your answers. 
# Or any other experiments missing 