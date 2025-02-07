
import random
import time
import timeit 
import matplotlib.pyplot as plt
import numpy as np
from collections import deque
from typing import List
import copy

# Utility functions - some are implemented, others you must implement yourself.

# function to plot the bar graph and average runtimes of N trials
# Please note that this function only plots the graph and does not save it
# To save the graphs you must use plot.save(). Refer to matplotlib documentation
def draw_plot(sizes, mvc1_proportions, mvc2_proportions, mvc3_proportions, filename, title, y_label):
    """
    Plots a bar chart with all bars for mvc_1 first, then mvc_2, then mvc_3.
    
    Each bar is annotated with its label (e.g., "mvc_1(20)") horizontally at the bottom.
    A dashed average line is drawn over each block (group) of bars, each with its own color.
    The legend displays all three average lines.
    
    Parameters:
      sizes: list of the independent variable values (e.g. edge counts or node sizes).
      mvc1_proportions, mvc2_proportions, mvc3_proportions: lists of performance ratios
         (one value per size) for each MVC method.
      filename: the filename to save the figure.
      title: title for the plot.
    """
    n = len(sizes)           # number of bars per MVC method
    total_bars = 3 * n       # three MVC methods in sequence
    x = np.arange(total_bars)
    bar_width = 0.8

    plt.figure(figsize=(20, 8))
    
    # Plot bars for mvc_1: indices 0 to n-1
    plt.bar(x[0:n], mvc1_proportions, width=bar_width, color='blue', label='mvc_1')
    # Plot bars for mvc_2: indices n to 2*n - 1
    plt.bar(x[n:2*n], mvc2_proportions, width=bar_width, color='green', label='mvc_2')
    # Plot bars for mvc_3: indices 2*n to 3*n - 1
    plt.bar(x[2*n:3*n], mvc3_proportions, width=bar_width, color='orange', label='mvc_3')
    
    # Annotate each bar with its label, placing the text just below the top of the bar.
    for i in range(n):
        plt.text(x[i], mvc1_proportions[i], f"mvc_1({sizes[i]})", ha='center', va='bottom', 
                 rotation=0, fontsize=8)
    for i in range(n):
        plt.text(x[n + i], mvc2_proportions[i], f"mvc_2({sizes[i]})", ha='center', va='bottom', 
                 rotation=0, fontsize=8)
    for i in range(n):
        plt.text(x[2*n + i], mvc3_proportions[i], f"mvc_3({sizes[i]})", ha='center', va='bottom', 
                 rotation=0, fontsize=8)
    
    # Compute the average performance ratio for each MVC method.
    avg_mvc1 = np.mean(mvc1_proportions)
    avg_mvc2 = np.mean(mvc2_proportions)
    avg_mvc3 = np.mean(mvc3_proportions)
    
    # Draw average lines across the entire block for each MVC.
    # For mvc_1, the x-range is from x[0] - half bar width to x[n-1] + half bar width.
    plt.axhline(avg_mvc1, color = "red", linestyle = "--", linewidth = 2, label = "Avg mvc_1")
    plt.axhline(avg_mvc2, color = "purple", linestyle = "--", linewidth = 2, label = "Avg mvc_2")
    plt.axhline(avg_mvc3, color = "black", linestyle = "--", linewidth = 2, label = "Avg mvc_3")
        
    # Set x-axis ticks to show the MVC groups in the middle of each block.
    tick_positions = [ (x[0] + x[n-1]) / 2, (x[n] + x[2*n-1]) / 2, (x[2*n] + x[3*n-1]) / 2 ]
    plt.xticks(tick_positions, ['mvc_1', 'mvc_2', 'mvc_3'])
    
    plt.xlabel('MVC Type')
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.savefig(filename)
    plt.show()

    return

# function to generate random graphs 
# @args : nodes = number of nodes 
#       : edges = number of edges
def create_random_graph(nodes, edges):

    # your implementation for Part 4 goes here
    
    graph = GraphII(nodes)
    max_number_edges = nodes * (nodes + 1) // 2

    if edges > max_number_edges:
        return -1

    added = set()

    while len(added) < edges:
        start = random.randrange(nodes)
        end = random.randrange(nodes)
        
        edge = (min(start, end), max(start, end))

        if edge not in added:
            added.add(edge)
            graph.add_edge(start, end)

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
    nodes = [i for i in range(len(G.graph))]
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

        if graph.has_cycle():
            count += 1
    
    proportion = count/100 

    return proportion

def experiment_2():

    # your implementation for experiment in part 7 goes here

    edge_sizes = [1,5,10,15,20]
    total_optimal_sizes = []
    mvc1_sizes = []
    mvc2_sizes = []
    mvc3_sizes = []

    mvc1_proportions = []
    mvc2_proportions = []
    mvc3_proportions = []

    mvc1_accuracies = []
    mvc2_accuracies = []
    mvc3_accuracies = []

    for edge in edge_sizes:
        optimal_total = 0
        mvc1_total = 0
        mvc2_total = 0
        mvc3_total = 0
        for _ in range (100):
            graph = create_random_graph(6,edge)

            optimal_graph = copy.deepcopy(graph)
            mvc1_graph = copy.deepcopy(graph)
            mvc2_graph = copy.deepcopy(graph)
            mvc3_graph = copy.deepcopy(graph)

            optimal_mvc = MVC(optimal_graph)
            mvc1 = mvc_1(mvc1_graph)
            mvc2 = mvc_2(mvc2_graph)
            mvc3 = mvc_3(mvc3_graph)

            optimal_total += len(optimal_mvc)
            mvc1_total += len(mvc1)
            mvc2_total += len(mvc2)
            mvc3_total += len(mvc3)

        total_optimal_sizes.append(optimal_total)
        mvc1_sizes.append(mvc1_total)
        mvc2_sizes.append(mvc2_total)
        mvc3_sizes.append(mvc3_total)
    
    for i in range (len(edge_sizes)):
        mvc1_proportions.append((mvc1_sizes[i]/total_optimal_sizes[i]))
        mvc2_proportions.append((mvc2_sizes[i]/total_optimal_sizes[i]))
        mvc3_proportions.append((mvc3_sizes[i]/total_optimal_sizes[i]))

        mvc1_accuracies.append((total_optimal_sizes[i]/mvc1_sizes[i]) * 100)
        mvc2_accuracies.append((total_optimal_sizes[i]/mvc2_sizes[i]) * 100)
        mvc3_accuracies.append((total_optimal_sizes[i]/mvc3_sizes[i]) * 100)
    
    print(mvc1_proportions)
    print(mvc2_proportions)
    print(mvc3_proportions)

    print(mvc1_accuracies)
    print(mvc2_accuracies)
    print(mvc3_accuracies)

    draw_plot(edge_sizes, mvc1_proportions, mvc2_proportions, mvc3_proportions, "experiment2proportion.png", "Part 7: Experiment 2 variable edge size proportion", "Performance ratio (approx/optimal)")
    draw_plot(edge_sizes, mvc1_accuracies, mvc2_accuracies, mvc3_accuracies, "experiment2accuracy.png", "Part 7: Experiment 2 variable edge size accuracy", "Accuracy (%)")

    return True

def experiment_3():

    # your implementation for any other 
    # supplemental experiments you need to run goes here (e.g For question 7.c)

    node_sizes = [5,7,9,11,13]

    total_optimal_sizes = []
    mvc1_sizes = []
    mvc2_sizes = []
    mvc3_sizes = []

    mvc1_proportions = []
    mvc2_proportions = []
    mvc3_proportions = []

    for i in range(len(node_sizes)):
        optimal_total = 0
        mvc1_total = 0
        mvc2_total = 0
        mvc3_total = 0
        for _ in range (100):
            graph = create_random_graph(node_sizes[i], 15)

            optimal_graph = copy.deepcopy(graph)
            mvc1_graph = copy.deepcopy(graph)
            mvc2_graph = copy.deepcopy(graph)
            mvc3_graph = copy.deepcopy(graph)

            optimal_mvc = MVC(optimal_graph)
            mvc1 = mvc_1(mvc1_graph)
            mvc2 = mvc_2(mvc2_graph)
            mvc3 = mvc_3(mvc3_graph)

            optimal_total += len(optimal_mvc)
            mvc1_total += len(mvc1)
            mvc2_total += len(mvc2)
            mvc3_total += len(mvc3)
        
        total_optimal_sizes.append(optimal_total)
        mvc1_sizes.append(mvc1_total)
        mvc2_sizes.append(mvc2_total)
        mvc3_sizes.append(mvc3_total)

    for i in range (len(node_sizes)):
        mvc1_proportions.append((mvc1_sizes[i]/total_optimal_sizes[i]))
        mvc2_proportions.append((mvc2_sizes[i]/total_optimal_sizes[i]))
        mvc3_proportions.append((mvc3_sizes[i]/total_optimal_sizes[i]))
    
    print(mvc1_proportions)
    print(mvc2_proportions)
    print(mvc3_proportions)

    

    draw_plot(node_sizes, mvc1_proportions, mvc2_proportions, mvc3_proportions, "experiment3.png", "Part 7: Experiment 3 variable node size", "Performance ratio (approx/optimal)" )

    return 0

def plot_experiment1(proportions):
    avg_proportion = np.mean(proportions)
    iterations = len(proportions)
    x = np.arange(iterations) + 1

    plt.figure(figsize=(20, 8))
    plt.bar(x, proportions, color='blue')
    plt.axhline(y=avg_proportion, color='red', linestyle='--', linewidth=2, 
                label=f'Average = {avg_proportion:.2f}')
    
    plt.xlabel('Iteration')
    plt.ylabel('Proportion')
    plt.title('Part 5: Experiment 1')
    plt.legend()
    plt.savefig("experiment_1.png")
    plt.show()

    return

def run_experiment_1(iterations):
    proportions = []
    for _ in range (iterations):
        res = experiment_1()
        proportions.append(res)
    
    
    print(proportions)
    
    plot_experiment1(proportions)
    

    return

#run_experiment_1(80)
#experiment_2()
experiment_3()



# Please feel free to include other experiments that support your answers. 
# Or any other experiments missing 