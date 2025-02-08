
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
        def dfs(current, parent):   # Uses a recursive DFS to check for cycles, when it hits a node in visited, it checks to see if the node is = to itself, it's not then there has to be a cycle.
            visited.add(current)
            for node in self.graph[current]:
                if node not in visited:
                    if dfs(node, current):
                        return True
                elif node != parent:
                    return True
            return False
        
        for node in range (len(self.graph)):    # Perform this for all nodes in a graph, even in disconnected graphs
            if node not in visited:
                if dfs(node, None):
                    return True
        return False
    
    def is_connected(self,node1,node2):
        # your implementation for Part 3 goes here
        
        # First case we check to see if the two nodes are the same, if they are then of course they are connected
        if node1 == node2:
            return True

        # Keeping track of a visited set, using a set is good here because there can only be one of each item/node in there (no duplicates)
        visited = set([node1])
        q = deque([node1])  # Using a queue for BFS

        while q:    # Basically a BFS approach to find if two edges are connected, returns true if the current node it's examining is equal to the destination node
            current = q.popleft()
            if current == node2:
                return True
            for node in self.graph[current]:
                if node not in visited:
                    visited.add(node)
                    q.append(node)

        return False # Returns false if it's searched all connected nodes and the previous code did not return true
    
def BFS_2(graph,src,dst):
    path = []
    
    # Your implementation for Part 1 goes here

    if src == dst:
        return path.append(src)

    reached_end = False
    visited = set([src])

    parent = {src: None}    # Keeping track of a parent dictionary so we can reconstruct the path again at the end

    q = deque([src])

    while q:    # Normal BFS
        current = q.popleft()

        if current == dst:
            reached_end = True
            break

        for node in graph.graph[current]:
            if node not in visited:
                visited.add(node)
                parent[node] = current
                q.append(node)
    
    if reached_end: # Once we've reached the end, we reconstruct the path using the parent dictionary to get a path in reverse order
        node = dst
        while node is not None:
            path.append(node)
            node = parent[node]
        path.reverse()  # To solve this we then reverse the path to get it to return the correct order

    return path

def DFS_2(graph,src,dst):
    path = []
    # Your implementation for Part 1 goes here
    
    if src == dst:
        return path.append(src)

    visited = set()

    def dfs (node): # Using a recursive approach for this problem, we use the path itself as the stack for DFS
        visited.add(node)
        path.append(node)

        if node == dst:
            return True # Check if we have reached the end or not, it returns true so that we know the destination node is connect to the source

        for neighbor in graph.graph[neighbor]:  # Standard implementation of DFS to search for neighbors and adding them into the visited set (marking them)
            if neighbor not in visited:
                if dfs(neighbor):
                    return True
        path.pop()
        return False
    
    reached_end = dfs(src)

    if reached_end: #Will return a path if the destination node is connected to the source
        return path
    else:
        return []


def BFS_3(graph,src):
    # Your implementation for Part 2 goes here

    predecessors = {}   # Keep a dictionary of a node's predecessors, we use the searched node as the key and the current node that we are searching from as values
                        # This shows the parent nodes as values and nodes as keys 
    visited = set([src])
    q = deque([src])

    while q:    # Standard BFS implementation here
        current = q.popleft()
        for node in graph[current]:
            if node not in visited:
                visited.add(node)
                predecessors[node] = current
                q.append(node)
                
    return predecessors

def DFS_3(graph,src):
    # Your implementation for Part 2 goes here
    
    predecessors = {}   # Exact same idea as BFS_3 but using DFS with a stack insetad
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
        highest_degree = -99999 # Set initial highest degree to be a very small number
        highest_vertex = None   # Set initial highest vertex as none because we have not examined any vertices yet
        for vertex in range(len(G.graph)):
            degree = len(G.graph[vertex])   # Set the degree as the length of the graph at the specific vertex
            if degree > highest_degree and vertex not in C: # Checking the highest values and vertex and re-assign them accordingly
                highest_degree = degree 
                highest_vertex = vertex
        
        C.add(highest_vertex)

        neighbors = list(G.graph[highest_vertex])   # Now we want a list of all the neighbors of the vertex

        for node in neighbors:
            G.graph[node].remove(highest_vertex)    # Removing the edges that are incident to the highest vertex because they would just cover the same edges as the highest
        
        G.graph[highest_vertex].clear()
            
    return list(C)  

def mvc_2(G):
    # Your implementation for part 6.b goes here
    C = set()
    size = len(G.graph)
    nodes = set(range(size))

    while not is_vertex_cover(G, C):    # Checking if our current set creates an MVC
        remaining = list(nodes - C) # Take away any selected nodes so we don't randomly select a duplicate node

        if not remaining:
            break

        node = random.choice(remaining) # Pick a random node from the remaining list of nodes
        C.add(node)

    return list(C)

def mvc_3(G):
    # Your implementation for part 6.c goes here
    C = set()
    e = set()

    for u, neighbors in enumerate(G.graph): # Enumerate here gives us the index of each node which acts as the source and a list of it's neighbors which is the node it's connected to
        for v in neighbors:
            edge = (min(u,v), max(u,v))     # We then build the edge by taking the min value of the two and the max value and put them into a tuple and added to the set e.
            e.add(edge)                     # The second for loop makes sure we add all edges connected to the node u
    
    while not is_vertex_cover(G, C):
        u, v = random.choice(list(e))   # Randomly pick edges from the set e and add the nodes into the set C
        C.add(u)
        C.add(v)

        e = {edge for edge in e if u not in edge and v not in edge}

    print (list(C))

    return list(C)

def experiment_1():

    # your implementation for experiment in part 5 goes here
    count = 0
    for _ in range (100):
        graph = create_random_graph(1000,500) # Creating 100 random graphs with 1000 nodes and 500 edges.

        if graph.has_cycle():   # Checking to see if the graph has a cycle or not, if it does then we increment a running total
            count += 1
    
    proportion = count/100 # Divide the count by 100 to find the chance that a randomly generated graph has a cycle

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
            graph = create_random_graph(6,edge)     # Creating 100 random graphs of 6 edges and the corresponding edge size given from the edge_sizes list

            optimal_graph = copy.deepcopy(graph)    # Make copies of the original graph to ensure that any changes in the original graph is not reflected on the results of the following functions
            mvc1_graph = copy.deepcopy(graph)       # It essentially isolates each function to avoid any unexpected errors.
            mvc2_graph = copy.deepcopy(graph)
            mvc3_graph = copy.deepcopy(graph)

            optimal_mvc = MVC(optimal_graph)
            mvc1 = mvc_1(mvc1_graph)
            mvc2 = mvc_2(mvc2_graph)
            mvc3 = mvc_3(mvc3_graph)

            optimal_total += len(optimal_mvc)   # Running each MVC algorithm and totalling up their sizes 
            mvc1_total += len(mvc1)
            mvc2_total += len(mvc2)
            mvc3_total += len(mvc3)

        total_optimal_sizes.append(optimal_total)   # Add each of the sizes to the sizes array
        mvc1_sizes.append(mvc1_total)
        mvc2_sizes.append(mvc2_total)
        mvc3_sizes.append(mvc3_total)
    
    for i in range (len(edge_sizes)):
        mvc1_proportions.append((mvc1_sizes[i]/total_optimal_sizes[i]))     # Then to calculate the proportion we just divide each of the mvc_1,2,3 sizes with the optimal size
        mvc2_proportions.append((mvc2_sizes[i]/total_optimal_sizes[i]))
        mvc3_proportions.append((mvc3_sizes[i]/total_optimal_sizes[i]))

        mvc1_accuracies.append((total_optimal_sizes[i]/mvc1_sizes[i]) * 100)    # For accuracy we just flip the calculation of the proportion and multiply by 100 to give us a percentage
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
            graph = create_random_graph(node_sizes[i], 15)  # Exact same approach as the previous experiment but we have a list of varying node counts and a constant number of edges instead

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
        mvc1_proportions.append((mvc1_sizes[i]/total_optimal_sizes[i])) # Calculation of the optimal is the same as the previous experiment
        mvc2_proportions.append((mvc2_sizes[i]/total_optimal_sizes[i]))
        mvc3_proportions.append((mvc3_sizes[i]/total_optimal_sizes[i]))
    
    print(mvc1_proportions)
    print(mvc2_proportions)
    print(mvc3_proportions)

    

    draw_plot(node_sizes, mvc1_proportions, mvc2_proportions, mvc3_proportions, "experiment3.png", "Part 7: Experiment 3 variable node size", "Performance ratio (approx/optimal)" )

    return 0

def plot_experiment1(proportions):  # Function to plot the bar chart for experiment 1, I chose to make a seperate function since the graph for experiments 2 and 3 are quite different to experiment 1
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

def run_experiment_1(iterations):   # A function to run experiment 1 which runs it for a specific number of iterations
    proportions = []
    for _ in range (iterations):
        res = experiment_1()
        proportions.append(res) # Keeps track of all the proportions of each iteration before plotting the bar chart
    
    
    print(proportions)
    
    plot_experiment1(proportions)
    

    return

#run_experiment_1(80)
#experiment_2()
experiment_3()



# Please feel free to include other experiments that support your answers. 
# Or any other experiments missing 