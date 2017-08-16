import math
import os
import sys


#Represents a city.
# Has three attributes
#   [1] name, an integer identifer
#   [2] x, the integer x-position
#   [3] y, the integer y-position
class City:
	def __init__(self, name, x, y):
		self.name = name
		self.x = x
		self.y = y

#Represents an edge
# has three attributes
#   [1] u, the name of one City of the Edge.
#   [2] v, the name of the other City of the Edge
class Edge:
	def __init__(self, city1, city2):
		self.u = city1.name
		self.v = city2.name
		self.w = self.int_distance(city1, city2)

        # Calculates Euclidean distance between 2 Cities based on their positions
	def int_distance(self, city1, city2):
		dist = (city1.x - city2.x)**2 + (city1.y - city2.y)**2
		dist = int(round(math.sqrt(dist)))
		return dist

        def swap(self):
            tmp = self.u
            self.u = self.v
            self.v = tmp

        def copy(self):
            edge = Edge(City(1, 1, 1) )
            edge.u = self.u
            edge.v = self.v
            edge.w = self.w
            return edge

#Represents a Graph by storing all the Edges of that graph
# has 2 attributes
#   [1] V, the number of vertices in the graph
#   [2] graph, a list of edge properties that make up the Graph
class Graph:
	def __init__(self, vertices):
		self.V = vertices
		self.graph = []

        # Adds a list of edge properties to the list of edges:
        #   edge.u: id of one city
        #   edge.v: id of another city
        #   edge.w: DISTANCE between cities
	def addEdge(self, edge):
		self.graph.append([edge.u, edge.v, edge.w])

        #Finds the name (ID) of something??
        # Use: MST only
	def find(self, parent, i):
		if parent[i] == i:
			return i
		return self.find(parent, parent[i])

        # Merges two Graphs on the condition that they are TREES
        # Use: MST only
	def union(self, parent, rank, x, y):
		root_x = self.find(parent, x)
		root_y = self.find(parent, y)
		if rank[root_x] < rank[root_y]:
			parent[root_x] = root_y
		elif rank[root_x] > rank[root_y]:
			parent[root_y] = root_x
		else:
			parent[root_y] = root_x
			rank[root_x] += 1

        # Uses Kruskal's algorithm to find an MST of the Graph (self)
        # Returned as a list of edges [u, v, w]
	def MST(self):
		res = []
		idx = 0
		edge = 0   #Keeps track of the edges we are processing
		parent = [] 
		rank = []

                # Sorts the list of edges by weight
		self.graph = sorted(self.graph, key=lambda x:x[2])

                #Hard for me to tell if the remainder is correctly
                # implementing Kruskal's algorithm (which is complex anyways
		for node in range(self.V):
			parent.append(node)
			rank.append(0)

		while edge < self.V-1:
			u,v,w = self.graph[idx]
			idx += 1
			x = self.find(parent, u)
			y = self.find(parent, v)

			if x != y:
				edge = edge + 1
				res.append([u,v,w])
				self.union(parent, rank, x, y)
		return res


# Takes as input an MST (Graph class) and V = number of vertices
# Returns a list of IDs of vertices with odd degree
def _odd_vertices(MST, V):
	odd_vertices = [0 for _ in range(V)]
        #Create a list of the degrees of each vertex
	for u,v,w in MST:
		odd_vertices[u] = odd_vertices[u]+1
		odd_vertices[v] = odd_vertices[v]+1
        #Filter out the indices of even degree
	odd_vertices = [vertex for vertex, degree in enumerate(odd_vertices) if degree%2 == 1]
	return odd_vertices


# Takes as input a list of cities (City class)
# Returns a list of cities (City class) in order of visit
def nearest_neighbors(odd_cities):
	unvisited = odd_cities
	visited = [unvisited.pop()]
	while unvisited:
		next_city = min(unvisited, key=lambda x: Edge(visited[-1], x).w)
		visited.append(next_city)
		unvisited.remove(next_city)
	return visited

#Takes as (hopefully) a list of Cities that make up a tour
# Returns the perfect matching edge set
# There MUST be an even number of odd-degree vertices in a graph.
def compute_pm(tour):
    # This algorithm fails for small num_vertices
    num_vertices = len(tour)
    #Calculate the weight of the first "perfect matching"
    perfect_match_1 = []
    M1 = 0
    i = 0
    while i < (num_vertices - 1):
        M1 = M1 + euc_dist(tour[i], tour[i+1])
        i = i + 2
        perfect_match_1.append(Edge( tour[i], tour[i+1] ))

    #Calculate the weight of the second "perfect matching"
    # TOO BAD THE PAPER DESCRIPTION SUCKS DONKEY @@@@
    perfect_match_2 = []
    M2 = 0
    i = 1
    while i < (num_vertices - 2):
        M2 = M2 + euc_distance(tour[i], tour[i+1])
        i = i + 1
        perfect_match_2.append(Edge(tour[i], tour[i+1] ) )
    M2 = M2 + euc_distance(tour[0], tour[num_vertices - 1] )
    perfect_match_2.append(Edge(tour[0], tour[num_vertices - 1] ))

    if M1 < M2:
        return perfect_match_1
    else:
        return perfect_match_2

#Calculates euclidean distance between any 2 City objects
def euc_dist(city1, city2):
        dist = (city1.x - city2.x)**2 + (city1.y - city2.y)**2
        dist = int(round(math.sqrt(dist)))
        return dist

def main():

# READING INPUT FILE
	if (len(sys.argv) != 2):
		print('This program requires exactly one argument. Please refer to the README') # Haven't written a README yet
		quit()
	
	fpath = sys.argv[1]

	if not os.path.isfile(fpath):
		print('File not found')
		quit()

	fread = open(fpath, "r")
	#fwrite = open(savepath, "w")

	cities = []
	for line in fread:
		data = line.split()
                # turn all the data into integers
		data = [int(val) for val in data]
                #           id      x-pos       y-pos
		curr = City(data[0], data[1], data[2])
                #Add to the list of cities
		cities.append(curr)

        #At this point we have a complete list of cities
        #Time to build edges, 1 edge per pair of cities
	edges = []
	for i in range(len(cities)-1):
		for j in range(i+1, len(cities)):
			edge = Edge(cities[i], cities[j])
			edges.append(edge)

        #Now we have a list of cities and a list of all possible edges between them.	
	# We fuse these two lists into one data structure, a Graph
	G = Graph(len(cities))
	for edge in edges:
		G.addEdge(edge)


	# STEP 0: COMPUTE MST
	MST = G.MST()
	print("MST: ")
	print(MST)


	# STEP 1: FIND SET OF VERTICES (odd_vertices) WITH ODD DEGREE FROM MST
	odd_vertices = _odd_vertices(MST, len(cities))
	print("\n\nOdd vertices:")
	print(odd_vertices)
	
	odd_cities = [cities[i] for i in odd_vertices]
		

	# STEP 2: COMPUTE MINIMUM WEIGHT PERFECT MATCHING
	# STEP 2.1: FIND A TOUR OVER G* (complete graph containing odd degree cities) USING NEAREST NEIGHBORS HEURISTIC
	tour_C = nearest_neighbors(odd_cities)
	print("\n\nNearest neighbors tour:")
	view_tour_C = [city.name for city in tour_C]
	print(view_tour_C)
        perfect_match = compute_pm(tour_C)

    #merge the edges of the mst and the perfect match
	merged_graph = []
	for i in range(len(MST)):
		newEdge = Edge(City(MST[i][0], 1, 1), City(MST[i][1], 1, 1)) #dummy city objects with dummy x and y coordinates
		newEdge.w = MST[i][2]
		merged_graph[i] = newEdge
	for edge in perfect_match:
		merged_graph.append(edge))

        #The graph should only have vertices of even degree now, so
        #it must have an Eulierian circuit, which is visits each edge exactly once
        #To do this, first convert the graph to an adjacency list
        adjacency_list = convert(merged_graph)

        euler_circuit = hierholzer(adjacency_list)

        tour = clean_up(eurler_circuit)

	savepath = fpath + '.tour'
        write_to_file(tour, savepath)


# # CONVERTING INPUT TO ADJACENCY MATRIX FORMAT
# 	adjMat = [[0 for _ in range(len(cities))] for _ in range(len(cities))]
# 	for i in range(0, len(cities)):
# 		for j in range(i+1, len(cities)):
# 			temp = cities[i].distance_to(cities[j])
# 			adjMat[i][j] = temp
# 			adjMat[j][i] = temp

#As input, takes a list of Edge objects
def convert(merged_graph, numVertices):
   #First, initialize a properly sized list of lists
    adjacency = []
    for i in range(numVertices):
        adjacency.append([])

    #next, go through the list of Edge objects and update 
    #   the adjacency list accordingly
    for edge in merged_graph:
        adjacency[edge.u].append(edge)
        adjacency[edge.v].append(edge.copy().swap())

    #Return the adjacency list
    return adjacency
    
# Takes an adjacency list, where each index of the list represents a vertex
# and each index of the list contains a list of Edge objects that start
# at that vertex
#Returns an ordered list of the edges in an Euler circuit of the graph
# Uses the hierholzer algorithm for generating an Euler circuit
def hierholzer(adjacency_list):
    length = len(adjacency_list):
    circuit = []
    #Choose a starting vertex
    start = 0
    cur = 0

    # While there are still unvisited edges on the starting vertex
    # do a cycle that ends back on the starting vertex
    while start != length:
        first_edge = adjacency_list[start].pop()
        circuit.append(first_edge)
        #Remove the duplicate edge in the other vertex's adjacency list
        for dup in adjacency_list[first_edge.v]:
            if dup.v == cur:
                adjacency_list[edge.v].remove(dup)
                break
        cur = first_edge.v 

        while cur != start:
            edge = adjacency_list[cur].pop()
            circuit.append(edge)
            #Remove the duplicate edge in the other vertex's adjacency list
            for dup in adjacency_list[edge.v]:
                if dup.v == cur:
                    adjacency_list[edge.v].remove(dup)
            cur = edge.v

        #Once we get back to the starting vertex, we do another cycle
        #   if it still has more edges in its adjacency list.
        #   Otherwise we search for the next vertex that still has more
        #   edges in its adjacency list
        if len(adjacency_list[start]) == 0:
            start += 1
    return circuit






if __name__ == "__main__":
	main()
