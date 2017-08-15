import math
import os
import sys

class City:
	def __init__(self, name, x, y):
		self.name = name
		self.x = x
		self.y = y

class Edge:
	def __init__(self, city1, city2):
		self.u = city1.name
		self.v = city2.name
		self.w = self.int_distance(city1, city2)

	def int_distance(self, city1, city2):
		dist = (city1.x - city2.x)**2 + (city1.y - city2.y)**2
		dist = int(round(math.sqrt(dist)))
		return dist


class Graph:
	def __init__(self, vertices):
		self.V = vertices
		self.graph = []

	def addEdge(self, edge):
		self.graph.append([edge.u, edge.v, edge.w])

	def find(self, parent, i):
		if parent[i] == i:
			return i
		return self.find(parent, parent[i])

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

	def MST(self):
		res = []
		idx = 0
		edge = 0
		parent = []
		rank = []

		self.graph = sorted(self.graph, key=lambda x:x[2])

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
# Returns indices of odd degree vertices
def _odd_vertices(MST, V):
	odd_vertices = [0 for _ in range(V)]
	for u,v,w in MST:
		odd_vertices[u] = odd_vertices[u]+1
		odd_vertices[v] = odd_vertices[v]+1
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



def main():

# READING INPUT FILE
	if (len(sys.argv) != 2):
		print('This program requires exactly one argument. Please refer to the README') # Haven't written a README yet
		quit()
	
	fpath = sys.argv[1]
	savepath = fpath + '.tour'

	if not os.path.isfile(fpath):
		print('File not found')
		quit()

	fread = open(fpath, "r")
	#fwrite = open(savepath, "w")

	cities = []
	for line in fread:
		data = line.split()
		data = [int(val) for val in data]
		curr = City(data[0], data[1], data[2])
		cities.append(curr)

	edges = []
	for i in range(len(cities)-1):
		for j in range(i+1, len(cities)):
			edge = Edge(cities[i], cities[j])
			edges.append(edge)

	
	# BEGIN CHRISTOFIDES ALGORITHM
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



# # CONVERTING INPUT TO ADJACENCY MATRIX FORMAT
# 	adjMat = [[0 for _ in range(len(cities))] for _ in range(len(cities))]
# 	for i in range(0, len(cities)):
# 		for j in range(i+1, len(cities)):
# 			temp = cities[i].distance_to(cities[j])
# 			adjMat[i][j] = temp
# 			adjMat[j][i] = temp







if __name__ == "__main__":
	main()