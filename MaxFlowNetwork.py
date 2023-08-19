from collections import deque

class Graph:
    def __init__(self, connections, maxIn, maxOut, origin, targets) -> None:
        """
        Function Description: This function makes a graph with the verticies for MaxIn, connections and MaxOut
        
        Approach Description: In order to make the graph consider all the factors it needs to coneect MaxIn, connections, and 
        MaxOut. So need to make a connection between all the factors.   
        
        since we determine the max number of vertices by looping through connections this takes O(C) time. The also make two arrays
        of the size it's length which take O(L) time. 

        When we loop through the to make the 3 arrays for size L so we are taking O(L) space as we are making an array of the size lenght vertices
        
        :param connections: list of connections, maxIn: list of maxIn, maxOut: list of maxOut, origin: an integer respensenting staring point, 
        targets: the possble targets of the flow network
        :return: None 

        Time Complexity: O(L+C) where R is lenght of vertices and C is the length of connections
        Space Complexity: O(L) where L is lenght of vertices
        """
        self.connections = connections
        self.maxIn = maxIn
        self.maxOut = maxOut
        self.origin = origin
        self.targets = targets
        max_vertices = 0

        # Find the max number of vertices
        for i in range(len(connections)):
            if connections[i][0] > max_vertices:
                max_vertices = connections[i][0]
            if connections[i][1] > max_vertices:
                max_vertices = connections[i][1]

        # Make 3 arrays of the size of the max number of vertices for MaxIn, connections, and MaxOut
        self.vertices = [None] * (max_vertices + 2)
        self.vertices_maxIn = [None] * (max_vertices + 2)
        self.vertices_maxOut = [None] * (max_vertices + 2)

        # Make the values of the arrays vertices, vertices_maxIn, and vertices_maxOut
        for i in range(len(self.vertices)):
            self.vertices[i] = Vertex(i % (max_vertices + 2))
            self.vertices_maxIn[i] = Vertex(f'{i % (max_vertices + 2)} in')
            self.vertices_maxOut[i] = Vertex(f'{i % (max_vertices + 2)} out')

        self.add_edges(connections)

    def __str__(self) -> str:
        """
        Function Description: This function prints the graph

        Approach Description: We loop through the vertices and print them

        This takes O(V) time as we loop through the vertices
        This takes O(1) space as we are not creating any new variables

        :param None
        :return: None

        Time Complexity: O(V) where V is the number of vertices
        Space Complexity: O(1)
        """
        # Print all the vertices
        return_string = ""
        for vertex in self.vertices:
            return_string = return_string + "Vertex " + str(vertex) + "\n"

        for vertex in self.vertices_maxIn:
            return_string = return_string + "Vertex " + str(vertex) + "\n"

        for vertex in self.vertices_maxOut:
            return_string = return_string + "Vertex " + str(vertex) + "\n"
        return return_string

    def add_edges(self, argv_edges):
        """
        Function Description: This function adds all the edges to the graph and makes a super sink 

        Approach Description: The way I connect edges is that maxIn connects to the connection, then 
        connection connects to maxOut finally if we want to go to (0, 1, 3000) then I make connect my 
        0 maxOut to 1 maxIn then repeat the stated steps.

        This takes O(L + T + E) time as we loop all the verticies to add the current_to_input, current_to_output 
        etc. edges, then + O(T) as we make a super sink for all the targets. Finally + O(E) as we loop through all 
        the edges to make the connections 

        This takes O(1) space as we are not creating any new variables just reusing the already created one 

        :param argv_edges: list of edges
        :return: None

        Time Complexity: O(L + T + E) where L is lenght of vertices, T is the number of targets and E is the number of edges
        Space Complexity: O(1)
        """
        # Add all the edges to the graph
        for i in range(len(self.maxIn)):
            # I connect maxIn to the connection, then connection connects to maxOut, 
            current_to_input = Edge(self.vertices_maxIn[i], self.vertices[i], self.maxIn[i])
            # Every connection I make I also make a corresponding residual edge
            input_to_current = Edge(self.vertices[i], self.vertices_maxIn[i], 0)
            current_to_output = Edge(self.vertices[i], self.vertices_maxOut[i], self.maxOut[i])
            output_to_current = Edge(self.vertices_maxOut[i], self.vertices[i], 0)
            current_vetex = self.vertices[i]
            input_vertex = self.vertices_maxIn[i]
            current_vetex.add_edge(current_to_output, output_to_current)
            input_vertex.add_edge(current_to_input, input_to_current)
        
        # Make a super sink for all the targets 
        for i in range(len(self.targets)):
            current_edge = Edge(self.vertices_maxOut[self.targets[i]], self.vertices[-1], self.maxOut[self.targets[i]])
            residual_edge = Edge(self.vertices[-1], self.vertices_maxOut[self.targets[i]], 0)
            current_vertex = self.vertices_maxOut[self.targets[i]]
            current_vertex.add_edge(current_edge, residual_edge)

        # Lastly I make the connections by conection maxOut of the edge starting at to maxIn of the edge going to 
        for edge in argv_edges:
            u = edge[0]
            v = edge[1]
            w = edge[2]
            output_edge = Edge(self.vertices_maxOut[u], self.vertices_maxIn[v], w)
            residual_edge = Edge(self.vertices_maxIn[v], self.vertices_maxOut[u], 0)
            output_vertex = self.vertices_maxOut[u]
            output_vertex.add_edge(output_edge, residual_edge)

class Vertex:
    def __init__(self, id) -> None:
        """
        Function description: Initialises the vertex

        This take O(1) time as we just assign the values
        This take O(1) space as we only assigning values to variables
        :param: id: the id of the vertex
        :return: None

        Time complexity: complexity: O(1)
        Aux space complexity: O(1)
        """
        self.id = id
        self.edges = []
        self.discovered = False
        self.visited = False
        self.distance = 0
        self.previous = None

    def add_edge(self, edge, reverse_edge):
        """
        Function description: Adds an edge to the vertex and make another 
        feature reverse edge that give the 

        This take O(1) time as we just append the edge to the list and make the 
        corresponding reverse edge variable 
        This take O(1) space as we only append the edge to the list and update the list

        :param: edge: the edge to be added
        :return: None

        Time complexity: complexity: O(1)
        Aux space complexity: O(1)
        """
        # Add the edge to the list of edges
        self.edges.append(edge)
        edge.reverse_edge = reverse_edge

    def __str__(self) -> str:
        """
        Function description: Returns the string representation of the vertex

        This take O(1) time as we just return the string
        This take O(1) space as we only return a string

        :param: N/A
        :return: None

        Time complexity: complexity: O(1)
        Aux space complexity: O(1)
        """
        # Return the string representation of the vertex
        return_string = str(self.id)
        for edge in self.edges:
            return_string = return_string + "\n Edges: " + str(edge)
        return return_string

    def added_to_queue(self):
        """
        Function description: Sets the vertex to discovered

        This take O(1) time as we just assign the values
        This take O(1) space as we only assigning True to a variable
        :param: N/A 
        :return: None
		
		Time complexity: complexity: O(1)
		Aux space complexity: O(1)
        """
        self.discovered = True

    def removed_from_queue(self):
        """
        Function description: Sets the vertex to visited

        This take O(1) time as we just assign the values
        This take O(1) space as we only assigning True to a variable
        :param: N/A 
        :return: None
		
		Time complexity: complexity: O(1)
		Aux space complexity: O(1)
        """
        self.visited = True

class Edge:
    def __init__(self, u, v, w) -> None:
        """
	    Function description: Initialises the edge

        This take O(1) time as we just assign the values
        This take O(1) space as we only assigning the inputs
        :param: u, v, w
        :return: None
		
		Time complexity: complexity: O(1)
		Aux space complexity: O(1)
	    """
        self.u = u
        self.v = v
        self.w = w
        # This is the reverse edge that is used in the residual graph
        self.reverse_edge = None

    def __str__(self) -> str:
        """
        Function description: Returns the string representation of the edge

        This take O(1) time as we just return the string
        This take O(1) space as we only return the string
        
        :param: None
        :return: None

        Time complexity: complexity: O(1)
        Aux space complexity: O(1)
        """
        return_string = "(" + str(self.u.id) + "," + str(self.v.id) + "," + str(self.w) + ")"
        return return_string

class ResidualNetwork:
    def __init__(self, connections, maxIn, maxOut, origin, targets):
        """
        Function Description: This function makes graph and changes the id the for the source and the sink 
        
        Approach Description: Just make the graph and in order to know where to start and where to finish I need to make a source and a sink, 
        for source since it's an integer I can just use the origin as the source but for the sink 
        
        In graph since we determine the max number of vertices by looping through connections this takes O(C) time. The also make two arrays
        of the size it's length which take O(L) time. 

        In graph when we loop through the to make the 3 arrays for size L so we are taking O(L) space as we are making an array of the size lenght vertices
        
        :param connections: list of connections, maxIn: list of maxIn, maxOut: list of maxOut, origin: an integer respensenting staring point, 
        targets: the possble targets of the flow network
        :return: None 

        Time Complexity: O(L+C) where R is lenght of vertices and C is the length of connections
        Space Complexity: O(L) where L is lenght of vertices
        """
        # Make the graph and change the id for the source and the sink
        self.graph = Graph(connections, maxIn, maxOut, origin, targets)
        self.maxIn = maxIn
        self.maxOut = maxOut
        self.origin = origin
        self.targets = targets

        for i in range(len(self.graph.vertices)):
            if i == self.origin:
                self.graph.vertices[i].id = "s"

        for i in range(len(self.graph.vertices)):
            if i == len(self.graph.vertices) - 1:
                self.graph.vertices[i].id = "t"

    def reset(self):
        """
        Function description: resets the vertex for the bfs

        This take O(1) time as we just assign the values
        This take O(1) space as we only assigning values to variables
        :param: id: None
        :return: None

        Time complexity: complexity: O(1)
        Aux space complexity: O(1)
        """
        # Reset the vertex in connections for the bfs 
        for vertex in self.graph.vertices:
            vertex.discovered = False
            vertex.visited = False
            vertex.previous = None

        # Reset the vertex in MaxIn for the bfs 
        for vertex in self.graph.vertices_maxIn:
            vertex.discovered = False
            vertex.visited = False
            vertex.previous = None

        # Reset the vertex in MaxOut for the bfs 
        for vertex in self.graph.vertices_maxOut:
            vertex.discovered = False
            vertex.visited = False
            vertex.previous = None

    def get_Augmenting_Path(self):
        """
        Function Description: This function does the bfs and does gives the path of it 

        Approach Description: Uses the bfs to find the path from source to sink, then calculates all the edges in the path

        This takes O(|D| · |C|^2) time as the bfs takes O(|D| · |C|) time and the path takes O(|C|) time
        This takes O(|C|) space as the the path takes O(|C|) space

        :param: None
        :return: None

        Time Complexity: O(|D| · |C|^2) where |D| is the data centres and |C| is communication channels
        Space Complexity: O(|C|) where |C| is the communication channels
        """
        # This is the bfs to find the path from source to sink
        discovered = deque()
        sink = None
        path = []
        # Add the source to the queue
        discovered.append(self.graph.vertices[self.origin])
        # Loop through the queue unitl it's empty
        while len(discovered) > 0:
            # Pop the first element in the queue and mark as visited 
            u = discovered.popleft()
            u.visited = True
            # If we find t means we found the sink so we break out of the loop
            if u.id == "t":
                sink = u
                break
            # in the loop we loop through the edges and if the edge is not visited and the weight is greater than 0 we add it to the queue
            for edge in u.edges:
                v = edge.v
                if not v.discovered and not v.visited and edge.w > 0:
                    v.discovered = True
                    # Make the backtracking an edge so the residual graph can the reverse edge
                    v.previous = edge
                    discovered.append(v)
        # Here we calculate the path from source to sink
        if sink is not None:
            edge = sink.previous
            while edge is not None:
                path.append(edge)
                edge = edge.u.previous
            path.reverse()
        self.reset()
        return path

    def residual_capacity(self, path):
        """
        Function Description: This function calculates the residual capacity of the path

        Approach Description: Loops through the path and finds the smallest residual capacity
        This takes O(|C|) time as we loop through the path
        This takes O(1) space as we only assign the value to the variable

        :param: path: the path from source to sink
        :return: None
        """
        # Here I calculate the residual capacity suing the path
        residual_capacity = float("inf")
        # I save the smallest residual capacity
        for edge in path:
            if edge.w < residual_capacity:
                residual_capacity = edge.w
        return residual_capacity

    def augment_flow(self, path):
        """
        Function Description: This function augments the flow of the path

        Approach Description: Loops through the path and augments the flow of the path

        This takes O(|C|) time as we loop through the path
        This takes O(1) space as we only assign the value to the variable

        :param: path: the path from source to sink
        :return: None
        """
        # Here I augment the flow of the path
        residual_capacity = self.residual_capacity(path)
        # subtract the residual capacity from the edge and add it for the residual edge
        for edge in path:
            edge.w -= residual_capacity
            edge.reverse_edge.w += residual_capacity

def maxThroughput(connections, maxIn, maxOut, origin, targets):
    """
    Function Description: This function calculates the max throughput of the network

    Approach Description: Uses the residual network to calculate the max throughput

    This takes O(|D| · |C|^2) time as the bfs takes O(|D| · |C|) time and the path takes O(|C|) time
    This takes O(|C|) space as the the path takes O(|C|) space

    :param: connections: list of connections, maxIn: list of maxIn, maxOut: list of maxOut, origin: an integer respensenting staring point,
    targets: the possble targets of the flow network
    :return: max flow of the netwrk from source to sink
    """
    flow = 0
    # Make the residual network
    residual_Network = ResidualNetwork(connections, maxIn, maxOut, origin, targets)
    path = residual_Network.get_Augmenting_Path()
    # While there is a path we augment the flow and find the next path and when there is none just return the flow calculated till that point
    while path != []:
        residual_capacity = residual_Network.residual_capacity(path)
        # every time we augment the flow we the calculated residual capacity
        flow += residual_capacity
        residual_Network.augment_flow(path)
        path = residual_Network.get_Augmenting_Path()
    return flow

if __name__ == "__main__":
    # Example
    connections = [(0, 1, 3000), (1, 2, 2000), (1, 3, 1000),
    (0, 3, 2000), (3, 4, 2000), (3, 2, 1000)]
    maxIn = [5000, 3000, 3000, 3000, 2000]
    maxOut = [5000, 3000, 3000, 2500, 1500]
    origin = 0
    targets = [4, 2]
    print(maxThroughput(connections, maxIn, maxOut, origin, targets))
