from __future__ import annotations
""" Basic class implementation of an array of references for FIT units"""

from ctypes import py_object
from typing import TypeVar, Generic

T = TypeVar('T')

class ArrayR(Generic[T]):
    def __init__(self, length: int) -> None:
        """
		Function description: Creates an array of references to objects of the given length

		Approach description: We want to initilise a new list with a specific lenght. IF the lenght 0 or less then we raise a
        value error otherwise we lnitalise self.array using py_object. Each element in the array is then set to None and a shallow
        copy of self.array is made.    
		
		Making this costs O(l) time, as we are setting each element to None.

        When we make a new list is also costs O(l) space as we are making a new list with l None elements in it.

        :param length: the length of the array we want to create
        :return: None

        Time complexity: complexity: O(l), where l is the length of the array
		Aux space complexity: O(1), where l is the length of the array
	    """
        #check if length is valid
        if length <= 0:
            raise ValueError("Array length should be larger than 0.")
        #initialise the array and set each element to None
        self.array = (length * py_object)() # initialises the space
        self.array[:] =  [None for _ in range(length)]

    def __len__(self) -> int:
        """
	    Function description: Returns the length of the array
		
		Output: length of the array

        This take O(1) time as we are just returning the length of the array
        This take O(1) space as we are not creating any new variables

        :param: None
        :return: None
		
		Time complexity: complexity: O(1)
		Aux space complexity: O(1)
	    """
        return len(self.array)

    def __getitem__(self, index: int) -> T:
        """
	    Function description: index in between 0 and length - self.array[] checks it,
        Returns the object in position index
		
		Output: object in position index

        This take O(1) time as adressing an element in an array is done in contsant time
        This take O(1) space as we are not creating any new variables
        
        :param index: the index of the item we want to return
        :return: object in position index
		
		Time complexity: complexity: O(1)
		Aux space complexity: O(1)
	    """
        return self.array[index]

    def __setitem__(self, index: int, value: T) -> None:
        """
	    Function description: index in between 0 and length - self.array[] checks it,
        Sets the object in position index to value
		
		Output: None

        This take O(1) time as adressing an element in an array is done in contsant time
        This take O(1) space as we modify value already stored not create any new variables

        :param index, value: index the position we to sent the value to, value the value we want to set the index to
        :return: None
		
		Time complexity: complexity: O(1)
		Aux space complexity: O(1)
	    """
        self.array[index] = value


from typing import Generic

class MinHeap(Generic[T]):
    """Min Heap implemented using an array"""
    MIN_CAPACITY = 1

    def __init__(self, max_size: int) -> None:
        """
	    Function description: sets the initial lenght to 0 and creates an array of size max_size + 1

        This take O(1) time as we just set the length and array in constant time
        This take O(a) space as we are creating a new array of size max_size + 1

        :param max_size: max_size of the array to be created
        :return: None
		
		Time complexity: complexity: O(1)
		Aux space complexity: O(a), where is the a is an array of size max_size + 1
	    """
        self.length = 0
        self.the_array = ArrayR(max(self.MIN_CAPACITY, max_size) + 1)

    def __len__(self) -> int:
        """
	    Function description: Returns the length of the MinHeap
		
		Output: length of the MinHeap

        This take O(1) time as we are just returning the length of the MinHeap
        This take O(1) space as we are not creating any new variables
		
		Time complexity: complexity: O(1)
		Aux space complexity: O(1)
	    """
        return self.length

    def is_full(self) -> bool:
        """
        Function description: Returns boolean if the MinHeap is full or not
        
        Output: True or False depending on if the MinHeap is full or not

        This take O(1) time as we just do simple maths
        This take O(1) space as we are not creating any new variables
        
        Time complexity: complexity: O(1)
        Aux space complexity: O(1)
        """
        return self.length + 1 == len(self.the_array)

    def rise(self, k: int) -> None:
        """
		Function description: 1 <= k <= self.length,  Rise element at index k to its correct position

		Approach description: We want to make sure that the heap is maintained after adding a new element.
        We achieve this by comparing the new element with its parent and swapping them if the parent is larger.
        We continue this process until the parent is smaller than the new element or we reach the root of the heap.
	    
        This take O(log k) time as we are comparing the new element with its parent and swapping them if the parent is larger.
        This take O(1) space as it only modifies we are not creating any new variables
        
        :param k: number of elements in the heap
        :return: None

        Time complexity: complexity: O(log k), where k is the position of the new element
        Aux space complexity: O(1)
        """
        item = self.the_array[k]
        # compare the new element with its parent and swapping them if the parent is larger
        while k > 1 and item < self.the_array[k // 2]:
            #swap the elem
            self.the_array[k] = self.the_array[k // 2]
            # update the heap index and resize k 
            self.the_array[k].heap_index = k
            k = k // 2
        # set the item to the new position
        self.the_array[k] = item
        self.the_array[k].heap_index = k

    def add(self, element: T) -> None:
        """
        Function description: Adds element to the heap

        Approach description: IF the heap is full we raise an IndexError, else we add the element to the heap and rise it to its correct position

        This take O(1) time as we do simple maths which is in constant time
        This take O(1) space as it only modifies we are not creating any new variables

        :param element: the element to be added to the heap
        :return: None

        Time complexity: complexity: O(1)
        Aux space complexity: O(1)
        """
        # raise an IndexError if the heap is full
        if self.is_full():
            raise IndexError
        # add the element to the heap and rise it to its correct position
        self.length += 1
        self.the_array[self.length] = element
        element.heap_index = self.length
        self.rise(self.length)
    
    def remove(self, element: T) -> None:
        """
        Function description: Remove the element from the heap

        Approach description: If the element is not in the heap we raise a ValueError, else we remove the element from the heap and sink it to its correct position

        This take O(l) as we loop through the heap to find the element 
        This take O(1) space as it only modifies we are not creating any new variables

        :param element: the element to be removed to the heap
        :return: None

        Time complexity: complexity: O(l), where l is the number of elements in the heap
        Aux space complexity: O(1)
        """
        index = -1
        # find the index of the element
        for i in range(1, self.length+1):
            if self.the_array[i] == element:
                index = i
                break
        # raise a ValueError if the element is not in the heap
        if index == -1:
            raise ValueError("Element not found in heap")
        # remove the element from the heap and sink it to its correct position
        self.the_array[index] = self.the_array[self.length]
        self.length -= 1
        self.sink(index)


    def smallest_child(self, k: int) -> int:
        """
        Function description: 1 <= k <= self.length // 2, Returns the index of k's child with smallest value.

        Approach description: finds the index of a child with the smallest value, if the left child is smaller than the 
        right child we return the index of the left child else we return the index of the right child

        This take O(1) time as we only do simple maths  
        This take O(1) space as it only modifies we are not creating any new variables

        :param element: the element to be removed to the heap
        :return: None

        Time complexity: complexity: O(1)
        Aux space complexity: O(1)
        """
        # finds the index of a child with the smallest value
        if 2 * k == self.length or \
                self.the_array[2 * k] < self.the_array[2 * k + 1]:
            return 2 * k
        else:
            return 2 * k + 1

    def sink(self, k: int) -> None:
        """
        Function description: 1 <= k <= self.length Make the element at index k sink to the correct position.

        Approach description: move the element at index k down the heap by  swapping it with its smallest child
        until it is in the correct position also update the heap index of the element

        This take O(1) time as we only do simple maths  
        This take O(1) space as it only modifies we are not creating any new variables

        :param k: the index of the element to be sunk
        :return: None

        Time complexity: complexity: O(1)
        Aux space complexity: O(1)
        """
        item = self.the_array[k]
        # move the element at index k down the heap by swapping it with its smallest child
        while 2 * k <= self.length:
            min_child = self.smallest_child(k)
            # if the element at index k is smaller than its smallest child we break 
            if self.the_array[min_child] > item:
                break
            # swap the element at index k with its smallest child
            self.the_array[k] = self.the_array[min_child]
            self.the_array[k].heap_index = k
            k = min_child
        # set the item to the new position
        self.the_array[k] = item
        self.the_array[k].heap_index = k

    def get_min(self) -> T:
        """
        Function description: Remove (and return) the maximum element from the heap.

        Approach description: If the heap is not empty we find the minimum element 

        This take O(1) time as we only do simple maths  
        This take O(1) space as it only modifies we are not creating any new variables

        :param k: the index of the element to be sunk
        :return: None

        Time complexity: complexity: O(1)
        Aux space complexity: O(1)
        """
        # raise an IndexError if the heap is empty
        if self.length == 0:
            raise IndexError
        # find the minimum element
        min_elt = self.the_array[1]
        # remove the minimum element from the heap and sink it to its correct position
        self.length -= 1
        if self.length > 0:
            self.the_array[1] = self.the_array[self.length+1]
            self.sink(1)
        return min_elt

class Graph:
    def __init__(self, passenger, roads) -> None:
        """
        Function Description: This function makes a layred graph with two layers one for the solo and one for the passenger
        
        Approach Description: We make a graph with two layers one for the solo and one for the passenger. We then add the
        edges between the layers and the edges within the layers.

        since we determine the max number of vertices by looping through roads this takes O(Road) time. The also make two arrays
        of the size it's length which take O(R) time. 

        When we loop through the to make the array we are taking O(L) space as we are making an array of the size lenght vertices_layer1  
        
        :param passenger, roads: list of passengers and list of roads
        :return: None 

        Time Complexity: O(L+R) where R is lenght of vertices_layer1 and Road is the length of roads
        Space Complexity: O(L) where L is lenght of vertices_layer1 
        """
        self.passenger = passenger
        self.roads = roads
        max_vertices = 0
        # using roads find the maximum number of vertices
        for i in range(len(roads)):
            # See all the first index of the roads
            if roads[i][0] > max_vertices:
                max_vertices = roads[i][0]
            # See all the second index of the roads and set max to the biggest
            if roads[i][1] > max_vertices:
                max_vertices = roads[i][1]

        # create two arrays of vertices suing the max number of vertices
        self.vertices_layer1 = [None] * (max_vertices + 1)
        self.vertices_layer2 = [None] * (max_vertices + 1)

        # create vertices 
        for i in range(len(self.vertices_layer1)):
            self.vertices_layer1[i] = Vertex(i % (max_vertices +1))
            self.vertices_layer2[i] = Vertex(i % (max_vertices +1))

        # add edges from the roads to the vertices
        self.add_edges(roads)

        # combine the graph to make it 2 layer graph 
        self.vertices = self.vertices_layer1 + self.vertices_layer2
          
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
        return_string = ""
        for vertex in self.vertices_layer1 or self.vertices_layer2:
            return_string = return_string + "Vertex " + str(vertex) + "\n"
    
    def add_edges(self, argv_edges):
        """
        Function Description: This function adds edges to the graph

        Approach Description: We loop through the edges and add them to the graph

        This takes O(E) time as we loop through the edges

        :param argv_edges: list of edges
        :return: None

        Time Complexity: O(E) where E is the number of edges
        Space Complexity: O(1)
        """
        for edge in argv_edges:
            u = edge[0]
            v = edge[1]
            w1 = edge[2]
            w2 = edge[3]
            # For layer 1 add the non carpool time edges 
            current_edge_layer1 =Edge(self.vertices_layer1[u], self.vertices_layer1[v], w1)
            current_vetex = self.vertices_layer1[u]
            current_vetex.add_edge(current_edge_layer1)
            # For layer 2 add the carpool time edges
            current_edge_layer2 = Edge(self.vertices_layer2[u], self.vertices_layer2[v], w2)
            current_vetex = self.vertices_layer2[u]
            current_vetex.add_edge(current_edge_layer2)
        
        # if there is a passenger add an edge connecting the two graph 
        for i in self.passenger:
            edge = Edge(self.vertices_layer1[i], self.vertices_layer2[i], 0)
            self.vertices_layer1[i].add_edge(edge)
            
    def dijkstra(self, source, destination):
        """
        Function Description: This function finds the shortest path from the source to the destination

        Approach Description: We use dijkstra's algorithm to find the shortest path from the source to the destination

        This takes O(RlogL) time as we loop through the edges and vertices and use the priority queue

        This takes O(L) space as we are creating a list of the path

        :param source, destination: the source and destination of the path
        :return: the path from the source to the destination

        Time Complexity: O(RlogL) where R is the roads and L is the key locations
        Space Complexity: O(R) which the route of path we are taking 
        """
        # the inital distance is 0 
        self.vertices[source].distance = 0
        # use the priority queue min heap to store the vertices
        discovered = MinHeap(len(self.vertices))
        discovered.add(self.vertices[source])
        
        # while the queue is not empty
        while len(discovered) > 0:
            # get the minimum element from the queue and set it to visited
            u = discovered.get_min()
            u.removed_from_queue()

            # if the vertex is the destination return the path
            if u.id == destination:
                path = []
                current = u
                # add the path to the list
                while current is not None:
                    path.append(current.id)
                    current = current.previous
                path.reverse()
                return path

            # Perform edge relaxation on all adjacent edges
            for edge in u.edges:
                v = edge.v
                # when vertex is not visited or discovered 
                if not v.discovered and not v.visited:
                    v.discovered = True
                    v.distance = u.distance + edge.w
                    # consider the special case when the edge is 0 to connect the graphs 
                    if edge.w != 0:    
                        v.previous = u
                    else:
                        v.previous = u.previous
                    # add the vertex to the queue
                    discovered.add(v)
                #update distance if needed
                elif v.distance > u.distance + edge.w:
                    v.distance = u.distance + edge.w
                     # consider the special case when the edge is 0 to connect the graphs 
                    if edge.w != 0:    
                        v.previous = u
                    else:
                        v.previous = u.previous
                    # update vertex in heap
                    discovered.add(v)
        raise ValueError("Path not found")

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
        self.heap_index = None
    
    def add_edge(self, edge):
        """
        Function description: Adds an edge to the vertex

        This take O(1) time as we just append the edge to the list
        This take O(1) space as we only append the edge to the list

        :param: edge: the edge to be added
        :return: None

        Time complexity: complexity: O(1)
        Aux space complexity: O(1)
        """
        self.edges.append(edge)

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
        return_string = str(self.id)
        for edge in self.edges:
            return_string = return_string + "\n Edges: "  + str(edge)
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

    def __lt__(self, other):
        """
        Function description: Compares the distance of two vertex objects

        This take O(1) time as we only do simple maths
        This take O(1) space as we only do a comparision and not create any new variables

        :param other: the other vertex object to be compared to
        :return: True if the distance of the current vertex is less than the other vertex, False otherwise

        Time complexity: complexity: O(1)
        Aux space complexity: O(1)
        Citations: https://www.geeksforgeeks.org/dijkstras-shortest-path-algorithm-using-priority_queue-stl/,
        https://www.geeksforgeeks.org/python-__lt__-magic-method/
        """
        return self.distance < other.distance
    
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
    
def optimalRoute(start, end, passengers, roads):
    """
    Function Description: This function returns the optimal route from the start to the destination with the least 
    time taken. This function uses Dijkstra's algorithm to find the shortest path from the start to the destination.

    Approach Description: We make a layered graph with two layers with the first layer being the route with normal time 
    and the second graph being carpool time. We have to seitch to the second graph when we have more than 1 passenger.
    We do that by adding an edge of weight 0 between the two graphs. We then use Dijkstra's algorithm to find the
    shortest path from the start to the destination. We then return the path taken to reach the destination in the least 
    time. 

    We use the init which has a time Time Complexity: O(R+Road) where R is lenght of vertices_layer1 and Road is the length of roads
    and we use also use dijkstra's algorithm which has a Time Complexity of O(RlogL) where R is the roads and L is the key locations
    O(L+R) + O(RlogL) = O(RlogL) 
        
    Space Complexity: We use init which has O(L) where L is lenght of vertices_layer1 and we also use dijkstra's algorithm which has a
    Space Complexity of (R) which the route of path we are taking. Thus our total space complexity is O(L) + O(R) = O(L+R) 
    
    :param start, end, passengers, 
    :return: Array containing the path taken to reach the destination in the least time

    Time Complexity: O(RlogL) where R is the roads and L is the key locations  
    Space Complexity: O(L+R)
    """
    g = Graph(passengers, roads)
    return g.dijkstra(start, end)

if __name__ == "__main__":
    # Example
    start = 0
    end = 4
    # The locations where there are potential passengers
    passengers = [2, 1]
    # The roads represented as a list of tuple
    roads = [(0, 3, 5, 3), (3, 4, 35, 15), (3, 2, 2, 2), (4, 0, 15, 10),
    (2, 4, 30, 25), (2, 0, 2, 2), (0, 1, 10, 10), (1, 4, 30, 20)]
    # Your function should return the optimal route (which takes 27 minutes).
    print(optimalRoute(start, end, passengers, roads))
