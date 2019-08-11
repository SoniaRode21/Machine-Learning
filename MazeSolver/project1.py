__author__ = 'Soniya Rode'

from graph import Graph
import math
import random
import turtle
import sys

def file_len(fname):
    '''
    This function returns the height and width of the maze

    :param fname: File name
    :return: height, width of the maze
    '''
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
        width = len(l.replace(" ", ""))
        if (width % 2 != 0):
            width -= 1

    return i + 1, width


def getMatrix(fname, height, width):
    my_file = open(fname, "r")

    # Create a matrix to store the pond data
    Matrix = [[0 for x in range(width)] for y in range(height)]

    # i and j for iteration
    i = 0
    j = 0
    for line in my_file:

        # strip off the spaces
        line = line.strip()

        ## Remove all the spaces in python

        line = line.replace(" ", "")

        for _ in line:
            Matrix[i][j] = _
            j += 1
        j = 0
        i += 1
    return Matrix


def createGraph(Matrix, height, width):
    # Create an Graph object
    escapeRoutes = Graph()

    # For each node find all the reachable nodes
    for i in range(0, height):
        # listVertices = []

        for j in range(0, width):
            listVertices = []

            # Functions to get connected nodes
            horizontalUp(listVertices, Matrix, Matrix[i][j], i, j, height, width)
            horizontalDown(listVertices, Matrix, Matrix[i][j], i, j, height, width)
            verticalForward(listVertices, Matrix, Matrix[i][j], i, j, height, width)
            verticalBackward(listVertices, Matrix, Matrix[i][j], i, j, height, width)

            string = str(i) + "," + str(j)

            # Add the connected nodes to the graph
            for _ in listVertices:
                escapeRoutes.addEdge(string, _)
    return escapeRoutes
def iddfs(start,end):
    depth=250
    i=0

    while i<depth:
        visited = set()
        visited.add(start)

        path=__findPathDLS(start,end,visited,i)

        if path!=None:
            break
        else:
            i+=1

    return path


def __findPathDLS(current, end, visited,depth):
    """
    Private recursive helper function that finds the path, if one exists,
    from the current vertex to the end vertex
    :param current (Vertex): The current vertex in the traversal
    :param end (Vertex): The destination vertex
    :param visited (set of Vertex): the vertices already visited
    :return: A list of Vertex objects from start to end, if a path exists,
        otherwise None
    """

    # A successful base case is when we traverse to the end vertex.  In this
    # case, wrap it in a list and return it to the caller to construct the
    # full path
    if current == end:
        return [current.name]
    if depth==0:
        return None
    for neighbor in current.getConnections():

        if neighbor not in visited:
            visited.add(neighbor)

            path = __findPathDLS(neighbor, end, visited,depth-1)
            # If the path is not None, current is part of the solution path,
            # so add it to the front of the path list and return it
            if path != None:
                path.insert(0, current.name)
                return path
    # No path was found, so pass back None
    return None



def aStar(start, end, heuristicFunction):
    open_list = []
    closed_list = []

    # Add the start node
    open_list.append(start)

    # Loop until you find the end
    while len(open_list) > 0:
        # Get the current node
        current = open_list[0]
        current_index = 0
        for index, item in enumerate(open_list):
            if functionF(heuristicFunction, start, item, end) < functionF(heuristicFunction, start, current, end):
                current = item
                current_index = index
        # Pop current off open list, add to closed list
        for neighbours in open_list[current_index].getConnections():
            if neighbours not in open_list and neighbours not in closed_list:
                open_list.append(neighbours)
        open_list.pop(current_index)
        closed_list.append(current)

        # Found the goal
        if current == end:
            return closed_list
    if current == end:
        return None


def functionF(heuristicFunction, start, current, goal):
    current = current.name
    current = current.split(",")
    start = start.name
    start = start.split(",")
    goal = goal.name
    goal = goal.split(",")

    g = math.sqrt((int(current[0]) - int(start[0])) ** 2 + (int(current[1]) - int(start[1])) ** 2)
    h = heuristicFunction(current, goal)
    return g + h


# Manhattan Distance
def heuristicOne(current, goal):
    x1 = int(current[0])
    y1 = int(current[1])
    x2 = int(goal[0])
    y2 = int(goal[1])
    return (abs(x1 - x2) + abs(y1 - y2))



#Random Bad heuristic
def heuristicTwo(current, goal):
    return random.randint(0, 200)
'''
#Euclidean Distance
def heuristicThree(current, goal):
    x1 = int(current[0])
    y1 = int(current[1])
    x2 = int(goal[0])
    y2 = int(goal[1])
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
'''
#Squared Euclidean Distance
def heuristicThree(current, goal):
    x1 = int(current[0])
    y1 = int(current[1])
    x2 = int(goal[0])
    y2 = int(goal[1])
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2)





#Diagonal distance
def heuristicFour(current,goal):


    x1=int(current[0])
    y1=int(current[1])
    x2=int(goal[0])
    y2=int(goal[1])
    return max(abs(x1 - x2), abs(y1 - y2))


def main():
    '''
        The main function reads the graph file and prints out how quickly you can exit from each node
        and whether an exit is possible or not
    '''
    if(len(sys.argv)<2):
        filename="maze.txt"

    else:
        filename=sys.argv[1]
    height, width = file_len(filename)

    exit = str(height - 1) + "," + str(width - 1)

    Matrix = getMatrix(filename, height, width)
    escapeRoutes = createGraph(Matrix, height, width)

    # Exit node

    startVertex = escapeRoutes.getVertex("0,0")
    # Get nodes connected to the exit node
    endVertex = escapeRoutes.getVertex(exit)
    maxCount = 0

    path = (aStar(startVertex, endVertex, heuristicOne))
    if path != None:
        print("_____Path Found_____")
        print("Path by heuristic function: Manhattan Distance :")
        answer = ""
        for _ in path:
            answer += " -> " + _.name
        print(answer)
        print("Number of steps required : ", len(path))
        h1=len(path)



        path = (aStar(startVertex, endVertex, heuristicFour))
        print("Path by heuristic function : Diagonal Distance :")
        answer = ""
        for _ in path:
            answer += " -> " + _.name
        print(answer)
        print("Number of steps required : ", len(path))
        h4 = len(path)

        path = (aStar(startVertex, endVertex, heuristicThree))
        print("Path by heuristic function : Euclidean Distance:")
        answer = ""
        for _ in path:
            answer += " -> " + _.name
        print(answer)
        print("Number of steps required : ", len(path))
        h3 = len(path)
        path = (aStar(startVertex, endVertex, heuristicTwo))
        print("Path by heuristic function :  Random Heuristic:")
        answer = ""
        for _ in path:
            answer += " -> " + _.name
        print(answer)
        print("Number of steps required : ", len(path))
        h2 = len(path)

        print("Using IDS")
        path = iddfs(startVertex, endVertex)


        answer = ""
        for _ in path:
            answer += " -> " + _
        print(answer)
        print("Number of steps required : ", len(path))
        ids=len(path)


'''
        drawComparasion(h1,h2,h3,h4,ids)

        drawPath(path)

def drawComparasion(h1,h2,h3,h4,ids):
    pass

def drawPath(_path):
    path=[]
    for _ in _path:

        _=_.split(",")
        path.append((_))
    for node in path:
        turtle.goto(int(node[0])+100,int(node[1])+100)
        print("hey")

    turtle.exitonclick()



'''

def horizontalUp(list, matrix, node, i, j, height, width):
    '''
    This function finds the reachable node above the current node
    :param list: list of nodes connected to the current node
    :param matrix: Matrix containing the graph
    :param node: Current node
    :param i: i th position of the node
    :param j: j th position of the node
    :param height: Height of the pond
    :param width: Widht of the pond
    :return: returns the updated list
    '''

    # If the current node is a stone
    if node == "1":
        return list
    # For the first row
    if i == 0:

        return list
    # For all other nodes
    else:
        if matrix[i - 1][j] == "0":
            string = str(i - 1) + "," + str(j)
            list.append(string)

    return list


def horizontalDown(list, matrix, node, i, j, height, width):
    '''
    This function finds the reachable node below the current node
    :param list: list of nodes connected to the current node
    :param matrix: Matrix containing the graph
    :param node: Current node
    :param i: i th position of the node
    :param j: j th position of the node
    :param height: Height of the pond
    :param width: Widht of the pond
    :return: returns the updated list
    '''

    # If the current node is a stone
    if node == "1":
        return list

    # For the last row nodes
    if i >= height - 1:

        return list

    # For all other nodes
    else:

        if matrix[i + 1][j] == "0":
            string = str(i + 1) + "," + str(j)
            list.append(string)

    return list


def verticalForward(list, matrix, node, i, j, height, width):
    '''
    This function finds the reachable node to  the right of the  current node
    :param list: list of nodes connected to the current node
    :param matrix: Matrix containing the graph
    :param node: Current node
    :param i: i th position of the node
    :param j: j th position of the node
    :param height: Height of the pond
    :param width: Widht of the pond
    :return: returns the updated list
    '''
    # If the current node is a stone
    if node == "1":
        return list

    # For the last row nodes
    if j == width - 1:
        return list

    # For all other nodes
    else:

        if matrix[i][j + 1] == "0":
            string = str(i) + "," + str(j + 1)
            list.append(string)

    return list


def verticalBackward(list, matrix, node, i, j, height, width):
    '''
    This function finds the reachable to the left the current node
    :param list: list of nodes connected to the current node
    :param matrix: Matrix containing the graph
    :param node: Current node
    :param i: i th position of the node
    :param j: j th position of the node
    :param height: Height of the pond
    :param width: Widht of the pond
    :return: returns the updated list
    '''

    # If the current node is a stone
    if node == "1":
        return list

    # For the last row nodes
    if j == 0:

        return list

    # For all other nodes
    else:

        if matrix[i][j - 1] == "0":
            string = str(i) + "," + str(j - 1)
            list.append(string)

    return list


main()
