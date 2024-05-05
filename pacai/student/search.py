"""
In this file, you will implement generic search algorithms which are called by Pacman agents.
"""

from pacai.util.stack import Stack
from pacai.util.queue import Queue
from pacai.util.priorityQueue import PriorityQueue

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first [p 85].

    Your search algorithm needs to return a list of actions that reaches the goal.
    Make sure to implement a graph search algorithm [Fig. 3.7].

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:
    ```
    print("Start: %s" % (str(problem.startingState())))
    print("Is the start a goal?: %s" % (problem.isGoal(problem.startingState())))
    print("Start's successors: %s" % (problem.successorStates(problem.startingState())))
    ```
    """

    # DFS is LIFO so set it to a stack
    frontier = Stack()
    
    visited = set()
    
    frontier.push((problem.startingState(), []))

    # Loop until there are no more logical adjacent squares to explore
    while not frontier.isEmpty():
        current, path = frontier.pop()

        # Update visited states so that there is not a repeat check
        visited.add(current)

        # No need to continue if the goal has been reached
        if problem.isGoal(current):
            print(f'The final path to the food was: {path}')
            return path

        # As long as the goal hasn't been reached, move to the next state
        for successor, action, _ in problem.successorStates(current):
            if successor not in visited:
                # Go to the next state by putting its successor in the stack, update path
                frontier.push((successor, path + [action]))

    raise NotImplementedError("No solution found.")

def breadthFirstSearch(problem):
    """
    Search the shallowest nodes in the search tree first. [p 81]
    """
    # BFS is FIFO so set it to a queue
    frontier = Queue()
    visited = set()
    
    # Initialize queue at initial state and append path along the way, begins empty
    frontier.push((problem.startingState(), []))

    # Loop until there are no more logical adjacent squares to explore
    while not frontier.isEmpty():
        current, path = frontier.pop()
    
        # Update visited states so that there is not a repeat check
        visited.add(current)
    
        # No need to continue if the goal has been reached
        if problem.isGoal(current):
           # print(f'The final path to the food was: {path}')
            return path
        
        # As long as the goal hasn't been reached, move to the next state
        for successor, action, _ in problem.successorStates(current):
            if successor not in visited:
                # Go to the next state by putting its successor in the queue, update path
                frontier.push((successor, path + [action]))

    raise NotImplementedError("No solution found.")

def uniformCostSearch(problem):
    """
    Search the node of least total cost first.
    """

    # Uniform cost search uses the weights in the graph so it is a priority queue
    frontier = PriorityQueue()
    
    # Need to track lowest cost to each state visited
    visited = {}
    frontier.push((problem.startingState(), []), 0)

    # Loop until there are no more logical adjacent squares to explore
    while not frontier.isEmpty():
        current, path = frontier.pop()

        # No need to continue if the goal has been reached
        if problem.isGoal(current):
            print(f'The final path to the food was: {path}')
            return path

        # Calculate the cost to the next state in order to determine the successor
        current_cost = problem.actionsCost(path)
        
        visited[current] = current_cost
    
        # As long as the goal hasn't been reached, move to the next state
        for successor, action, _ in problem.successorStates(current):
    
            # Update the path and calculate the total cost
            this_path = path + [action]
            this_cost = problem.actionsCost(this_path)
            
            # Check whether to continue on the path or go down a more cost effective route
            if successor not in visited or this_cost < visited[successor]:
                
                # Go to the next state by putting its successor in the queue, update path
                frontier.push((successor, this_path), this_cost)

    return []

def aStarSearch(problem, heuristic):
    """
    Search the node that has the lowest combined cost and heuristic first.
    """
    
    # A* search uses the heuristic in the graph so it is a priority queue
    frontier = PriorityQueue()
    check_states = set()

    # Account for the distance from the start and the goal state together
    frontier.push((problem.startingState(), [], 0), heuristic(problem.startingState(), problem))

    # Loop until there are no more logical adjacent squares to explore
    while not frontier.isEmpty():
        current, path, cost = frontier.pop()

        # No need to continue if the goal has been reached
        if problem.isGoal(current):
            print(f'The final path to win was: {path}')
            return path
        
        check_states.add(current)

        # As long as the goal hasn't been reached, move to the next state
        for successor, action, step_cost in problem.successorStates(current):
    
            # Calculate the cost to move to the next state 
            # + the distance from goal state and minimize the sum
            successor_cost = cost + step_cost
            successor_eval = successor_cost + heuristic(successor, problem)

            if not any(successor == check_state for check_state in check_states):
    
                # Go to the next state by putting its successor in the queue, update path
                frontier.push((successor, path + [action], successor_cost), successor_eval)
    return []