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

    frontier = Stack()
    visited_states = set()
    frontier.push((problem.startingState(), []))

    while not frontier.isEmpty():
        current_state, path = frontier.pop()

        # if current_state in visited_states:
        #     continue  # Skip already visited states

        visited_states.add(current_state)

        if problem.isGoal(current_state):
            print(f'The final path to the food was: {path}')
            return path

        for successor, action, _ in problem.successorStates(current_state):
            if successor not in visited_states:
                frontier.push((successor, path + [action]))

    raise NotImplementedError("No solution found.")




def breadthFirstSearch(problem):
    
    frontier = Queue()
    visited_states = set()
    frontier.push((problem.startingState(), []))

    while not frontier.isEmpty():
        current_state, path = frontier.pop()

        # if current_state in visited_states:
        #     continue  # Skip already visited states

        visited_states.add(current_state)

        if problem.isGoal(current_state):
            print(f'The final path to the food was: {path}')
            return path

        for successor, action, _ in problem.successorStates(current_state):
            if successor not in visited_states:
                frontier.push((successor, path + [action]))

    raise NotImplementedError("No solution found.")

def uniformCostSearch(problem):
    """
    Search the node of least total cost first.
    """

    frontier = PriorityQueue()
    visited_states = {}
    frontier.push((problem.startingState(), []), 0)

    while not frontier.isEmpty():
        current_state, path = frontier.pop()

        # if current_state in visited_states:
        #     continue  # Skip already visited states

        if problem.isGoal(current_state):
            print(f'The final path to the food was: {path}')
            return path

        current_cost = problem.actionsCost(path)
        
        visited_states[current_state] = current_cost
        
        for successor, action, _ in problem.successorStates(current_state):
            this_path = path +[action]
            this_cost = problem.actionsCost(this_path)
            if successor not in visited_states or this_cost < visited_states[successor]:
                frontier.push((successor, this_path), this_cost)

    return []

def aStarSearch(problem, heuristic):
    """
    Search the node that has the lowest combined cost and heuristic first.
    """

    frontier = PriorityQueue()
    check_states = set()

    frontier.push((problem.startingState(), [], 0), heuristic(problem.startingState(), problem))

    while not frontier.isEmpty():
        current_state, path, cost = frontier.pop()

        if problem.isGoal(current_state):
            print(f'The final path to win was: {path}')
            return path
        
        check_states.add(current_state)

        for successor, action, step_cost in problem.successorStates(current_state):
            successor_cost = cost + step_cost
            successor_eval = successor_cost + heuristic(successor, problem)

            if not any(successor == check_state for check_state in check_states):
                frontier.push((successor, path + [action], successor_cost), successor_eval)
    return []


