import random

from pacai.agents.base import BaseAgent
from pacai.agents.search.multiagent import MultiAgentSearchAgent
from pacai.core import distance
from pacai.core.directions import Directions

class ReflexAgent(BaseAgent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.
    You are welcome to change it in any way you see fit,
    so long as you don't touch the method headers.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        `ReflexAgent.getAction` chooses among the best options according to the evaluation function.

        Just like in the previous project, this method takes a
        `pacai.core.gamestate.AbstractGameState` and returns some value from
        `pacai.core.directions.Directions`.
        """

        # Collect legal moves.
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions.
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best.

        return legalMoves[chosenIndex]
    
    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current `pacai.bin.pacman.PacmanGameState`
        and an action, and returns a number, where higher numbers are better.
        Make sure to understand the range of different values before you combine them
        in your evaluation function.
        """

        # Useful information you can extract.
        # newPosition = successorGameState.getPacmanPosition()
        # oldFood = currentGameState.getFood()
        # newGhostStates = successorGameState.getGhostStates()
        # newScaredTimes = [ghostState.getScaredTimer() for ghostState in newGhostStates]

        # *** Your Code Here ***

        successorGameState = currentGameState.generatePacmanSuccessor(action)

        newFood = successorGameState.getFood().asList()
        oldFood = currentGameState.getFood().asList()
        newGhostStates = successorGameState.getGhostStates()

        totalEval = 0

        # Get all weights for different functions
        food_weight = 25
        ghost_penalty = -500
        repeat_penalty = -30

        # Check if all food is eaten
        if not newFood:
            return 0

        # Boost score if action is food
        if len(newFood) < len(oldFood):
            totalEval += food_weight

        pacman_pos = currentGameState.getPacmanPosition()
        pacman_next_pos = successorGameState.getPacmanPosition()
        
        # Lower score if ghosts are close
        for ghostState in newGhostStates:
            distance_to_ghost = distance.manhattan(pacman_next_pos, ghostState.getPosition())
            if distance_to_ghost == 0:
                return 0
            if ghostState.getScaredTimer() <= 1 and distance_to_ghost < 5:
                totalEval += ghost_penalty / distance_to_ghost
                
                # Add to score if ghosts are scared and close
            if ghostState.getScaredTimer() > distance_to_ghost:
                totalEval -= ghost_penalty / distance_to_ghost
            
        min_dist = min(distance.manhattan(pacman_pos, food) for food in newFood)
        next_min_dist = min(distance.manhattan(pacman_next_pos, food) for food in newFood)

        # Add to score if the action gets you closer to the next food
        if next_min_dist < min_dist:
            totalEval += (4 * food_weight) / next_min_dist
        
        # Lower score if pacman stays still
        if (currentGameState.getPacmanPosition() == successorGameState.getPacmanPosition()):
            totalEval += repeat_penalty
            
        return totalEval
    
class MinimaxAgent(MultiAgentSearchAgent):
    """
    A minimax agent.

    Here are some method calls that might be useful when implementing minimax.

    `pacai.core.gamestate.AbstractGameState.getNumAgents()`:
    Get the total number of agents in the game

    `pacai.core.gamestate.AbstractGameState.getLegalActions`:
    Returns a list of legal actions for an agent.
    Pacman is always at index 0, and ghosts are >= 1.

    `pacai.core.gamestate.AbstractGameState.generateSuccessor`:
    Get the successor game state after an agent takes an action.

    `pacai.core.directions.Directions.STOP`:
    The stop direction, which is always legal, but you may not want to include in your search.

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Returns the minimax action from the current gameState using
    `pacai.agents.search.multiagent.MultiAgentSearchAgent.getTreeDepth`
    and `pacai.agents.search.multiagent.MultiAgentSearchAgent.getEvaluationFunction`.
    """
    
    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)
        
    def getAction(self, state):
        """
        Returns the minimax action from the current gameState.
        """
        legalActions = state.getLegalActions(0)
        bestAction = Directions.STOP
        bestScore = float("-inf")

        # Check all legal actions in algorithm and compare
        for action in legalActions:
            if action != Directions.STOP:
                successor = state.generateSuccessor(0, action)
                score = self.minValue(successor, 1)

                if score > bestScore:
                    bestScore = score
                    bestAction = action

        return bestAction

    def maxValue(self, state, depth):
        """
        Pac mans turn 
        """
        
        # exit the recursion when hitting a base case
        if depth == self.getTreeDepth() or state.isWin() or state.isLose():
            return self.getEvaluationFunction()(state)
        
        value = float("-inf")
        legalActions = state.getLegalActions(0)

        # check every pacman legal action and choose 
        # the best one using a given evaluation function
        for action in legalActions:
            if action != Directions.STOP:
                successor = state.generateSuccessor(0, action)
                value = max(value, self.minValue(successor, depth + 1))

        return value

    def minValue(self, state, depth):
        """
        Ghost turn
        """
        # exit the recursion when hitting a base case
        if depth == self.getTreeDepth() or state.isWin() or state.isLose():
            return self.getEvaluationFunction()(state)

        value = float("inf")
        numAgents = state.getNumAgents()

        # loop over every ghost agent 
        for agentIndex in range(1, numAgents):
            legalActions = state.getLegalActions(agentIndex)
            
            # pick the best action for each ghost to take
            for action in legalActions:
                if action != Directions.STOP:
                    successor = state.generateSuccessor(agentIndex, action)
                    value = min(value, self.maxValue(successor, depth))

        return value

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    A minimax agent with alpha-beta pruning.

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Returns the minimax action from the current gameState using
    `pacai.agents.search.multiagent.MultiAgentSearchAgent.getTreeDepth`
    and `pacai.agents.search.multiagent.MultiAgentSearchAgent.getEvaluationFunction`.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    def getAction(self, state):
        """
        Returns the minimax action from the current gameState.
        """
        legalActions = state.getLegalActions(0)
        bestAction = Directions.STOP
        bestScore = float("-inf")
        alpha = float("-inf")
        beta = float("inf")

        # Check all legal actions in algorithm and compare
        for action in legalActions:
            if action != Directions.STOP:
                successor = state.generateSuccessor(0, action)
                score = self.minValue(successor, 1, alpha, beta)

                if score > bestScore:
                    bestScore = score
                    bestAction = action

                alpha = max(alpha, bestScore)

        return bestAction

    def maxValue(self, state, depth, alpha, beta):
        """
        Pacmans turn
        """
        
        # exit the recursion when hitting a base case
        if depth == self.getTreeDepth() or state.isWin() or state.isLose():
            return self.getEvaluationFunction()(state)

        value = float("-inf")
        legalActions = state.getLegalActions(0)

        # check every pacman legal action and choose 
        # the best one using a given evaluation function
        for action in legalActions:
            if action != Directions.STOP:
                successor = state.generateSuccessor(0, action)
                value = max(value, self.minValue(successor, depth + 1, alpha, beta))

                #exit once alpha is greater than beta thereby pruning branches
                if value >= beta:
                    return value

                alpha = max(alpha, value)

        return value

    def minValue(self, state, depth, alpha, beta):
        """
        Ghost turn
        """
        
        # exit the recursion when hitting a base case
        if depth == self.getTreeDepth() or state.isWin() or state.isLose():
            return self.getEvaluationFunction()(state)

        value = float("inf")
        numAgents = state.getNumAgents()

        # loop over every ghost agent 
        for agentIndex in range(1, numAgents):
            legalActions = state.getLegalActions(agentIndex)
            
            # pick the best action for each ghost to take
            for action in legalActions:
                if action != Directions.STOP:
                    successor = state.generateSuccessor(agentIndex, action)
                    value = min(value, self.maxValue(successor, depth, alpha, beta))

                    #exit once alpha is greater than beta thereby pruning branches
                    if value <= alpha:
                        return value

                    beta = min(beta, value)

        return value


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    An expectimax agent.

    All ghosts should be modeled as choosing uniformly at random from their legal moves.

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Returns the expectimax action from the current gameState using
    `pacai.agents.search.multiagent.MultiAgentSearchAgent.getTreeDepth`
    and `pacai.agents.search.multiagent.MultiAgentSearchAgent.getEvaluationFunction`.
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action from the current gameState.
        """
        def max_agent(state, depth):
            """
            Pacmans turn
            """
            
            #exit recursion if base case is reached
            if state.isWin() or state.isLose() or depth == self.getTreeDepth():
                return self.getEvaluationFunction()(state)

            best_score = float("-inf")
            best_action = Directions.STOP

            # loop over legal actions for Pacman 
            for action in state.getLegalActions(0):
                successor = state.generateSuccessor(0, action)
                score = exp_value(successor, depth, 1)
                if score > best_score:
                    best_score = score
                    best_action = action

            # Return the best action or score depending on the depth.
            return best_action if depth == 0 else best_score

        def exp_value(state, depth, agent):
            """
            Expected value calculation for ghosts' turns.
            """
            
            #exit recursion if base case is reached
            if state.isWin() or state.isLose() or depth == self.getTreeDepth():
                return self.getEvaluationFunction()(state)

            actions = state.getLegalActions(agent)
            total_score = 0

            # Iterate over legal actions for the current ghost agent.
            for action in actions:
                successor = state.generateSuccessor(agent, action)
                # If you hit the last ghost, its pacmans turn
                if agent == state.getNumAgents() - 1:
                    total_score += max_agent(successor, depth + 1)
                # Otherwise, its the next ghosts turn
                else:
                    total_score += exp_value(successor, depth, agent + 1)

            return total_score / len(actions)
        
        return max_agent(gameState, 0)


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing,
    food-gobbling, unstoppable evaluation function.

    DESCRIPTION: This evaluation function considers several factors
    such as the score, distance to the closest food,
    distance to the closest ghost, and whether the action
    results in eating food or not. It assigns weights to these factors
    and combines them to produce a final evaluation score.
    """

    score = currentGameState.getScore()
    pacman_pos = currentGameState.getPacmanPosition()
    food_list = currentGameState.getFood().asList()
    ghost_states = currentGameState.getGhostStates()

    # Calculate distance to the nearest food
    min_food_distance = (
    min(distance.manhattan(pacman_pos, food) for food in food_list)
    if food_list
    else 0
    )

    # Penalize proximity to ghosts
    ghost_penalty = sum(
        1.0 / (distance.manhattan(pacman_pos, ghost.getPosition()) + 1)
        for ghost in ghost_states
    )

    # Incentivize pacman to go after powerups
    capsules_remaining = len(currentGameState.getCapsules())
    
    return score + (1 / (min_food_distance + 1)) - ghost_penalty - (10 * capsules_remaining)

class ContestAgent(MultiAgentSearchAgent):
    """
    Your agent for the mini-contest.

    You can use any method you want and search to any depth you want.
    Just remember that the mini-contest is timed, so you have to trade off speed and computation.

    Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
    just make a beeline straight towards Pacman (or away if they're scared!)

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)
