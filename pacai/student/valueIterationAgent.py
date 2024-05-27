# from pacai.agents.learning.value import ValueEstimationAgent

# class ValueIterationAgent(ValueEstimationAgent):
#     """
#     A value iteration agent.

#     Make sure to read `pacai.agents.learning` before working on this class.

#     A `ValueIterationAgent` takes a `pacai.core.mdp.MarkovDecisionProcess` on initialization,
#     and runs value iteration for a given number of iterations using the supplied discount factor.

#     Some useful mdp methods you will use:
#     `pacai.core.mdp.MarkovDecisionProcess.getStates`,
#     `pacai.core.mdp.MarkovDecisionProcess.getPossibleActions`,
#     `pacai.core.mdp.MarkovDecisionProcess.getTransitionStatesAndProbs`,
#     `pacai.core.mdp.MarkovDecisionProcess.getReward`.

#     Additional methods to implement:

#     `pacai.agents.learning.value.ValueEstimationAgent.getQValue`:
#     The q-value of the state action pair (after the indicated number of value iteration passes).
#     Note that value iteration does not necessarily create this quantity,
#     and you may have to derive it on the fly.

#     `pacai.agents.learning.value.ValueEstimationAgent.getPolicy`:
#     The policy is the best action in the given state
#     according to the values computed by value iteration.
#     You may break ties any way you see fit.
#     Note that if there are no legal actions, which is the case at the terminal state,
#     you should return None.
#     """

from pacai.agents.learning.value import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
    def __init__(self, index, mdp, discountRate=0.9, iters=100, **kwargs):
        super().__init__(index, **kwargs)

        self.mdp = mdp
        self.discountRate = discountRate
        self.iters = iters
        self.values = {}

        self.valueIteration()

    def valueIteration(self):
        for i in range(self.iters):
            newValues = {}
            for state in self.mdp.getStates():
                if self.mdp.isTerminal(state):
                    newValues[state] = 0
                else:
                    bestAction = self.getPolicy(state)
                    newValues[state] = self.getQValue(state, bestAction)
            self.values = newValues

    def getValue(self, state):
        return self.values.get(state, 0.0)

    def getAction(self, state):
        return self.getPolicy(state)

    def getQValue(self, state, action):
        qValue = 0
        transitions = self.mdp.getTransitionStatesAndProbs(state, action)
        for nextState, prob in transitions:
            reward = self.mdp.getReward(state, action, nextState)
            qValue += prob * (reward + self.discountRate * self.getValue(nextState))
        return qValue

    def getPolicy(self, state):
        bestAction = None
        bestValue = float("-inf")
        for action in self.mdp.getPossibleActions(state):
            value = self.getQValue(state, action)
            if value > bestValue:
                bestValue = value
                bestAction = action
        return bestAction
