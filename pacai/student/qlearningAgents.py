from pacai.agents.learning.reinforcement import ReinforcementAgent
from pacai.util import reflection
import random

class QLearningAgent(ReinforcementAgent):
    """
    A Q-Learning agent.

    Some functions that may be useful:

    `pacai.agents.learning.reinforcement.ReinforcementAgent.getAlpha`:
    Get the learning rate.

    `pacai.agents.learning.reinforcement.ReinforcementAgent.getDiscountRate`:
    Get the discount rate.

    `pacai.agents.learning.reinforcement.ReinforcementAgent.getEpsilon`:
    Get the exploration probability.

    `pacai.agents.learning.reinforcement.ReinforcementAgent.getLegalActions`:
    Get the legal actions for a reinforcement agent.

    `pacai.util.probability.flipCoin`:
    Flip a coin (get a binary value) with some probability.

    `random.choice`:
    Pick randomly from a list.

    Additional methods to implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Compute the action to take in the current state.
    With probability `pacai.agents.learning.reinforcement.ReinforcementAgent.getEpsilon`,
    we should take a random action and take the best policy action otherwise.
    Note that if there are no legal actions, which is the case at the terminal state,
    you should choose None as the action.

    `pacai.agents.learning.reinforcement.ReinforcementAgent.update`:
    The parent class calls this to observe a state transition and reward.
    You should do your Q-Value update here.
    Note that you should never call this function, it will be called on your behalf.

    DESCRIPTION: <Write something here so we know what you did.>
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)
        self.qValues = {}

    def getAction(self, state):
        """
        Compute the action to take in the current state.
        """
        legalActions = self.getLegalActions(state)

        if not legalActions:
            return None  # Terminal state.

        epsilon = self.getEpsilon()

        #epsilon percent of the time choose a random action
        #otherwise take a greedy action
        if random.random() < epsilon:
            return random.choice(legalActions)
        else:
            return self.getPolicy(state)

    def update(self, state, action, nextState, reward):
        """
        Update the Q-value based on the observed transition.
        """
        alpha = self.getAlpha()
        gamma = self.getDiscountRate()
        qValue = self.getQValue(state, action)
        nextValue = self.getValue(nextState)
        updatedQValue = (1 - alpha) * qValue + alpha * (reward + gamma * nextValue)
        self.qValues[(state, action)] = updatedQValue

    def getQValue(self, state, action):
        """
        Get the Q-Value for a state and action pair.
        Should return 0.0 if the pair has never been seen.
        """
        return self.qValues.get((state, action), 0.0)

    def getValue(self, state):
        """
        Return the value of the best action in a state.
        """
        legalActions = self.getLegalActions(state)

        if not legalActions:
            return 0.0

        #return best action until a terminal state is reached
        return max([self.getQValue(state, action) for action in legalActions])

    def getPolicy(self, state):
        """
        Return the best action in a state.
        """
        legalActions = self.getLegalActions(state)

        if not legalActions:
            return None

        bestActions = []
        for action in legalActions:
            if self.getQValue(state, action) == self.getValue(state):
                bestActions.append(action)

        return random.choice(bestActions)

class PacmanQAgent(QLearningAgent):
    """
    Exactly the same as `QLearningAgent`, but with different default parameters.
    """

    def __init__(self, index, epsilon = 0.05, gamma = 0.8, alpha = 0.2, numTraining = 0, **kwargs):
        kwargs['epsilon'] = epsilon
        kwargs['gamma'] = gamma
        kwargs['alpha'] = alpha
        kwargs['numTraining'] = numTraining

        super().__init__(index, **kwargs)

    def getAction(self, state):
        """
        Simply calls the super getAction method and then informs the parent of an action for Pacman.
        Do not change or remove this method.
        """

        action = super().getAction(state)
        self.doAction(state, action)

        return action

class ApproximateQAgent(PacmanQAgent):
    """
    An approximate Q-learning agent.

    You should only have to overwrite `QLearningAgent.getQValue`
    and `pacai.agents.learning.reinforcement.ReinforcementAgent.update`.
    All other `QLearningAgent` functions should work as is.

    Additional methods to implement:

    `QLearningAgent.getQValue`:
    Should return `Q(state, action) = w * featureVector`,
    where `*` is the dotProduct operator.

    `pacai.agents.learning.reinforcement.ReinforcementAgent.update`:
    Should update your weights based on transition.

    DESCRIPTION: This class implements an approximate Q-learning agent
    that learns weights for features of states.
    """

    def __init__(self, index, extractor='pacai.core.featureExtractors.IdentityExtractor', **kwargs):
        super().__init__(index, **kwargs)
        self.featExtractor = reflection.qualifiedImport(extractor)
        self.weights = {}

    def final(self, state):
        """
        Called at the end of each game.
        """
        super().final(state)

    def getQValue(self, state, action):
        """
        Get the Q-value for a state and action pair.
        Should return `Q(state, action) = w * featureVector`,
        where `*` is the dotProduct operator.
        """
        features = self.featExtractor.getFeatures(self, state, action)
        qValue = 0.0

        for feature, value in features.items():
            if feature in self.weights:
                qValue += self.weights[feature] * value

        return qValue

    def update(self, state, action, nextState, reward):
        """
        Update the weight vector based on the observed transition.
        """
        alpha = self.getAlpha()
        gamma = self.getDiscountRate()
        difference = (reward + gamma * self.getValue(nextState)) - self.getQValue(state, action)
        features = self.featExtractor.getFeatures(self, state, action)

        for feature, value in features.items():
            if feature not in self.weights:
                self.weights[feature] = 0.0
            self.weights[feature] += alpha * difference * value