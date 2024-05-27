"""
Analysis question.
Change these default values to obtain the specified policies through value iteration.
If any question is not possible, return just the constant NOT_POSSIBLE:
```
return NOT_POSSIBLE
```
"""

NOT_POSSIBLE = None

def question2():
    """
    No randomality, always goes the optimal path
    """

    answerDiscount = 0.9
    answerNoise = 0

    return answerDiscount, answerNoise

def question3a():
    """
    It costs more to not go in a good direction
    Some randomness involved
    """

    answerDiscount = 0.9
    answerNoise = 0.31
    answerLivingReward = -1.1

    return answerDiscount, answerNoise, answerLivingReward

def question3b():
    """
    The answer is more heavily discounted
    The living reward is lower
    """

    answerDiscount = 0.5
    answerNoise = 0.2
    answerLivingReward = -1

    return answerDiscount, answerNoise, answerLivingReward

def question3c():
    """
    It costs more to live
    """

    answerDiscount = 0.9
    answerNoise = 0.2
    answerLivingReward = -1

    return answerDiscount, answerNoise, answerLivingReward

def question3d():
    """
    Let the agent roam around 
    No living expense
    """

    answerDiscount = 0.9
    answerNoise = 0.2
    answerLivingReward = 0.0

    return answerDiscount, answerNoise, answerLivingReward

def question3e():
    """
    A lot of randomness and expensive to live
    Meaning it will scope out and
    figure out the right moves
    """

    answerDiscount = 0.9
    answerNoise = 1
    answerLivingReward = -2

    return answerDiscount, answerNoise, answerLivingReward

def question6():
    """
    Its not possible to find the optimal
    policy in only 50 episodes
    """

    return NOT_POSSIBLE

if __name__ == '__main__':
    questions = [
        question2,
        question3a,
        question3b,
        question3c,
        question3d,
        question3e,
        question6,
    ]

    print('Answers to analysis questions:')
    for question in questions:
        response = question()
        print('    Question %-10s:\t%s' % (question.__name__, str(response)))