# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util
import sys

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, game_state: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = game_state.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(game_state, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, current_game_state: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (new_food) and Pacman position after moving (new_pos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successor_game_state = current_game_state.generatePacmanSuccessor(action)
        new_pos = successor_game_state.getPacmanPosition()
        new_food = successor_game_state.getFood()
        new_ghost_states = successor_game_state.getGhostStates()
        newScaredTimes = [ghost_state.scaredTimer for ghost_state in new_ghost_states]

        "*** YOUR CODE HERE ***"

        # distancia da comida mais perto
        new_food = new_food.asList()
        if new_food:
            aux = []
            for food in new_food:
                aux.append(manhattanDistance(new_pos, food))
            score = min(aux)
        else:
            score = 1

        return successor_game_state.getScore() + 1/score

def scoreEvaluationFunction(current_game_state: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return current_game_state.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, game_state: GameState):
        """
        Returns the minimax action from the current game_state using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        game_state.getLegalActions(agent_index):
        Returns a list of legal actions for an agent
        agent_index=0 means Pacman, ghosts are >= 1

        game_state.generateSuccessor(agent_index, action):
        Returns the successor game state after an agent takes an action

        game_state.getNumAgents():
        Returns the total number of agents in the game

        game_state.isWin():
        Returns whether or not the game state is a winning state

        game_state.isLose():
        Returns whether or not the game state is a losing state
        """

        "*** YOUR CODE HERE ***"
        max_value, next_action = self.minimaxDecision(game_state, 0, self.depth)
        return next_action

    def minimaxDecision(self, game_state, agent_index, depth):

        # coloca os scores nas folhas da arvore
        if depth == 0 or game_state.isLose() or game_state.isWin():
            return self.evaluationFunction(game_state), None

        # se agente for pacman usa maxValue, se for fantasma usa minValue
        if agent_index == 0:
            return self.maxValue(game_state, agent_index, depth)
        else:
            return self.minValue(game_state, agent_index, depth)

    def minValue(self, game_state, agent_index, depth):

        min_score = float("inf")
        min_action = None

        # se o agente for o último fantasma, o prox agente é o pacman
        # se não o prox agente é o prox fantasma
        if agent_index == game_state.getNumAgents() - 1:
            next_agent = 0
            next_depth = depth - 1
        else:
            next_agent = agent_index + 1
            next_depth = depth

        # acha a melhor opção entre todas as ações possíveis
        for action in game_state.getLegalActions(agent_index):
            successor_game_state = game_state.generateSuccessor(agent_index, action)
            new_score, new_action = self.minimaxDecision(successor_game_state, next_agent, next_depth)

            if new_score < min_score:
                min_score = new_score
                min_action = action

        return min_score, min_action

    def maxValue(self, game_state, agent_index, depth):

        max_score = float("-inf")
        max_action = None

        # se o agente for o último fantasma, o prox agente é o pacman
        # se não o prox agente é o prox fantasma
        if agent_index == game_state.getNumAgents() - 1:
            next_agent = 0
            next_depth = depth - 1
        else:
            next_agent = agent_index + 1
            next_depth = depth

        # acha a melhor opção entre todas as ações possíveis
        for action in game_state.getLegalActions(agent_index):
            successor_game_state = game_state.generateSuccessor(agent_index, action)
            new_score, new_action = self.minimaxDecision(successor_game_state, next_agent, next_depth)

            if new_score > max_score:
                max_score = new_score
                max_action = action

        return max_score, max_action

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, game_state: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """

        "*** YOUR CODE HERE ***"

        max_value, next_action = self.alphaBetaDecision(game_state, 0, self.depth, float("-inf"), float("inf"))
        return next_action

    def alphaBetaDecision(self, game_state, agent_index, depth, alpha, beta):

        # coloca os scores nas folhas da arvore
        if depth == 0 or game_state.isLose() or game_state.isWin():
            return self.evaluationFunction(game_state), None

        # se agente for pacman usa alphaValue, se for fantasma usa betaValue
        if agent_index == 0:
            return self.alphaValue(game_state, agent_index, depth, alpha, beta)
        else:
            return self.betaValue(game_state, agent_index, depth, alpha, beta)

    def alphaValue(self, game_state, agent_index, depth, alpha, beta):

        max_score = float("-inf")
        max_action = None

        # se o agente for o último fantasma, o prox agente é o pacman
        # se não o prox agente é o prox fantasma
        if agent_index == game_state.getNumAgents() - 1:
            next_agent = 0
            next_depth = depth - 1
        else:
            next_agent = agent_index + 1
            next_depth = depth

        for action in game_state.getLegalActions(agent_index):
            successor_game_state = game_state.generateSuccessor(agent_index, action)
            new_score, aux = self.alphaBetaDecision(successor_game_state, next_agent, next_depth, alpha, beta)

            if new_score > max_score:
                max_score = new_score
                max_action = action

            if new_score > beta:
                return new_score, action

            alpha = max(alpha, max_score)

        return max_score, max_action

    def betaValue(self, game_state, agent_index, depth, alpha, beta):

        min_score = float("inf")
        min_action = None

        # se o agente for o último fantasma, o prox agente é o pacman
        # se não o prox agente é o prox fantasma
        if agent_index == game_state.getNumAgents() - 1:
            next_agent = 0
            next_depth = depth - 1
        else:
            next_agent = agent_index + 1
            next_depth = depth

        for action in game_state.getLegalActions(agent_index):
            successor_game_state = game_state.generateSuccessor(agent_index, action)
            new_score, aux = self.alphaBetaDecision(successor_game_state, next_agent, next_depth, alpha, beta)

            if new_score < min_score:
                min_score = new_score
                min_action = action
            
            if new_score < alpha:
                return new_score, action
            
            beta = min(beta, min_score)

        return min_score, min_action

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, game_state: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        
        max_value, next_action = self.expectimaxDecision(game_state, 0, self.depth)
        return next_action

    def expectimaxDecision(self, game_state, agent_index, depth):

        # add scores nas folhas
        if depth == 0 or game_state.isLose() or game_state.isWin():
            return self.evaluationFunction(game_state), None

        # se agente for pacman usa maxValue, se for fantasma usa expectationValue
        return self.maxValue(game_state, agent_index, depth) if agent_index == 0 else self.expectationValue(game_state, agent_index, depth)

    def maxValue(self, game_state, agent_index, depth):

        max_score = float("-inf")
        max_action = None

        # se o agente for o último fantasma, o prox agente é o pacman
        # se não o prox agente é o prox fantasma
        if agent_index == game_state.getNumAgents() - 1:
            next_agent = 0
            next_depth = depth - 1
        else:
            next_agent = agent_index + 1
            next_depth = depth

        # acha a melhor opção entre todas as ações possíveis
        for action in game_state.getLegalActions(agent_index):
            successor_game_state = game_state.generateSuccessor(agent_index, action)
            new_score, new_action = self.expectimaxDecision(successor_game_state, next_agent, next_depth)

            if new_score > max_score:
                max_score = new_score
                max_action = action

        return max_score, max_action

    def expectationValue(self, game_state, agent_index, depth):

        score = 0
        action = None
        actions = game_state.getLegalActions(agent_index)

        # se o agente for o último fantasma, o prox agente é o pacman
        # se não o prox agente é o prox fantasma
        if agent_index == game_state.getNumAgents() - 1:
            next_agent = 0
            next_depth = depth - 1
        else:
            next_agent = agent_index + 1
            next_depth = depth

        # melhor opção entre todas as ações possíveis
        for action in actions:
            successor_game_state = game_state.generateSuccessor(agent_index, action)
            new_score, new_action = self.expectimaxDecision(successor_game_state, next_agent, next_depth)
            score += new_score

        score = score / len(actions)

        return score, action

def betterEvaluationFunction(current_game_state: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"

    new_food = current_game_state.getFood()
    game_score = current_game_state.getScore() 
    new_pos = current_game_state.getPacmanPosition()
    new_ghost_states = current_game_state.getGhostStates()

    # dist do fantasma mais perto
    aux = []
    for ghost_state in new_ghost_states:
        aux.append(manhattanDistance(new_pos, ghost_state.getPosition()))
    closest_ghost = min(aux)

    # dist da comida mais perto
    new_food = new_food.asList()
    if new_food:
        aux = []
        for food in new_food:
            aux.append(manhattanDistance(new_pos, food))
        closest_food = min(aux)
    else:
        closest_food = 0

    return (10 / (closest_food + 1)) + (200 * game_score) 

# Abbreviation
better = betterEvaluationFunction
