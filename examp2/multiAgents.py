# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for 
# educational purposes provided that (1) you do not distribute or publish 
# solutions, (2) you retain this notice, and (3) you provide clear 
# attribution to UC Berkeley, including a link to 
# http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero 
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and 
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent


class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        closestghost = min([manhattanDistance(newPos, ghost.getPosition()) for ghost in newGhostStates])

        if closestghost:
            ghost_dist = -10 / closestghost
        else:
            ghost_dist = -1000

        foodList = newFood.asList()
        if foodList:
            closestfood = min([manhattanDistance(newPos, food) for food in foodList])
        else:
            closestfood = 0
        return (-2 * closestfood) + ghost_dist - (100 * len(foodList))


def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()


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

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        agents = gameState.getNumAgents()
        depth = agents * self.depth

        def minimax(gameState, depth, agent):
            legalActions = gameState.getLegalActions(agent)
            if depth == 0 or len(legalActions) == 0: #si estamos en una estado terminal
                return self.evaluationFunction(gameState)
            if agent == 0: #si es pacman
                return max(
                    minimax(gameState.generateSuccessor(agent, newState), depth - 1, (agent + 1) % agents) for newState
                    in legalActions)
            else: #minimizamos en los fantasmas
                return min(
                    minimax(gameState.generateSuccessor(agent, newState), depth - 1, (agent + 1) % agents) for newState
                    in legalActions)

        return max([(minimax(gameState.generateSuccessor(0, action), depth - 1, 1), action) for action in
                    gameState.getLegalActions(0)])[1]


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        agents = gameState.getNumAgents()
        depth = agents * self.depth
        alpha = float("-inf")
        beta = float("inf")

        def alphabetapruning(gameState, depth, alpha, beta, agentIndex):
            legalActions = gameState.getLegalActions(agentIndex)
            if depth == 0 or len(legalActions) == 0:
                return (self.evaluationFunction(gameState), None)

            result_action = None
            if agentIndex == 0:
                result_v = float("-inf")
                for action in legalActions:
                    v = alphabetapruning(gameState.generateSuccessor(agentIndex, action), depth - 1, alpha, beta,
                                     (agentIndex + 1) % agents)
                    if v[0] > result_v: #nos quedamos con el mejor resultado. Podria haber utilizado la funcion max
                        (result_v, result_action) = (v[0], action)
                    if result_v > beta: break
                    alpha = max(alpha, result_v) #actualizamos alpha
            else:
                result_v = float("inf")
                for action in legalActions:
                    v = alphabetapruning(gameState.generateSuccessor(agentIndex, action), depth - 1, alpha, beta,
                                     (agentIndex + 1) % agents)
                    if v[0] < result_v:
                        (result_v, result_action) = (v[0], action)
                    if result_v < alpha: break
                    beta = min(beta, result_v) #actualizamos beta

            return (result_v, result_action)

        result = alphabetapruning(gameState, depth, alpha, beta, 0)
        return result[1]


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        agents = gameState.getNumAgents()
        depth = agents * self.depth

        def expectimax(gameState, depth, agentIndex):
            legalActions = gameState.getLegalActions(agentIndex)
            if depth == 0 or len(legalActions) == 0:
                return (self.evaluationFunction(gameState), None)

            result_action = None
            if agentIndex == 0:
                result_v = float("-inf")
                for action in legalActions:
                    v = expectimax(gameState.generateSuccessor(agentIndex, action), depth - 1,
                                     (agentIndex + 1) % agents)
                    if v[0] > result_v:
                        (result_v, result_action) = (v[0], action)
            else:
                result_v = 0
                for action in legalActions:
                    v = expectimax(gameState.generateSuccessor(agentIndex, action), depth - 1,
                                     (agentIndex + 1) % agents)
                    result_v += v[0]
                result_v /= float(len(legalActions)) #sumamos nuestros resultados y dividimos entre el numero total de
                                                    #acciones
            return (result_v, result_action)

        result = expectimax(gameState, depth, 0)
        return result[1]


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <He pensado que podria implementarl de forma sencilla, dandole ciertos
      valores de peso a mis conclusiones.
      1. Utilizo suma de manhadttan distance para las comidas como en la practica anterior
      2.Calculo la distancia con cada uno de los fantasmas
        a. Compruebo si el fantasma no esta asustado y esta a corta distancia, huyo!!!!!
            b. Compruebo si el fantasma esta asustado y a una distancia suficiente, lo suficiente para acercarme
        c. Compruebo si el fantasma esta asustado, me lo como
      3. Despues junto todos los pesos en total>
    """
    "*** YOUR CODE HERE ***"
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood().asList()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    food = sum([manhattanDistance(food, newPos) for food in newFood])

    ghost = 0
    for i in range(len(newGhostStates)):
        d = manhattanDistance(newPos,newGhostStates[i].getPosition())
        if newScaredTimes[i] == 0 and d < 1:
            ghost -= 1. / (1 - d)
        elif newScaredTimes[i] < d:
            ghost += 1. / d
        if newScaredTimes[0] > 0:
            ghost += 20.0
    resultado = 1. / (1 + food * len(newFood)) + ghost + currentGameState.getScore()
    return resultado



# Abbreviation
better = betterEvaluationFunction


class expectimaxMinimaxAgent2(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        agents = gameState.getNumAgents()
        depth = agents * self.depth

        def expectimaxMinimaxAgent2(gameState, depth, agentIndex):
            legalActions = gameState.getLegalActions(agentIndex)
            if depth == 0 or len(legalActions) == 0:
                return (self.evaluationFunction(gameState), None)

            result_action = None
            if agentIndex == 0:
                result_v = float("-inf")
                for action in legalActions:
                    v = expectimaxMinimaxAgent2(gameState.generateSuccessor(agentIndex, action), depth - 1, (agentIndex + 1) % agents)
                    if v[0] > result_v:
                        (result_v, result_action) = (v[0], action)
                return (result_v, result_action)
            else:
                if agentIndex % 2 == 1:
                    result_v = float("inf")
                    for action in legalActions:
                        v = expectimaxMinimaxAgent2(gameState.generateSuccessor(agentIndex, action), depth - 1, (agentIndex + 1) % agents)

                        if v[0] < result_v:
                            (result_v, result_action) = (v[0], action)
                    return (result_v, result_action)
                else:
                    result_v = 0
                    result_aux = []
                    cont = len(legalActions)
                    if (cont<=1):
                        for action in legalActions:
                            v = expectimaxMinimaxAgent2(gameState.generateSuccessor(agentIndex, action), depth - 1, (agentIndex + 1) % agents)
                            result_v += v[0]
                        result_v /= float(len(legalActions))

                    else:
                        cont = 1
                        eliminaciones = 0
                        for action in legalActions:
                            succesors = gameState.generateSuccessor(agentIndex, action)
                            list = [succesors]
                            list_values = [succesors.getScore()]
                            max_value = max(list_values)
                            for i in list:
                                if i.getScore() == max_value and cont>0:
                                    cont -=1
                                    eliminaciones +=1
                                else:
                                    v = expectimaxMinimaxAgent2(succesors, depth - 1, (agentIndex + 1) % agents)
                                    result_v += v[0]
                                    cont = 1
                        result_v /= float((len(legalActions)-eliminaciones))

            return (result_v, result_action)

        result = expectimaxMinimaxAgent2(gameState, depth, 0)
        return result[1]