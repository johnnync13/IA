# -*- coding: utf-8 -*-
# qlearningAgents.py
# ------------------
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


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random, util, math


class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """

    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        "*** YOUR CODE HERE ***"
        self.QValues = {}
        self.timesV = util.Counter()

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        #if (state, action) in self.QValues:
            #return self.QValues[(state, action)]
        #else:
            #return 0.0

        #Como podras observar he cambiado mi propio codigo para obtener el retorno en una linea.

        return self.QValues[(state, action)] if (state,action) in self.QValues else 0.0
    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        #Retornar el mejor valor encontrado en los posibles movimientos legales
        QValues = [self.getQValue(state, action) for action in self.getLegalActions(state)]
        return 0.0 if not len(QValues) else max(QValues)

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        # Devuelve una acción random de entre las posibles mejores o con mismo valor
        "*** YOUR CODE HERE ***"
        best_value = self.getValue(state)
        best_actions = [action for action in self.getLegalActions(state) if self.getQValue(state, action) == best_value]
        return None if not len(best_actions) else random.choice(best_actions)

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legalActions = self.getLegalActions(state)
        action = None
        # Use el valor de epsilon para obtener una acción aleatoria
        "*** YOUR CODE HERE ***"
        if util.flipCoin(self.epsilon):
            action = random.choice(legalActions)
        else:
            action = self.getPolicy(state)

        #Guardamos las acciones
        self.timesV[(action,state)]
        self.timesV[(action,state)] +=1
        return action

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """

        """
         Realizamos el update cuando llegamos al estado s'
         utilizando la formula de diferencia temporal con el estado actual,
         haciendo un híbrido.
         Q(s,a) = Q(s,a) + alpha * (R(s,a,s) + gamma * max{a'}[Q(s',a')] - Q(s,a) )
         tambien se puede escribir de la forma:
         - Q(s, a) = (1-alpha) * Q(s, a) + alpha * (R(s,a,s') + disc * max{a'}[Q(s',a')])

        - alpha : tasa de aprendizaje
        -Q(s,a) QValor actual
        -R(s,a,s') recomenpensa al llevar acabo una accion
        -gamma coeficiente de descuento por realizar una accion futura
        """
        "*** YOUR CODE HERE ***"
        gamma = self.discount
        alpha = self.alpha
        QValue = self.getQValue(state, action)
        #next_value = self.getValue(nextState)
        numerador = 0
        denominador = 0
        for valor in self.getLegalActions(state):
            cont_num = self.timesV.get((state, action))
            if cont_num is None:
                cont_num = 0
            cont_denom = self.timesV.get((state,valor))
            if cont_denom is None:
                cont_denom = 0
            numerador += (QValue * cont_num)
            denominador += cont_denom

        newQV = numerador / (denominador + 1)
        sample = reward + (gamma * newQV)
        new_value = (1 - alpha) * QValue + alpha * sample

        #new_value = QValue + (alpha * ((reward + gamma * next_value) - QValue))
        # new_value = (1 - alpha) * QValue + alpha * (reward + gamma * next_value)
        self.QValues[(state, action)] = new_value

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class QLearningAgentPractica(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """

    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        "*** YOUR CODE HERE ***"
        self.QValues = {}

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        #if (state, action) in self.QValues:
            #return self.QValues[(state, action)]
        #else:
            #return 0.0

        #Como podras observar he cambiado mi propio codigo para obtener el retorno en una linea.

        return self.QValues[(state, action)] if (state,action) in self.QValues else 0.0
    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        #Retornar el mejor valor encontrado en los posibles movimientos legales
        QValues = [self.getQValue(state, action) for action in self.getLegalActions(state)]
        return 0.0 if not len(QValues) else max(QValues)

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        # Devuelve una acción random de entre las posibles mejores o con mismo valor
        "*** YOUR CODE HERE ***"
        best_value = self.getValue(state)
        best_actions = [action for action in self.getLegalActions(state) if self.getQValue(state, action) == best_value]
        return None if not len(best_actions) else random.choice(best_actions)

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legalActions = self.getLegalActions(state)
        action = None
        # Use el valor de epsilon para obtener una acción aleatoria
        "*** YOUR CODE HERE ***"
        if util.flipCoin(self.epsilon):
            action = random.choice(legalActions)
        else:
            action = self.getPolicy(state)

        return action

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """

        """
         Realizamos el update cuando llegamos al estado s'
         utilizando la formula de diferencia temporal con el estado actual,
         haciendo un híbrido.
         Q(s,a) = Q(s,a) + alpha * (R(s,a,s) + gamma * max{a'}[Q(s',a')] - Q(s,a) )
         tambien se puede escribir de la forma:
         - Q(s, a) = (1-alpha) * Q(s, a) + alpha * (R(s,a,s') + disc * max{a'}[Q(s',a')])

        - alpha : tasa de aprendizaje
        -Q(s,a) QValor actual
        -R(s,a,s') recomenpensa al llevar acabo una accion
        -gamma coeficiente de descuento por realizar una accion futura
        """
        "*** YOUR CODE HERE ***"
        gamma = self.discount
        alpha = self.alpha
        QValue = self.getQValue(state, action)
        next_value = self.getValue(nextState)
        new_value = QValue + (alpha * ((reward + gamma * next_value) - QValue))
        # new_value = (1 - alpha) * QValue + alpha * (reward + gamma * next_value)

        self.QValues[(state, action)] = new_value

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)



class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    #He estado probando por terminal y por aqui diferentes valores.
    #def __init__(self, epsilon=0.05, gamma=0.9, alpha=-0.04, numTraining=0, **args):
    def __init__(self, epsilon=0.05, gamma=0.8, alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon  # Ratio de aprendizaje
        args['gamma'] = gamma  # Descuento
        args['alpha'] = alpha  # tasa de aprendizaje multiplica al TD, formula diferencia temporal.
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self, state)
        self.doAction(state, action)
        return action
