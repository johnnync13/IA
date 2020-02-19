# -*- coding: utf-8 -*-
#
# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    "*** YOUR CODE HERE ***"
    # Inicializamos la pila, ya que estamos en dfs
    stack = util.Stack()
    # inicializamos la lista de nodos visitados
    visited = []
    # anadimos a la lista el primer nodo, (estado inicial,action,coste)
    stack.push((problem.getStartState(), [], 0))
    # mientras frontera no este vacia
    while not stack.isEmpty():
        # (estado actual, accion, coste)
        curState, action, stepCost = stack.pop()
        # si el estado actual es un estado objetivo, devolvemos toda la accion
        if problem.isGoalState(curState):
            return action
        # si el estado actual no esta en visitados, lo anadimos
        if curState not in visited:
            visited.append(curState)
            # por cada estado sucesor del estado actual, expandimos creando un nodo y lo metemos en frontera
            # nuevo nodo (pos del sucesor, anadimos el nuevo movimiento, y aumentamos el coste)
            for pos, act, cost in problem.getSuccessors(curState):
                stack.push((pos, action + [act], cost + stepCost))

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    # Inicializamos la pila, ya que estamos en dfs
    stack = util.Queue()
    # inicializamos la lista de nodos visitados
    visited = []
    # anadimos a la lista el primer nodo, (estado inicial,action,coste)
    stack.push((problem.getStartState(), [], 0))
    # mientras frontera no este vacia
    while not stack.isEmpty():
        # (estado actual, accion, coste)
        curState, action, stepCost = stack.pop()
        # si el estado actual es un estado objetivo, devolvemos toda la accion
        if problem.isGoalState(curState):
            return action
        # si el estado actual no esta en visitados, lo anadimos
        if curState not in visited:
            visited.append(curState)
            # por cada estado sucesor del estado actual, expandimos creando un nodo y lo metemos en frontera
            # nuevo nodo (pos del sucesor, anadimos el nuevo movimiento, y aumentamos el coste)
            for pos, act, cost in problem.getSuccessors(curState):
                stack.push((pos, action + [act], cost + stepCost))

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    # Inicializamos la cola prioritaria, ya que utilizaremos la heuristica para tal fin
    queue = util.PriorityQueue()
    # inicializamos la lista de nodos visitados
    visited = []
    # anadimos a la lista el primer nodo, (estado inicial,action,coste, coste heuristica)
    queue.push((problem.getStartState(), [], 0), 0)
    # mientras frontera no este vacia
    while not queue.isEmpty():
        # (estado actual, accion, coste)
        curState, action, stepCost = queue.pop()
        # si el estado actual es un estado objetivo, devolvemos toda la accion
        if problem.isGoalState(curState):
            return action
        # si el estado actual no esta en visitados, lo anadimos
        if curState not in visited:
            visited.append(curState)
            # por cada estado sucesor del estado actual, expandimos creando un nodo y lo metemos en frontera
            # nuevo nodo (pos del sucesor, anadimos el nuevo movimiento, y aumentamos el coste)
            for pos, act, cost in problem.getSuccessors(curState):
                queue.push((pos, action + [act], cost + stepCost), cost + stepCost)

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    # Inicializamos la cola prioritaria, ya que utilizaremos la heuristica para tal fin
    queue = util.PriorityQueue()
    # inicializamos la lista de nodos visitados
    visited = []
    # anadimos a la lista el primer nodo, (estado inicial,action,coste, coste heuristica)
    queue.push((problem.getStartState(), [], 0), 0)
    # mientras frontera no este vacia
    while not queue.isEmpty():
        # (estado actual, accion, coste)
        curState, action, stepCost = queue.pop()
        # si el estado actual es un estado objetivo, devolvemos toda la accion
        if problem.isGoalState(curState):
            return action
        # si el estado actual no esta en visitados, lo anadimos
        if curState not in visited:
            visited.append(curState)
            # por cada estado sucesor del estado actual, expandimos creando un nodo y lo metemos en frontera
            # nuevo nodo (pos del sucesor, anadimos el nuevo movimiento, y aumentamos el coste)
            #Ademas anadimos la heuristica para hacer funcionar la cola prioritaria
            for pos, act, cost in problem.getSuccessors(curState):
                queue.push((pos, action + [act], cost + stepCost), (cost + stepCost + heuristic(pos, problem)))

def busquedaUB(problem, heuristic = nullHeuristic, beta = 35):
    # Inicializamos la cola prioritaria, ya que utilizaremos la heuristica para tal fin
    queue = util.PriorityQueue()
    queuep = util.PriorityQueue()
    # inicializamos la lista de nodos visitados
    visited = []
    # anadimos a la lista el primer nodo, (estado inicial,action,coste, coste heuristica)
    queue.push((problem.getStartState(), [], 0), 0)
    # mientras frontera no este vacia
    count=1
    while not queue.isEmpty():
        # (estado actual, accion, coste)
        curState, action, stepCost = queue.pop()
        count -= 1
        # si el estado actual es un estado objetivo, devolvemos toda la accion
        if problem.isGoalState(curState):
            return action
        # si el estado actual no esta en visitados, lo anadimos
        if curState not in visited:
            visited.append(curState)
            # por cada estado sucesor del estado actual, expandimos creando un nodo y lo metemos en frontera
            # nuevo nodo (pos del sucesor, anadimos el nuevo movimiento, y aumentamos el coste)
            # Ademas anadimos la heuristica para hacer funcionar la cola prioritaria
            for pos, act, cost in problem.getSuccessors(curState):
                queuep.push((pos, action + [act], cost + stepCost), (cost + stepCost + heuristic(pos, problem)))

            while not queuep.isEmpty():
                pos_, action_, cost_ = queuep.pop()
                if(len(queue.heap)< beta):
                    count+=1
                    queue.push((pos_, action_, cost_), (cost_ + heuristic(pos_, problem)))

                print(count)
# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
bus= busquedaUB
