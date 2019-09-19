# -*- coding: utf-8 -*-
# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util
# import sys


class SearchProblem:
    """This class outlines the structure of a search problem, but doesn't implement
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
    return [s, s, w, s, w, w, s, w]


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
    print("Start: ", problem.getStartState())
    print("Is the start the goal: ",
          problem.isGoalState(problem.getStartState()))
    print("Start's successors: ",
          problem.getSuccessors(problem.getStartState()))
    util.raiseNotDefined()


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the
    nearest goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def testHeuristic(position, problem=None):
    xy1 = position
    xy2 = problem.goal
    return abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])


def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first"""
    from util import PriorityQueue

    # print("\nStart position: " + str(problem.getStartState()))
    # print("\nGoal: " + str(problem.goal))
    # print("Heuristic for the start state: " + str(heuristic(problem.getStartState(), problem)))

    currentState = problem.getStartState()
    nodesToVisit = PriorityQueue()
    tree = {}
    tree[str(currentState)] = [None, heuristic(currentState, problem), None]  # {"node": [parent node, cost to get to the son using this path, action to get from the parent to the son] }

    # iter = 1
    while(not(problem.isGoalState(currentState))):
        # print('\nIteration #' + str(iter))
        # print('Current position: ' + str(currentState))
        successors = problem.getSuccessors(currentState)

        # Visiting the current node
        # print('Possible actions for the current node:')
        for suc in successors:
            cost = heuristic(suc[0], problem) + suc[2]
            tracebackState = currentState
            while(tracebackState != problem.getStartState()):
                cost += 1
                tracebackState = tree[str(tracebackState)][0]

            # Checking the parenting tree to avoid loops and updating it when needed
            # Update is needed if the son does not exist or the cost of the found path is smaller than the one stored in tree
            if((str(suc[0]) not in tree.keys()) or (cost < tree[str(suc[0])][1])):
                # print('Moving to ' + str(suc[0]) + ' Cost: ' + str(cost))
                nodesToVisit.update(suc[0], cost)
                tree[str(suc[0])] = [currentState, cost, suc[1]]
                # print('Tree updated: ' + '\'' + str(suc[0]) + '\' : ' + str([currentState, cost, suc[1]]))
            # else:
                # print('Tree did not update: ' + '\'' + str(suc[0]) + '\' : ' + str([currentState, cost, suc[1]]) +
                #       ' because previous cost was ' + str(tree[str(suc[0])][1]))
        # print('Parenting tree: ')
        # print(tree)

        # Updating to the next node
        currentState = nodesToVisit.pop()

        # iter += 1

    # print('\nFinal tree: ' + str(tree) + '\n')

    # Building the path based on the parenting tree
    solution = []
    while(currentState != problem.getStartState()):
        solution.insert(0, tree[str(currentState)][2])
        currentState = tree[str(currentState)][0]

    # print('Solution: ' + str(solution))

    return solution


def aStarSearchTimer(problem, heuristic=nullHeuristic):
    from timeit import default_timer
    repetitions = 50
    start = default_timer()
    for i in range(repetitions):
        solution = aStarSearch(problem, heuristic)
    end = default_timer()
    print('Execution time: ' + str((end - start)) + 's')
    return solution


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
timer = aStarSearchTimer
