from expectimax import Node
import numpy as np

class Agent:
    '''Agent Base.'''

    def __init__(self, board, heuristic_f, display=None, ):
        self.board = board
        self.display = display
        self.heuristic_f = heuristic_f

    def step(self):
        direction = int(input("0: left, 1: down, 2: right, 3: up = ")) % 4
        return direction


class ExpMaxAgent(Agent):

    def step(self):
        '''To define the agent's 1-step behavior given the `game`.
        You can find more instance in [`agents.py`](game2048/agents.py).

        :return direction: 0: left, 1: down, 2: right, 3: up
        '''
        board = self.board
        root = Node(board, weight=1, depth=1, max_depth=3, heuristic_f=self.heuristic_f)
        direction = root.best_move()
        return direction