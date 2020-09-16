import numpy as np

'''
direction:
    0: left
    1: down
    2: right
    3: up
'''


WeightMatrix1 = np.matrix('1 2 3 4; 8 7 6 5; 9 10 11 12; 16 15 14 13')
WeightMatrix2 = np.matrix('7 6 5 4; 6 5 4 3; 5 4 3 2; 4 3 2 1')
WeightMatix = WeightMatrix1
BlankBonus = 10
PROBS = [0.9, 0.1]


def get_weight(a, b):
    sum = np.sum(np.multiply(a, b)).item()
    return sum


def countZero(board):
    return np.count_nonzero(board == 0)


def heuristic_f1(board):
    WeightMatix = WeightMatrix1
    score = get_weight(WeightMatix, board) + countZero(board)*BlankBonus
    return score


def heuristic_f2(board):
    WeightMatix = WeightMatrix2
    score = get_weight(WeightMatix, board) + countZero(board)*BlankBonus
    return score


def heuristic_f3(board):
    score = countZero(board)
    return score


class Node:
    def __init__(self, board, weight, depth, max_depth=3, heuristic_f=heuristic_f2):
        self.board = board
        self.weight = weight
        self.depth = depth
        self.max_depth = max_depth
        self.children = {0: [],
                         1: [],
                         2: [],
                         3: []}
        self.h = {0: 0,
                  1: 0,
                  2: 0,
                  3: 0}
        self._best_move = None
        self._best_h = None
        self.heuristic_f = heuristic_f

    def __repr__(self):
        board = "State:\n"
        for row in self.board:
            board += ('\t' + '{:8d}' *
                      4 + '\n').format(*map(int, row))
        board += "Prob: {0:2f}".format(self.weight)
        return board

    @property
    def best_h(self):

        if self._best_h is not None:
            return self._best_h

        if self.depth == self.max_depth:
            self._best_h = self.heuristic_f(self.board)
        elif self.depth < self.max_depth:
            for move in (0, 1, 2, 3):
                if len(self.children[move]):
                    hs = [c.best_h for c in self.children[move]]
                    probs = [c.weight for c in self.children[move]]
                    h = np.average(hs, weights=probs)
                    self.h[move] = h
            self._best_move = sorted([0, 1, 2, 3], key=lambda x: self.h[x], reverse=True)[0]
            self._best_h = self.h[self._best_move]
        else:
            raise RuntimeError('depth > max_depth')

        return self._best_h

    def expand(self):
        if self.depth <= self.max_depth:
            for move in (0, 1, 2, 3):
                boards, probs = self._all_board_on_move(move)
                if len(boards) > 0:
                    for board, prob in zip(boards, probs):
                        node = Node(board, prob, self.depth + 1, self.max_depth)
                        self.children[move].append(node)
                        node.expand()

    def _all_board_on_move(self, move):
        assert move in (0, 1, 2, 3)
        boards = []
        probs = []
        moved_board = fake_move(self.board, move)
        zero_loc = [(i, j) for i in range(4) for j in range(4) if moved_board[i, j] == 0]
        for new, prob in zip([2, 4], PROBS):
            for i, j in zero_loc:
                new_board = moved_board.copy()
                new_board[i, j] = new
                boards.append(new_board)
                probs.append(prob)
        return boards, probs

    def best_move(self):
        self.expand()
        best_h = self.best_h
        best_move = self._best_move
        return best_move


def fake_move(board, move):
    board = board.copy()
    board_to_left = np.rot90(board, -move)
    for row in range(4):
        core = _merge(board_to_left[row])
        board_to_left[row, :len(core)] = core
        board_to_left[row, len(core):] = 0

    # rotation to the original
    return np.rot90(board_to_left, move)


def _merge(row):
    '''merge the row, there may be some improvement'''
    non_zero = row[row != 0]  # remove zeros
    core = [None]
    for elem in non_zero:
        if core[-1] is None:
            core[-1] = elem
        elif core[-1] == elem:
            core[-1] = 2 * elem
            core.append(None)
        else:
            core.append(elem)
    if core[-1] is None:
        core.pop()
    return core
