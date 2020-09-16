import multiprocessing
import json
import Game
from expectimaxAgent import ExpMaxAgent
from expectimax import heuristic_f1, heuristic_f2, heuristic_f3
from tqdm import tqdm


def play_on_h(h):
    return Game.game_play(ExpMaxAgent, heuristic_f=h)


if __name__ == '__main__':
    cores = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=4)
    N_ROUNDS = 100

    results = {'f1': {'score': [],
                      'move': [],
                      'max_block': []},
               'f2': {'score': [],
                      'move': [],
                      'max_block': []},
               'f3': {'score': [],
                      'move': [],
                      'max_block': []}}

    print('aaa')

    for score, moves, max_block in tqdm(pool.imap(play_on_h, [heuristic_f1] * N_ROUNDS)):
        results['f1']['score'].append(score)
        results['f1']['move'].append(moves)
        results['f1']['max_block'].append(max_block)

    with open('stats.json', 'w') as f:
        json.dump(results)

