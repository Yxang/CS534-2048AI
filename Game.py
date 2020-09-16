import gym_2048
import gym
import numpy as np

from expectimaxAgent import ExpMaxAgent

trans_agent_to_env = {0: 0,
                      1: 3,
                      2: 2,
                      3: 1}


def game_play(Agent, render=False, **kwargs):
  env = gym.make('2048-v0')

  env.reset()
  if render:
    #env.render('human')
    print(env.board)

  done = False
  moves = 0
  score = 0
  next_state = env.board
  while not done:
    agent = Agent(next_state, **kwargs)
    action = agent.step()
    action = trans_agent_to_env[action]
    next_state, reward, done, info = env.step(action)
    score += reward
    moves += 1

    print('Next Action: "{}"\n\nTotal Score: {}'.format(
     gym_2048.Base2048Env.ACTION_STRING[action], score))
    if render:
      #env.render('human')
      print(env.board)

  print('\nTotal Moves: {}'.format(moves))

  greatest_block = np.max(next_state)

  return score, moves, np.max(next_state)
