import gym
import gym_2048

class State:
    def __init__(self, env=None, score=0, done=False):
        if env is None:
            env = gym.make('2048-v0')
            env.reset()
        self.env = env
        self.score = score
        self.done = done

    def getPossibleActions(self):
        return (0, 1, 2, 3)

    def isTerminal(self):
        return self.done

    def takeAction(self, action):
        new_env = gym.make('2048-v0')
        new_env.board = self.env.board
        new_node = State(new_env, self.score, self.done)
        next_state, reward, done, info = new_node.env.step(action)
        new_node.score += reward
        new_node.done = done
        new_node.state = next_state
        return new_node

    def takeRealAction(self, action):
        next_state, reward, done, info = self.env.step(action)
        self.score += reward
        self.done = done
        self.state = next_state

    def getReward(self):
        return self.score

    @property
    def board(self):
        return self.env.board

