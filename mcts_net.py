from tqdm import tqdm
import numpy as np
import os
import json
import torch
import torch.utils.data as data
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import time
import random
from torch.autograd import Variable
import h5py
from collections import defaultdict

np.random.seed(2019)

N_FEATURE = 32


class FeatureBlock(nn.Module):
    def __init__(self, n_conv1=4, n_conv2=8, n_fc_f=16, n_out=N_FEATURE):
        super(FeatureBlock, self).__init__()

        self.conv1 = nn.Conv1d(2, n_conv1, 5)
        self.conv2 = nn.Conv1d(n_conv1, n_conv2, 5)
        self.fc_f = nn.Linear(16, n_fc_f)

        self.fc_1 = nn.Linear(112, n_out)

    def forward(self, x):
        n_batch = x.size()[0]

        x_h = x.view(n_batch, 1, -1)
        x_v = x.transpose(2, 1).reshape(n_batch, 1, -1)
        input_tensor = torch.cat((x_h, x_v), 1)

        f1 = F.leaky_relu(self.conv1(input_tensor))
        f1 = F.leaky_relu(self.conv2(f1))

        # f2 = F.LeakyReLU(self.fc_f(x_h.view(1, -1)))

        # f = torch.cat((f1, f2), 1)

        # out = F.LeakyReLU(self.fc_1(f))

        return f1


class OutBlock(nn.Module):
    def __init__(self, n_input=64, n_conv1_in=8, n_conv1_out=16):
        super(OutBlock, self).__init__()

        # value
        self.vfc_1 = nn.Linear(n_input, 32)
        self.vfc_2 = nn.Linear(32, 1)

        # policy
        self.conv1 = nn.Conv1d(n_conv1_in, n_conv1_out, 3)
        self.pfc_1 = nn.Linear(96, 32)
        self.pfc_2 = nn.Linear(32, 4)

    def forward(self, x):
        n_batch = x.size()[0]

        # value
        v = F.leaky_relu(self.vfc_1(x.view(n_batch, -1)))
        v = F.leaky_relu(self.vfc_2(v))

        # policy
        p = F.leaky_relu(self.conv1(x))
        p = F.leaky_relu(self.pfc_1(p.view(n_batch, -1)))
        p = F.log_softmax(self.pfc_2(p), dim=1)

        return v, p


class Net2048(nn.Module):
    def __init__(self):
        super(Net2048, self).__init__()

        self.feature = FeatureBlock()
        self.out = OutBlock()

    def forward(self, x):
        feature = self.feature(x)
        v, p = self.out(feature)

        return v, p


class ValueLoss(nn.Module):
    def __init__(self):
        super(ValueLoss, self).__init__()

    def forward(self, value_hat, value_true):
        value_error = (value_hat - value_true.log()) ** 2
        value_error = value_error.mean()
        return value_error


PolicyLoss = nn.NLLLoss


class Game2048Loss(nn.Module):
    def __init__(self):
        super(Game2048Loss, self).__init__()
        self.valueloss = ValueLoss()
        self.policyloss = PolicyLoss()

    def forward(self, value_hat, value_true, policy_hat, policy_true):
        loss = self.valueloss(value_hat, value_true) + self.policyloss(policy_hat, policy_true)
        return loss


import time
import math
import random
import numpy as np


def nnPolicy(state, nn, max_depth=10):
    depth = 0
    while not state.isTerminal() and depth < 10:
        try:
            board = state.board
            board = board.copy().reshape(1, 4, 4)
            nn_input = torch.from_numpy(board).float().to(device)
            _, p = nn(nn_input)
            action = torch.argmax(p).item()
        except IndexError:
            raise Exception("Non-terminal state has no possible actions: " + str(state))
        state = state.takeAction(action)
        depth += 1
    return state.getReward()


class treeNode():
    def __init__(self, state, parent, depth):
        self.state = state
        self.depth = depth
        self.isTerminal = state.isTerminal()
        self.isFullyExpanded = self.isTerminal
        self.parent = parent
        self.numVisits = 0
        self.totalReward = 0
        self.children = {}


class mcts():
    def __init__(self, timeLimit=None, iterationLimit=None, explorationConstant=1 / math.sqrt(2),
                 rolloutPolicy=nnPolicy, nn=nn):
        if timeLimit != None:
            if iterationLimit != None:
                raise ValueError("Cannot have both a time limit and an iteration limit")
            # time taken for each MCTS search in milliseconds
            self.timeLimit = timeLimit
            self.limitType = 'time'
        else:
            if iterationLimit == None:
                raise ValueError("Must have either a time limit or an iteration limit")
            # number of iterations of the search
            if iterationLimit < 1:
                raise ValueError("Iteration limit must be greater than one")
            self.searchLimit = iterationLimit
            self.limitType = 'iterations'
        self.explorationConstant = explorationConstant
        self.rollout = rolloutPolicy
        self.nn = nn
        nn.eval()

    def search(self, initialState):
        self.root = treeNode(initialState, None, 0)

        if self.limitType == 'time':
            timeLimit = time.time() + self.timeLimit / 1000
            while time.time() < timeLimit:
                self.executeRound()
        else:
            for i in range(self.searchLimit):
                self.executeRound()

        bestChild = self.getBestChild(self.root, 0)
        return self.getAction(self.root, bestChild)

    def executeRound(self):
        node = self.selectNode(self.root)
        reward = self.rollout(node.state, self.nn)
        self.backpropogate(node, reward)

    def selectNode(self, node):
        while not node.isTerminal:
            if node.isFullyExpanded:
                node = self.getBestChild(node, self.explorationConstant)
            else:
                return self.expand(node)
        return node

    def expand(self, node):
        actions = node.state.getPossibleActions()
        for action in actions:
            if action not in node.children:

                newNode = treeNode(node.state.takeAction(action), node, node.depth + 1)

                node.children[action] = newNode
                if len(actions) == len(node.children):
                    node.isFullyExpanded = True
                return newNode

        raise Exception("Should never reach here")

    def backpropogate(self, node, reward):
        while node is not None:
            node.numVisits += 1
            node.totalReward += reward
            node = node.parent

    def getBestChild(self, node, explorationValue):
        bestValue = float("-inf")
        bestNodes = []
        for child in node.children.values():
            nodeValue = child.totalReward / child.numVisits + explorationValue * math.sqrt(
                2 * math.log(node.numVisits) / child.numVisits)
            if nodeValue > bestValue:
                bestValue = nodeValue
                bestNodes = [child]
            elif nodeValue == bestValue:
                bestNodes.append(child)
        return random.choice(bestNodes)

    def getAction(self, root, bestChild):
        for action, node in root.children.items():
            if node is bestChild:
                return action


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


def single_game():
    global net
    state_and_action = []
    thisState=State()
    n = 0
    while not thisState.isTerminal():
        mcts_ins = mcts(iterationLimit=64, nn=net)
        state = thisState.board
        best_action = mcts_ins.search(thisState)
        thisState.takeRealAction(best_action)
        # print('n %d finished' % n)
        n += 1
        state_and_action.append((state, best_action))

    score = thisState.score
    max_block = np.max(thisState.board)
    return state_and_action, score, max_block


def get_data(n_rounds=16):
    data_dict = {'state_and_action': [],
                 'score': []}
    eval_dict = {'score': [],
                 'max_block': []}
    for i in tqdm(range(n_rounds)):
        state_and_action, score, max_block = single_game()
        n_ins = len(state_and_action)
        score_list = [score] * n_ins
        data_dict['state_and_action'].extend(state_and_action)
        data_dict['score'].extend(score_list)

        eval_dict['score'].append(score)
        eval_dict['max_block'].append(max_block)
    return data_dict, eval_dict


class Game2048Dataset(data.Dataset):

    def __init__(self, data_dict, ):
        super(Game2048Dataset, self).__init__()

        self.state_and_action = data_dict['state_and_action']
        self.score = data_dict['score']
        self._len = len(self.state_and_action)

    def __getitem__(self, index):
        state, action = self.state_and_action[index]

        score = self.score[index]

        return (torch.from_numpy(state.copy().reshape(1, 4, 4)).float(),
                torch.tensor(action).long(),
                torch.tensor(score).float())

    def __len__(self):
        return self._len

BATCH_SIZE = 64
LOADER_WORKERS = 1

loader_params = {'batch_size': BATCH_SIZE, 'shuffle': True}
#dl = data.DataLoader(ds, **loader_params)


def train_model(dataloder, model, criterion, optimizer, num_epochs=1):
    since = time.time()
    use_gpu = torch.cuda.is_available()
    best_acc = 0.0
    dataset_sizes = {'train': len(dataloder.dataset)}

    for epoch in range(num_epochs):
        for phase in ['train']:
            if phase == 'train':
                model.train()

            running_loss = 0.0

            for state, action, score in tqdm(dataloder):
                if use_gpu:
                    state, action, score = Variable(state.cuda()), Variable(action.cuda()), Variable(score.cuda())
                else:
                    state, action, score = Variable(state), Variable(action), Variable(score)

                optimizer.zero_grad()

                v, p = model(state)
                # return outputs, labels, preds
                loss = criterion(v, score, p, action)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                running_loss += loss.data.item()

            if phase == 'train':
                train_epoch_loss = running_loss / dataset_sizes[phase]

        print('Epoch [{}/{}] '.format(
            epoch, num_epochs - 1))

    return model


N_ITER = 10

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

net = Net2048().to(device)
eval_dicts = []
criterion = Game2048Loss()
optimizer = torch.optim.Adam(net.parameters())

for i in range(N_ITER):
    print('Iter %d' % i)
    print('Getting data:')
    data_dict, eval_dict = get_data()
    eval_dicts.append(eval_dict)

    print('Eval: max score %d, median score %d, max block %d, median block %d' %
          (np.max([eval_dict['score'] for eval_dict in eval_dicts]),
           np.median([eval_dict['score'] for eval_dict in eval_dicts]),
           np.max([eval_dict['max_block'] for eval_dict in eval_dicts]),
           np.median([eval_dict['max_block'] for eval_dict in eval_dicts]))
          )

    print('Training:')
    ds = Game2048Dataset(data_dict)
    dl = data.DataLoader(ds, **loader_params)
    train_model(dl, net, criterion, optimizer)