import collections
import os
import random
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as f
from matplotlib import pyplot as plt

from GridEnvironment import GridEnv
from ProblemParser import read, grid_parameters


class QNetwork(torch.nn.Module):
    def __init__(self, state_dim, action_dim, architecture):
        # initial the parent class
        super(QNetwork, self).__init__()
        # initial the network architecture
        self.architecture = architecture
        self.dense1_block = torch.nn.Sequential(
            OrderedDict([
                ('dense1', torch.nn.Linear(state_dim, self.architecture[0])),
                ('relu1', torch.nn.ReLU())
            ])
        )
        self.dense2_block = torch.nn.Sequential(
            OrderedDict([
                ('dense2', torch.nn.Linear(self.architecture[0], self.architecture[1])),
                ('relu2', torch.nn.ReLU())
            ])
        )
        self.dense3_block = torch.nn.Sequential(
            OrderedDict([
                ('dense3', torch.nn.Linear(self.architecture[1], self.architecture[2])),
                ('relu3', torch.nn.ReLU()),
                ('output', torch.nn.Linear(self.architecture[2], action_dim))
            ])
        )

    def forward(self, state):
        dense1_out = self.dense1_block(state)
        dense2_out = self.dense2_block(dense1_out)
        dense3_out = self.dense3_block(dense2_out)

        return dense3_out


class ReplayMemory:
    def __init__(self, capacity):
        self.memory = collections.deque(maxlen=capacity)

    def add(self, observation, action, reward, next_observation, done):  # add the data into replay memory
        self.memory.append((observation, action, reward, next_observation, done))

    def sample(self, batch_size):  # sample (batch_size) pieces of data
        transitions = random.sample(self.memory, batch_size)
        observation, action, reward, next_observation, done = zip(*transitions)
        transition_dict = {
            'observations': np.array(observation),
            'actions': action,
            'rewards': reward,
            'next_observations': np.array(next_observation),
            'dones': done
        }
        return transition_dict

    def size(self):  # return the current length of the replay memory
        return len(self.memory)


class DQNAgent:
    def __init__(self, env, observ_dim=12, action_dim=6, network_architecture=None,
                 replay_cap=10000, batch_size=32,
                 gamma=0.95, epsilon=0.05, target_update=10, learning_rate=0.0001,
                 device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
                 ):
        if network_architecture is None:
            network_architecture = [32, 64, 32]

        self.env = env

        self.observ_dim = observ_dim  # deletable
        self.action_dim = action_dim
        self.network_architecture = network_architecture  # deletable

        # training qNet
        self.qNet = QNetwork(observ_dim, self.action_dim, self.network_architecture)
        # target qNet
        self.target_qNet = QNetwork(self.observ_dim, self.action_dim, self.network_architecture)

        # using Adam optimizer
        self.learning_rate = learning_rate  # deletable
        self.optimizer = torch.optim.Adam(self.qNet.parameters(), lr=self.learning_rate)

        self.replay_cap = replay_cap  # deletable
        self.batch_size = batch_size
        # replay memory
        self.replay = ReplayMemory(self.replay_cap)

        self.gamma = gamma  # discount factor
        self.epsilon = epsilon  # use in epsilon-Greedy
        self.target_update = target_update  # the update step of target qNet
        self.count = 0  # step counter

        self.device = device

    def take_action(self, observ):  # take action with epsilon-Greedy
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            observ = torch.tensor(observ, dtype=torch.float)
            action = self.qNet(observ).argmax().item()
        return action

    def update(self, batch_dict):
        observations = torch.tensor(batch_dict['observations'], dtype=torch.float)
        actions = torch.tensor(batch_dict['actions']).view(-1, 1)
        rewards = torch.tensor(batch_dict['rewards'], dtype=torch.float).view(-1, 1)
        next_observations = torch.tensor(batch_dict['next_observations'], dtype=torch.float)
        dones = torch.tensor(batch_dict['dones'], dtype=torch.float).view(-1, 1)

        q_values = self.qNet(observations).gather(1, actions)

        max_next_q_values = self.target_qNet(next_observations).max(1)[0].view(-1, 1)
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)

        dqn_loss = f.mse_loss(q_values, q_targets, reduction='mean')

        self.optimizer.zero_grad()  # set grad to zero
        dqn_loss.backward()  # backward to update parameters
        self.optimizer.step()

        if self.count % self.target_update == 0:
            # update the target qNet by training qNet
            self.target_qNet.load_state_dict(self.qNet.state_dict())

        self.count += 1

    def train(self, max_episode=2000):
        reward_plot_list = []
        while self.env.episode <= max_episode:
            state, reward_plot, is_best = self.env.reset()
            print(self.env.episode)
            done = False
            if self.env.episode == max_episode:
                break
            while not done:
                observ = self.env.state2observ()
                action = agent.take_action(observ)
                next_state, reward, done, illegal_action = self.env.step(action)
                next_observ = self.env.state2observ()
                agent.replay.add(observ, action, reward, next_observ, done)
                if agent.replay.size() > agent.batch_size*5:
                    batch_dict = agent.replay.sample(agent.batch_size)
                    agent.update(batch_dict)
            if reward_plot[1]:
                reward_plot_list.append(reward_plot[0])
        return reward_plot_list


if __name__ == '__main__':
    benchmark_dir = 'benchmark'
    max_episode_ = 10000

    benchmark_i = 0
    for benchmark_file in os.listdir(benchmark_dir):
        benchmark_file = benchmark_dir + '/' + benchmark_file
        benchmark_info = read(benchmark_file)
        gridParameters = grid_parameters(benchmark_info)
        gridEnv = GridEnv(gridParameters)
        agent = DQNAgent(gridEnv)
        reward_plot_combo = agent.train(max_episode_)
        print('pos_twoPinNets / twoPinNets : ' + str(agent.env.posTwoPinNum) + '/' + str(len(agent.env.twoPinNetCombo)))
        if agent.env.posTwoPinNum >= len(agent.env.twoPinNetCombo):
            # Plot reward and save reward data
            n = np.linspace(1, max_episode_, len(reward_plot_combo))
            plt.figure()
            plt.plot(n, reward_plot_combo)
            plt.xlabel('episodes')
            plt.ylabel('reward')
            plt.savefig('benchmark_{dumpBench}.DQNRewardPlot.jpg'.format(dumpBench=benchmark_i + 1))
            plt.close()

            plt.figure()
            best_route = agent.env.best_route
            start = []
            end = []
            for route in best_route:
                start.append(route[0])
                end.append(route[-1])
                for i in range(len(route)-1):
                    pair_x = [route[i][0], route[i+1][0]]
                    pair_y = [route[i][1], route[i+1][1]]
                    pair_z = [route[i][2], route[i+1][2]]
                    if pair_z[0] == pair_z[1]:
                        # same layer
                        plt.plot(pair_x, pair_y, color='blue', linewidth=2.5)
                    else:
                        # exchange layer
                        plt.plot(pair_x[0], pair_y[0], 'v', color='red')
            # draw the start & end
            start = np.array(start)
            end = np.array(end)
            plt.plot(start[:, 0], start[:, 1], 'o', color='blue')
            plt.plot(end[:, 0], end[:, 1], 'o', color='blue')
            plt.xlim([-0.1, gridParameters['gridSize'][0] - 0.9])
            plt.ylim([-0.1, gridParameters['gridSize'][1] - 0.9])
            plt.savefig('DQNRoutingVisualize_benchmark2d_{dumpBench}.png'.format(dumpBench=benchmark_i + 1))
            plt.close()

        benchmark_i += 1
