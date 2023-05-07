import copy
import os
import random

import numpy as np

from ProblemParser import read, grid_parameters


class GridEnv:
    def __init__(self, grid_parameter):
        self.grid_parameter = grid_parameter
        self.grid_graph = self.generate_graph()
        # self.grid_capacity = self.generate_capacity()
        # self.x_cap, self.y_cap, self.layer_cap = self.generate_single_cap()

        self.init_state = None
        self.current_state = None
        self.goal_state = None

        self.current_step = 0
        self.max_step = 100

        self.twoPinNet_i = 0
        self.twoPinNetCombo = self.generate_two_pin_net()
        self.is_terminal = []

        self.route = []
        self.route_combo = []

        self.best_reward = 0
        self.best_route = []

        self.posTwoPinNum = 0
        self.episode = 0

        self.reward = 0
        self.inst_reward = 0
        self.instant_reward_combo = []

    def generate_graph(self):
        # if the grid is free, set it to 0
        grid_graph = np.zeros(shape=(self.grid_parameter['gridSize'][0] + 2,
                                     self.grid_parameter['gridSize'][1] + 2,
                                     self.grid_parameter['gridSize'][2] + 2))
        grid_graph[-1, :, :] = 1  # Set x+ direction boundary grid to 1
        grid_graph[0, :, :] = 1  # Set x- direction boundary grid to 1
        grid_graph[:, -1, :] = 1  # Set y+ direction boundary grid to 1
        grid_graph[:, 0, :] = 1  # Set y- direction boundary grid to 1
        grid_graph[:, :, -1] = 1  # Set z+ direction boundary grid to 1
        grid_graph[:, :, 0] = 1  # Set z- direction boundary grid to 1

        # if the grid is from an obstacle, set it to 1
        # TODO
        return grid_graph

    def generate_capacity(self):
        # capacity = [x,y,layer,direction]
        # direction: (x+,0) (x-,1) (y+,2) (y-,3) (z+,4) (z-,5)
        capacity = np.ones(shape=(self.grid_parameter['gridSize'][0], self.grid_parameter['gridSize'][1],
                                  self.grid_parameter['gridSize'][2], 6))
        capacity[-1, :, :, 0] = 0  # Set x+ direction boundary capacity to 0
        capacity[0, :, :, 1] = 0  # Set x- direction boundary capacity to 0
        capacity[:, -1, :, 2] = 0  # Set y+ direction boundary capacity to 0
        capacity[:, 0, :, 3] = 0  # Set y- direction boundary capacity to 0
        capacity[:, :, -1, 4] = 0  # Set z+ direction boundary capacity to 0
        capacity[:, :, 0, 5] = 0  # Set z- direction boundary capacity to 0

        # if the grid (x,y,layer) is from an obstacle ,set its capacity of all directions to 0
        # TODO
        return capacity

    def generate_single_cap(self):
        # capacity = [x_cap, y_cap, layer_cap]
        # x_cap = [x+1, y, layer]
        # y_cap = [x, y+1, layer]
        # layer_cap = [x, y, layer+1]
        x_cap = np.ones(shape=(self.grid_parameter['gridSize'][0] + 1, self.grid_parameter['gridSize'][1],
                               self.grid_parameter['gridSize'][2]))
        y_cap = np.ones(shape=(self.grid_parameter['gridSize'][0], self.grid_parameter['gridSize'][1] + 1,
                               self.grid_parameter['gridSize'][2]))
        layer_cap = np.ones(shape=(self.grid_parameter['gridSize'][0], self.grid_parameter['gridSize'][1],
                                   self.grid_parameter['gridSize'][2] + 1))
        x_cap[-1, :, :] = 0  # Set x+ direction boundary capacity to 0
        x_cap[0, :, :] = 0  # Set x- direction boundary capacity to 0
        y_cap[:, -1, :] = 0  # Set y+ direction boundary capacity to 0
        y_cap[:, 0, :] = 0  # Set y- direction boundary capacity to 0
        layer_cap[:, :, -1] = 0  # Set z+ direction boundary capacity to 0
        layer_cap[:, :, 0] = 0  # Set z- direction boundary capacity to 0

        # if the grid (x,y,layer) is from an obstacle ,set its capacity of all directions to 0
        # TODO
        return x_cap, y_cap, layer_cap

    def generate_two_pin_net(self):
        single_net_pins = []
        netlist = []
        net_order = []
        for i in range(self.grid_parameter['numNet']):
            for j in range(self.grid_parameter['netInfo'][i]['numPins']):
                single_net_pins.append(self.grid_parameter['netInfo'][i][str(j + 1)])
            netlist.append(single_net_pins)
            single_net_pins = []
            net_order.append(i)

        # sort the netlist with halfWireLength
        # TODO

        two_pin_net_nums = []
        two_pin_nets = []
        for i in range(self.grid_parameter['numNet']):
            two_pin_net_nums.append(len(netlist[i]) - 1)
            for j in range(len(netlist[i]) - 1):
                pin_start = netlist[i][j]
                pin_end = netlist[i][j + 1]
                two_pin_nets.append([pin_start, pin_end])

        # use MST to optimize two-pin nets
        # TODO

        return two_pin_nets

    def step(self, action):
        try:
            # action direction: (x+,0) (x-,1) (y+,2) (y-,3) (z+,4) (z-,5)
            # change the real cord to the gird_graph cord
            cur_x = self.current_state[0] + 1
            cur_y = self.current_state[1] + 1
            cur_layer = self.current_state[2] + 1

            if not self.grid_graph[cur_x][cur_y][cur_layer]:
                next_state = copy.deepcopy(self.current_state)
                reward = -1
                illegal_action = False
                if action == 0 and not self.grid_graph[cur_x + 1][cur_y][cur_layer]:
                    # x+ direction unobstructed
                    next_x = cur_x + 1
                    # change the gird_graph cord to the real cord
                    next_state[0] = next_x - 1
                    # update wire obstacle
                    # TODO
                    self.route.append(self.current_state[:3])
                elif action == 1 and not self.grid_graph[cur_x - 1][cur_y][cur_layer]:
                    # x- direction unobstructed
                    next_x = cur_x - 1
                    # change the gird_graph cord to the real cord
                    next_state[0] = next_x - 1
                    # update wire obstacle
                    # TODO
                    self.route.append(self.current_state[:3])
                elif action == 2 and not self.grid_graph[cur_x][cur_y + 1][cur_layer]:
                    # y+ direction unobstructed
                    next_y = cur_y + 1
                    # change the gird_graph cord to the real cord
                    next_state[1] = next_y - 1
                    # update wire obstacle
                    # TODO
                    self.route.append(self.current_state[:3])
                elif action == 3 and not self.grid_graph[cur_x][cur_y - 1][cur_layer]:
                    # y- direction unobstructed
                    next_y = cur_y - 1
                    # change the gird_graph cord to the real cord
                    next_state[1] = next_y - 1
                    # update wire obstacle
                    # TODO
                    self.route.append(self.current_state[:3])
                elif action == 4 and not self.grid_graph[cur_x][cur_y][cur_layer + 1]:
                    # z+ direction unobstructed
                    next_layer = cur_layer + 1
                    # change the gird_graph cord to the real cord
                    next_state[2] = next_layer - 1
                    # update wire obstacle
                    # TODO
                    self.route.append(self.current_state[:3])
                elif action == 5 and not self.grid_graph[cur_x][cur_y][cur_layer - 1]:
                    # z- direction unobstructed
                    next_layer = cur_layer - 1
                    # change the gird_graph cord to the real cord
                    next_state[2] = next_layer - 1
                    # update wire obstacle
                    # TODO
                    self.route.append(self.current_state[:3])
                else:
                    # The action is impracticable
                    illegal_action = True

                self.current_state = next_state
                self.current_step += 1
                done = False

                if self.current_state[:3] == self.goal_state[:3]:
                    # reach the goal
                    done = True
                    reward = 100.0
                    self.route.append(self.current_state[:3])
                    self.is_terminal.append(True)

                elif self.current_step >= self.max_step:
                    # reach the max step
                    done = True
                    self.is_terminal.append(False)

                self.reward += reward
                self.inst_reward += reward
                if done:
                    self.instant_reward_combo.append(self.inst_reward)
                    self.inst_reward = 0

                return next_state, reward, done, illegal_action
            else:
                raise RuntimeError('The state position is inside the obstacle!')
        except RuntimeError as e:
            print('Exception:', repr(e))

    def reset(self):
        reward_plot = [0, False]  # use to plot figure [reward, legal]
        is_best = False

        if self.twoPinNet_i >= len(self.twoPinNetCombo):
            # all the two pin nets are routed
            self.episode += 1

            reward_plot = [self.reward, True]

            self.twoPinNet_i = 0
            self.grid_graph = self.generate_graph()
            #
            self.route_combo.append(self.route)
            # pop the waste element out of the route_combo
            self.route_combo.pop(0)

            pos_two_pin_num = sum([1 for is_terminal in self.is_terminal if is_terminal])
            if pos_two_pin_num > self.posTwoPinNum:
                self.best_reward = self.reward
                self.best_route = self.route_combo
                is_best = True

                self.posTwoPinNum = pos_two_pin_num
            elif pos_two_pin_num == self.posTwoPinNum:
                if self.reward > self.best_reward:
                    self.best_reward = self.reward
                    self.best_route = self.route_combo
                    is_best = True

            self.is_terminal = []

            self.route_combo = []

            self.reward = 0
            self.inst_reward = 0
            self.instant_reward_combo = []

        # initialize state of gridEnv by the two pin net(self.twoPinNet_i)
        self.init_state = self.twoPinNetCombo[self.twoPinNet_i][0]
        self.goal_state = self.twoPinNetCombo[self.twoPinNet_i][1]
        self.current_state = self.init_state

        self.current_step = 0

        self.route_combo.append(self.route)
        self.route = []

        self.twoPinNet_i += 1

        return self.current_state, reward_plot, is_best

    def state2observ(self):
        state = np.array(self.current_state[:3])
        distance = np.array(self.goal_state[:3]) - state
        cur_x = state[0] + 1
        cur_y = state[1] + 1
        cur_z = state[2] + 1
        capacity = np.array([self.grid_graph[cur_x + 1, cur_y, cur_z],
                             self.grid_graph[cur_x - 1, cur_y, cur_z],
                             self.grid_graph[cur_x, cur_y + 1, cur_z],
                             self.grid_graph[cur_x, cur_y - 1, cur_z],
                             self.grid_graph[cur_x, cur_y, cur_z + 1],
                             self.grid_graph[cur_x, cur_y, cur_z - 1]
                             ])
        return np.concatenate((state, distance, capacity), axis=0)


if __name__ == '__main__':
    benchmark_dir = 'benchmark'
    for benchmark_file in os.listdir(benchmark_dir):
        benchmark_file = benchmark_dir + '/' + benchmark_file
        benchmark_info = read(benchmark_file)
        gridParameters = grid_parameters(benchmark_info)
        gridEnv = GridEnv(gridParameters)
        while gridEnv.episode <= 100:
            gridEnv.reset()
            done_ = False
            if gridEnv.episode == 100:
                break
            while not done_:
                action_ = random.randint(0, 5)
                print(gridEnv.state2observ())
                next_state_, reward_, done_, illegal_action_ = gridEnv.step(action_)
                print(gridEnv.state2observ())
        print(gridEnv)
