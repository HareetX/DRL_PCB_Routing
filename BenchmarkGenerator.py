import argparse
import os
import shutil

import numpy as np

np.random.seed(1)


def generator(benchmark_name, grid_size, layer_num, net_num, max_pin_num):
    file = open('%s' % benchmark_name, 'w+')

    # Write general information
    file.write('grid {gridSize} {gridSize} {layerNum}\n'.format(gridSize=grid_size, layerNum=layer_num))
    file.write('net num {netNum}\n'.format(netNum=net_num))

    # Write nets information
    pin_num = np.random.randint(2, max_pin_num + 1, net_num)  # Generate PinNum randomly
    for j in range(net_num):
        specific_pin_num = pin_num[j]
        file.write('A{netInd} {netInd} {pin}\n'.format(netInd=j + 1, pin=specific_pin_num))
        x_array = np.random.randint(0, grid_size, specific_pin_num)
        y_array = np.random.randint(0, grid_size, specific_pin_num)
        z_array = np.random.randint(0, layer_num, specific_pin_num)
        for j in range(specific_pin_num):
            file.write('{x}  {y}  {z}\n'.format(x=x_array[j], y=y_array[j], z=z_array[j]))

    # Write obstacles information
    # TODO
    file.close()
    return


def parse_arguments():
    parser = argparse.ArgumentParser('Benchmark Generator Parser')
    parser.add_argument('--benchNumber', type=int, dest='benchmarkNumber', default=5)
    parser.add_argument('--gridSize', type=int, dest='gridSize', default=128)
    parser.add_argument('--layerNum', type=int, dest='layerNum', default=2)
    parser.add_argument('--netNum', type=int, dest='netNum', default=3)
    parser.add_argument('--maxPinNum', type=int, dest='maxPinNum', default=2)

    return parser.parse_args()


if __name__ == '__main__':
    benchmark_dir = 'benchmark'
    if os.path.exists(benchmark_dir):
        shutil.rmtree(benchmark_dir)

    os.mkdir(benchmark_dir)

    args = parse_arguments()

    for i in range(args.benchmarkNumber):
        filename = benchmark_dir + '/benchmark{}.gr'.format(i + 1)
        generator(filename, args.gridSize, args.layerNum, args.netNum, args.maxPinNum)
