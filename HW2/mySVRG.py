#!/usr/bin/env python3

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import localtime, strftime

os.sys.path.append('SVM')
from svrg import SVRG


class ArgumentError(Exception):
    pass


def gen_results(X, y, m, numruns, loss=False):
    results = []
    for i in range(numruns):
        svrg = SVRG(M=m, calc_loss=loss)
        svrg.fit(X,y)
        results.append(svrg.fit_time)
        if loss:
            plt.figure()
            plt.xlabel('Iteration (every 10th saved)')
            plt.ylabel('Loss')
            plt.plot(svrg.training_loss)
            plt.savefig('plots/svrg/loss_vs_ite_{}.pdf'.format(
                strftime("%Y.%m.%d_%H.%M.%S", localtime()) +
                '__m_{}'.format(m),
                format='pdf'))
            plt.close('all')
        print('Run {} time: {}'.format(i, round(results[-1],3)))
    return results


def main(argv):
    if len(argv) == 3 or len(argv) == 4:
        filename, m, numruns = argv[0], int(argv[1]), int(argv[2])
        if len(argv) == 4:
            loss = True if argv[3] == 'loss' else False
        else:
            loss = False
    else:
        raise ArgumentError(
                'Please provide a filename in UNIX absolute path'
                ', m and the number of runs as well as "loss"'
                ' if Loss function plots are desired.')

    Data = np.genfromtxt(filename, delimiter=',')
    X = Data[:,1:]
    y = np.where(Data[:,0] == 1, -1, 1)

    print('\n-------------SVRG Runtimes---------------')
    results = gen_results(X, y, m, numruns, loss)
    print('-----------------')
    print('Avg: {}s, Std: {}s'.format(
        round(np.mean(results),3), round(np.std(results),3)))
    if loss:
        print('Loss plots saved to plots/svrg/')


if __name__ == '__main__':
    main(sys.argv[1:])

