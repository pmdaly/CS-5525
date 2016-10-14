#!/usr/bin/env python3

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import localtime, strftime

os.sys.path.append('SVM')
from smo import SMO


class ArgumentError(Exception):
    pass


def gen_results(X, y, numruns, loss=False):
    results = []
    for i in range(numruns):
        smo = SMO(calc_loss=loss)
        smo.fit(X,y)
        results.append(smo.fit_time)
        if loss:
            plt.figure()
            plt.xlabel('Iteration (every 10th saved)')
            plt.ylabel('Loss')
            plt.plot(smo.training_loss)
            plt.savefig('plots/smo/loss_vs_ite_{}.pdf'.format(
                strftime("%Y.%m.%d_%H.%M.%S", localtime()),
                format='pdf'))
            plt.close('all')
        print('Run {} time: {}'.format(i, round(results[-1],3)))
    return results


def main(argv):
    if len(argv) == 2 or len(argv) == 3:
        filename, numruns = argv[0], int(argv[1])
        if len(argv) == 3:
            loss = True if argv[2] == 'loss' else False
        else:
            loss = False
    else:
        raise ArgumentError(
                'Please provide a filename in UNIX absolute path'
                ' and the number of runs as well as "loss"'
                ' if Loss function plots are desired.')

    Data = np.genfromtxt(filename, delimiter=',')
    X = Data[:,1:]
    y = np.where(Data[:,0] == 1, -1, 1)

    print('\n------------SMO Runtimes---------------')
    results = gen_results(X, y, numruns, loss)
    print('-----------------')
    print('Avg: {}s, Std: {}s'.format(
        round(np.mean(results),3), round(np.std(results),3)))
    if loss:
        print('Loss plots saved to plots/smo/')


if __name__ == '__main__':
    main(sys.argv[1:])

