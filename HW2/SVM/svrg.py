#!/usr/bin/env python3

import numpy as np
from numpy.linalg import norm
import time

# debugging
import ipdb
#ipdb.set_trace()

class SVRG:

    def __init__(self, M=500, nu=1e-4, lam=1e-4, calc_loss=False):
        self.M = M
        self.nu = nu
        self.lam = lam
        self.calc_loss = calc_loss

    def fit(self, X, y):

        self.fit_time = time.time()

        w = w_bar = np.zeros(X.shape[1])
        if self.calc_loss:
            loss = []

        k_tot = 0
        for s in range(X.shape[0]):

            mu = self._grad(X,y,w)
            k_tot += X.shape[1]
            w = w_bar

            for t in range(self.M):
                i = np.random.choice(X.shape[0])
                grad_i = self._grad(X[i],y[i],w)
                grad_bar = self._grad(X[i],y[i],w_bar)
                w += -self.nu * (grad_i - grad_bar + mu)

            w_bar = w

            k_tot += 1
            if self.calc_loss:
                if k_tot%10 == 0:
                    loss.append(self._loss(X,y,w))
            if k_tot > 100*X.shape[0]:
                break
        if self.calc_loss:
            self.training_loss = loss
        self.w = w
        self.fit_time = time.time() - self.fit_time

    def _grad(self, X, y, w):

        if X.ndim == 1:
            grad = -y*X.T if y*X.dot(w) < 1 else np.zeros(len(w))
            return self.lam*w + grad
        else:
            yXw = y*X.dot(w)
            yX = X*y[:,np.newaxis]
            grad = [-yX[i] if yXw[i] < 1 else 0 for i in range(len(y))]
            return self.lam*w + 1/X.shape[0]*sum(grad).T


    def _loss(self, X, y, w):
        w_penalty = 0.5 * self.lam * norm(w)**2
        loss = np.mean(np.maximum(np.zeros(len(y)),1 - y*X.dot(w)))
        return w_penalty + loss

    def score(self, X, y):
        y_pred_rv = X.dot(self.w)
        y_pred = np.array([-1 if i < 0 else 1 for i in y_pred_rv])
        return np.mean(y_pred == y)

def main():
    import matplotlib.pyplot as plt
    from time import localtime, strftime

    Data = np.genfromtxt('../data/MNIST-13.csv', delimiter=',')
    X, y = Data[:,1:], Data[:,0]
    y = np.array([-1 if i == 1 else 1 for i in y])

    k_v = 50
    svrg = SVRG(calc_loss=True)
    svrg.fit(X,y)
    plt.figure()
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.plot(svrg.training_loss)
    plt.savefig('../plots/svrg/loss_vs_ite_{}.pdf'.format(
        strftime("%Y.%m.%d_%H.%M.%S", localtime()),
        format='pdf'))
    plt.close('all')
    print('Loss vs Iteration saved to ../plots/svrg')
    print('Time: {}'.format(round(svrg.fit_time,2)))
    print('Final objective function val: {}'.format(
        round(svrg.training_loss[-1], 2)))

if __name__ == "__main__":
    main()
