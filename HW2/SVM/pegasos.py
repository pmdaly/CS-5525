#!/usr/bin/env python3

import numpy as np
from numpy.linalg import norm
import time

class Pegasos:

    def __init__(self, k=10, lam=1e-4, calc_loss=False):
        self.lam = lam
        self.k = k
        self.calc_loss = calc_loss

    def fit(self, X, y):
        self.T = 100*X.shape[0]
        self.fit_time = time.time()
        w = np.zeros(X.shape[1])
        if self.calc_loss:
            loss = [self._loss(X,y,w)]
        k_tot = 0
        for t in range(self.T):
            A_t_nzl = self._sample(X,y,w)
            nu_t = 1 / (self.lam*t) if t != 0 else 1 / self.lam
            w = (1-nu_t*self.lam)*w + nu_t/self.k * X[A_t_nzl].T.dot(y[A_t_nzl])
            w = min(1, self.lam**(-0.5)/(norm(w)+1e-5))*w
            k_tot += len(A_t_nzl)
            if self.calc_loss:
                if t%10 == 0:
                    loss.append(self._loss(X,y,w))
            if k_tot > self.T or abs(loss[-1] - loss[-2]) < 1e-4:
                break
        if self.calc_loss:
            self.training_loss = loss[1:]
        self.w = w
        self.fit_time = time.time() - self.fit_time

    def _sample(self, X, y, w):
        # TODO(pmdaly): can't figure out the correct sampling method
        #   this will have to do for now
        #
        #k_1 = self.k//2
        #A_t = np.random.choice(np.where(y==1)[0], k_1, replace=False)
        #A_t = np.append(A_t, np.random.choice(
        #    np.where(y==-1)[0], self.k - k_1, replace=False))
        #A_t.sort() # not sure if needed
        #A_t = np.random.choice(len(y), self.k, replace=False)
        y_0, y_1 = np.where(y==0)[0], np.where(y==1)[0]
        np.random.shuffle(y_0)
        np.random.shuffle(y_1)
        k_1 = self.k//2
        A_t = np.append(y_0[:k_1], y_1[:(self.k-k_1)])
        A_t_nzl = A_t[np.where(y[A_t]*X[A_t].dot(w) < 1)[0]]
        return A_t_nzl

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
    for k_v in [1,10,50,100,500,1000,2000]:
        pgs = Pegasos(k=k_v,calc_loss=True)
        pgs.fit(X,y)
        plt.figure()
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.plot(pgs.training_loss)
        plt.savefig('../plots/pegasos/loss_vs_ite_{}.pdf'.format(
            strftime("%Y.%m.%d_%H.%M.%S", localtime()) + '__K_{}'.format(k_v),
            format='pdf'))
        plt.close('all')
        print('Loss vs Iteration saved to ../plots/pegasos')
        print('Time: {} in {} iterations'.format(round(pgs.fit_time, 2),
            10*len(pgs.training_loss)))
        print('Final objective function val w/ K={}: {}\n'.format(
            pgs.k, round(pgs.training_loss[-1], 5)))


if __name__ == "__main__":
    main()
