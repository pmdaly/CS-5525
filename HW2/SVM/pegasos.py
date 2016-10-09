import numpy as np
from numpy.linalg import norm
import time

class SVM:

    def __init__(self, T=1000, k=10, calc_loss=False):
   # def __init__(self, **kwargs):
        self.lam = 1e-4
        self.T = T
        self.k = k
        self.calc_loss = calc_loss

    def fit(self, X, y):
        self.fit_time = time.time()
        w = np.zeros(X.shape[1])
        if self.calc_loss:
            loss = np.zeros(len(y))
        for t in range(self.T):
            A_t_nzl = self._sample(X,y)
            nu_t = 1 / (self.lam*t) if t != 0 else 1 / self.lam
            w = (1-nu_t*self.lam)*w + nu_t/self.k * X[A_t_nzl].T.dot(y[A_t_nzl])
            w = min(1, 1/np.sqrt(self.T)/norm(w))*w
            if self.calc_loss:
                loss[t] = self._loss(X,y,w)
        if self.calc_loss:
            self.training_loss = loss
        self.w = w
        self.fit_time = time.time() - self.fit_time

    def _sample(self, X, y):
        k_1 = self.k//2
        k_n1 = self.k - k_1
        A_t = np.random.choice(np.where(y==1), k_1, replace=False)
        A_t.append(np.random.choice(np.where(y==-1), k_n1, replace=False))
        A_t_nzl = np.where(y[A_t]*X[A_t].dot(w) < 1)[0]
        return A_t_nzl

    def _loss(self, X, y, w):
        w_penalty = self.lam / 2 * norm(w)**2
        loss = np.mean(np.maximum(np.zeros(len(y)),1 - y*X.dot(w)))
        return w_penalty + loss

    def score(self, X, y):
        y_pred_rv = X.dot(self.w)
        y_pred = np.array([-1 if i < 0 else 1 for i in y_pred_rv])
        return np.mean(y_pred == y)

def main():
    Data = np.genfromtxt('../data/MNIST-13.csv', delimiter=',')
    X, y = Data[:,1:], Data[:,0]
    y = np.array([-1 if i == 1 else 1 for i in y])

    svm = SVM(calc_loss=True)
    svm.fit(X,y)

    print('Time: {}'.format(svm.fit_time))

if __name__ == "__main__":
    main()
