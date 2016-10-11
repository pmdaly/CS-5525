import numpy as np
import time

class SMO:

    def __init__(self, C=10, Tau=1e-5):
        self.C = C
        self.Tau = Taur

    def fit(self, X, y):
        alpha = np.zeros(X.shape[1])
        Q = X.T.dot(X)
        grad =
        while alpha is not stationary?:
            B = self._wss(Q,y,grad)

    def _wss(self, Q, y, grad):
        I_up, I_low = self._update_i_up_low(y)
        # search for i
        i, i_max_f = -1, np.infty
        for i_up in I_up:
            yt_grad = -y[i_up]*grad[i_up]
            if yt_grad > i_max_f:
                i, i_max_f = i_up, yt_grad
        # search for j
        j, j_min_f = -1, -np.infty
        for j_up in I_low:
            ba_ij = -self._b(Q,i,j_up)**2 / self._a(Q,i,j_up)
            if ba_ij < j_min_f and -y[j]*grad[j] < i_max_f:
                j, j_min_f = j_up, ba_ij
        return i, j

    def _a(self, Q, t, s):
        a_ts = Q[t,t] + Q[s,s] - 2*Q[t,s]
        return a_ts if a_ts > 0 else self.Tau

    def _b(self, grad, t, s):
        return -y[t]*grad[t] + y[s]*grad[s]

    def _update_i_up_low(self, y):
        I_up  = [t for t,yt in enumerate(y)
                if (self.alpha[t] < self.C and yt ==  1)
                or (self.alpha[t] > 0      and yt == -1)]
        I_low = [t for t,yt in enumerate(y)
                if (self.alpha[t] < self.C and yt == -1)
                or (self.alpha[t] > 0      and yt ==  1)]
        return I_up, I_low

    def score(self, X, y):
        return


def main():
    import matplotlib.pyplot as plt
    from time import localtime, strftime

    Data = np.genfromtxt('../data/MNIST-13.csv', delimiter=',')
    X, y = Data[:,1:], Data[:,0]
    y = np.array([-1 if i == 1 else 1 for i in y])

    smo = SMO()
    smo.fit(X,y)
    plt.figure()
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.plot(smo.training_loss)
    plt.savefig('../plots/loss_vs_ite_{}.pdf'.format(
        strftime("%Y.%m.%d_%H.%M.%S", localtime()),
        format='pdf'))
    plt.close('all')
    print('Loss vs Iteration saved to ../plots')
    print('Time: {}'.format(smo.fit_time))

if __name__ == '__main__':
    main()
