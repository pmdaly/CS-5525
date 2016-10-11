import numpy as np
import ipdb
import time

class SMO:

    def __init__(self, C=10, Tau=1e-5, max_ite = 1000, calc_loss=False):
        self.C = C
        self.Tau = Tau
        self.max_ite = max_ite
        self.calc_loss = calc_loss


    def fit(self, X, y):

        self.fit_time = time.time()
        alpha = np.zeros(X.shape[0])
        Q = X.dot(X.T)
        grad = np.empty(X.shape[0])

        i = j = ite = 0
        while (i != -1) and (j != -1) and (ite < self.max_ite):

            i,j  = self._wss(Q,y,grad,alpha)
            B = [i,j]
            N = [l for l in range(X.shape[0]) if l not in B]
            alpha_b, alpha_n = alpha[B], alpha[N]

            if self.calc_loss:
                loss = []

            if y[i] != y[j]:

                quad_coef = max(0, Q[i,i] + Q[j,j] + 2*Q[i,j])
                delta = -(grad[i]-grad[j])/quad_coef
                diff = alpha[i] - alpha[j]

                alpha[i] += delta
                alpha[j] += delta

                if alpha[i] < 0:
                    alpha[i] = 0
                else:
                    alpha[i] = min(alpha[i], self.C)
                if alpha[j] < 0:
                    alpha[j] = 0
                else:
                    alpha[j] = min(alpha[j], self.C)

            else:
                alpha[i] += y[i]*self._b(grad, y, i, j) / self._a(Q, i, j)

            for i in range(len(grad)):
                grad[i] += Q[i, B].dot(alpha[B] - alpha_b)

            if self.calc_loss:
                #if ite%10 == 0:
                loss.append(0.5*alpha.dot(Q).dot(alpha) + sum(alpha))
            ite += 1

        if self.calc_loss:
            self.training_loss = loss
        self.fit_time = time.time() - self.fit_time
        self.alpha = alpha


    def _wss(self, Q, y, grad, alpha):

        I_up, I_low = self._update_i_up_low(y, alpha)

        # search for i
        i, i_max_f = -1, -np.infty
        for i_up in I_up:
            yt_grad = -y[i_up]*grad[i_up]
            if yt_grad > i_max_f:
                i, i_max_f = i_up, yt_grad

        # search for j
        j, j_min_f = -1, np.infty
        for j_up in I_low:
            ba_ij = -self._b(grad,y,i,j_up)**2 / self._a(Q,i,j_up)
            if (ba_ij < j_min_f) and (-y[j]*grad[j] < i_max_f):
                j, j_min_f = j_up, ba_ij
        return i, j


    def _a(self, Q, t, s):
        return max(self.Tau, Q[t,t] + Q[s,s] - 2*Q[t,s])


    def _b(self, grad, y, t, s):
        return -y[t]*grad[t] + y[s]*grad[s]


    def _update_i_up_low(self, y, alpha):

        I_up  = [t for t,at in enumerate(alpha)
                if (at < self.C and y[t] ==  1)
                or (at > 0      and y[t] == -1)]

        I_low = [t for t,at in enumerate(alpha)
                if (at < self.C and y[t] == -1)
                or (at > 0      and y[t] ==  1)]

        return I_up, I_low


    def score(self, X, y):
        return


def main():
    # debugging
    #ipdb.set_trace()

    import matplotlib.pyplot as plt
    from time import localtime, strftime

    Data = np.genfromtxt('../data/MNIST-13.csv', delimiter=',')
    X, y = Data[:,1:], Data[:,0]
    y = np.array([-1 if i == 1 else 1 for i in y])

    smo = SMO(calc_loss=True)
    smo.fit(X,y)
    plt.figure()
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.plot(smo.training_loss)
    plt.savefig('../plots/smo/loss_vs_ite_{}.pdf'.format(
        strftime("%Y.%m.%d_%H.%M.%S", localtime()),
        format='pdf'))
    plt.close('all')
    print('Loss vs Iteration saved to ../plots/smo/')
    print('Time: {}'.format(smo.fit_time))

if __name__ == '__main__':
    main()
