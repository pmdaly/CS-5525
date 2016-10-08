import numpy as np

class SVM:

    def __init__(self, kernel='linear', C=1, epsilon=1, tol=1e-5, max_iter=500):
        self.kernel = get_kernel(kernel)
        self.C = C
        self.epsilon = epsilon
        self.toll = tol
        self.max_iter = max_iter

    def get_kernel(self, kernel):
        if kernel == 'linear':
            def linear(x,y):
                return x.dot(y)
            return linear
        else:
            def rbf(x,y, sigma=1):
                return exp(-np.linalg.norm(x-y)*2 / (2*sigma**2))
            return rbf


    def train(self, X, y):

    def score(self, X, y):
