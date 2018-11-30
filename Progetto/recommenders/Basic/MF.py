from Progetto.recommenders.MatrixFactorization.MatrixFactorization_Cython import MatrixFactorization_Cython


class MF_BPR(object):

    def __init__(self, u):
        self.u = u
        self.URM = 0
        self.W = None
        self.H = None

    def fit(self, URM, k, epochs, sgd_mode, lr):
        self.URM = URM

        matrixFactorization = MatrixFactorization_Cython(self.URM, positive_threshold=0, algorithm='MF_BPR')
        matrixFactorization.fit(epochs=epochs, num_factors=k, learning_rate=lr, sgd_mode=sgd_mode)
        self.W = matrixFactorization.W_best
        self.H = matrixFactorization.H_best.T

    def recommend(self, target_playlist):
        row = self.W[target_playlist].dot(self.H)
        return self.u.get_top_10(self.URM, target_playlist, row)
