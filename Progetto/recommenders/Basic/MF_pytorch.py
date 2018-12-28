from Progetto.recommenders.MF.pytorch.mf_mse import MF_MSE_PyTorch


class MF_BPR(object):

    def __init__(self, u):
        self.u = u
        self.URM = 0
        self.W = None
        self.H = None

    def fit(self, URM, k, epochs, lr):
        self.URM = URM

        matrixFactorization = MF_MSE_PyTorch(self.URM, positive_threshold=0)
        matrixFactorization.fit(epochs=epochs, num_factors=k, learning_rate=lr, use_cuda=False)
        self.W = matrixFactorization.W_best
        self.H = matrixFactorization.H_best.T

    def recommend(self, target_playlist):
        row = self.W[target_playlist].dot(self.H)
        return self.u.get_top_10(self.URM, target_playlist, row)