from Progetto.utils.BPR_utils import SlimBPR_utils
from Progetto.recommenders.Slim_BPR_Cython.SLIM_BPR_Cython import SLIM_BPR_Cython


class Ensemble_cfcb_sbpr(object):

    def __init__(self, u):
        self.u = u
        self.S_CB = 0
        self.S_CF_I = 0
        self.S_CF_U = 0
        self.S = 0
        self.URM = None
        self.weights = None

    def fit(self, URM, knn1, knn2, knn3, knn4, shrink, weights, cython, epochs=2, sgd_mode='rmsprop', lr=1e-4):
        self.URM = URM
        self.weights = weights

        if weights[3] != 0:
            if cython:
                slim_BPR_Cython = SLIM_BPR_Cython(self.URM, recompile_cython=False, positive_threshold=0, sparse_weights=True)
                slim_BPR_Cython.fit(epochs=epochs, validate_every_N_epochs=1, batch_size=1, sgd_mode=sgd_mode, learning_rate=lr)
                self.S = slim_BPR_Cython.S
            else:
                BPR_gen = SlimBPR_utils(self.URM)
                self.S = BPR_gen.get_S_SLIM_BPR(knn4)

        if weights[0] != 0:
            self.S_CF_I = self.u.get_itemsim_CF(self.URM, knn1, shrink, cython)
        if weights[1] != 0:
            self.S_CF_U = self.u.get_usersim_CF(self.URM, knn2, shrink, cython)
        if weights[2] != 0:
            self.S_CB = self.u.get_itemsim_CB(knn3, shrink, cython)

    def recommend(self, target_playlist):
            row_R_CB = self.URM[target_playlist].dot(self.S_CB)
            row_R_CF_I = self.URM[target_playlist].dot(self.S_CF_I)
            row_R_CF_U = self.S_CF_U[target_playlist].dot(self.URM)
            row_R_Slim_BPR = self.URM[target_playlist].dot(self.S)
            row = ((self.weights[0]*row_R_CF_I) + (self.weights[1] * row_R_CF_U) + (self.weights[2]*row_R_CB) + (
                        self.weights[3] * row_R_Slim_BPR)).toarray().ravel()

            return self.u.get_top_10(self.URM, target_playlist, row)
