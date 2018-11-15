from Progetto.recommenders.Slim_BPR_Cython.SLIM_BPR_Cython import SLIM_BPR_Cython


class Ensemble_pre(object):

    def __init__(self, u):
        self.u = u
        self.S_CF_U = 0
        self.URM = 0
        self.weights = 0
        self.S_item = 0

    def fit(self, URM, knn, shrink, weights, k, cython, epochs, sgd_mode='rmsprop', lr=0.1):
        self.URM = URM
        self.weights = weights
        S_CF_I = 0
        S_CB = 0
        S_SVD = 0
        S_Slim = 0

        if weights[0] != 0:
            S_CF_I = (self.u.get_itemsim_CF(self.URM, knn[0], shrink[0], cython)) * weights[0]

        if weights[1] != 0:
            self.S_CF_U = self.u.get_usersim_CF(self.URM, knn[1], shrink[1], cython)

        if weights[2] != 0:
            S_CB = (self.u.get_itemsim_CB(knn[2], shrink[2], cython)) * weights[2]

        if weights[3] != 0:
            S_SVD = (self.u.get_itemsim_SVD(self.URM, k, knn[3])) * weights[3]

        if weights[4] != 0:
            slim_BPR_Cython = SLIM_BPR_Cython(self.URM, recompile_cython=False, positive_threshold=0, sparse_weights=False)
            slim_BPR_Cython.fit(epochs=epochs, validate_every_N_epochs=1, batch_size=1, sgd_mode=sgd_mode, learning_rate=lr,
                                topK=knn[4])
            S_Slim = (slim_BPR_Cython.S) * weights[4]

        self.S_item = S_SVD + S_CF_I + S_CB + S_Slim

    def recommend(self, target_playlist):
        row_cf_u = 0

        row_item = self.URM[target_playlist].dot(self.S_item)

        if self.weights[1] != 0:
            row_cf_u = (self.S_CF_U[target_playlist].dot(self.URM)) * self.weights[1]

        row = (row_cf_u + row_item).toarray().ravel()

        return self.u.get_top_10(self.URM, target_playlist, row)
