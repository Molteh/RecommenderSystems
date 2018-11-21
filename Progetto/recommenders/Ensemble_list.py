import numpy as np
from Progetto.recommenders.Slim_BPR_Cython.SLIM_BPR_Cython import SLIM_BPR_Cython


class Ensemble_list(object):

    def __init__(self, u):
        self.u = u
        self.weights = 0
        self.S_CF_I = 0
        self.S_CF_U = 0
        self.S_CB = 0
        self.S_Slim = 0
        self.URM = 0

    def fit(self, URM, knn, shrink, weights, cython, epochs, lr, sgd_mode='rmsprop'):
        self.URM = URM
        self.weights = weights

        if weights[0] != 0:
            self.S_CF_I = self.u.get_itemsim_CF(self.URM, knn[0], shrink[0], cython)

        if weights[1] != 0:
            self.S_CF_U = self.u.get_usersim_CF(self.URM, knn[1], shrink[1], cython)

        if weights[2] != 0:
            self.S_CB = self.u.get_itemsim_CB(knn[2], shrink[2], cython)

        if weights[3] != 0:
            slim_BPR_Cython = SLIM_BPR_Cython(self.URM, recompile_cython=False, positive_threshold=0,
                                              sparse_weights=False)
            slim_BPR_Cython.fit(epochs=epochs, validate_every_N_epochs=1, batch_size=1, sgd_mode=sgd_mode,
                                learning_rate=lr,
                                topK=knn[3])
            self.S_Slim = slim_BPR_Cython.S

    def recommend(self, target_playlist):
        row_cb = 0
        row_cf_i = 0
        row_cf_u = 0
        row_slim = 0

        if self.weights[0] != 0:
            row_cf_i = (self.URM[target_playlist].dot(self.S_CF_I))

        if self.weights[1] != 0:
            row_cf_u = (self.S_CF_U[target_playlist].dot(self.URM))

        if self.weights[2] != 0:
            row_cb = (self.URM[target_playlist].dot(self.S_CB))

        if self.weights[3] != 0:
            row_slim = self.URM[target_playlist].dot(self.S_Slim).toarray().ravel()

        row_cf_i = row_cf_i * self.weights[0]
        row_cf_u = row_cf_u * self.weights[1]
        row_cb = row_cb * self.weights[2]

        row_item = (row_cf_i + row_cf_u + row_cb).toarray().ravel()

        top10_item = self.u.get_top_10(self.URM, target_playlist, row_item)
        top10_slim = self.u.get_top_10(self.URM, target_playlist, row_slim)

        r = []
        i = 0
        while True:
            if top10_item[i] not in r:
                r.append(top10_item[i])
                if len(r) >= 10:
                    break
            if top10_slim[i] not in r:
                r.append(top10_slim[i])
                if len(r) >= 10:
                    break
            i += 1
        return np.array(r)
