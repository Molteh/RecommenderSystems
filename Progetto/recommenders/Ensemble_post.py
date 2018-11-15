from Progetto.recommenders.Slim_BPR_Cython.SLIM_BPR_Cython import SLIM_BPR_Cython


class Ensemble_post(object):

    def __init__(self, u):
        self.u = u
        self.S_CB = 0
        self.S_CF_I = 0
        self.S_CF_U = 0
        self.S_SVD = 0
        self.S_Slim = 0
        self.URM = 0
        self.weights = 0

    def fit(self, URM, knn, shrink, weights, k, cython, epochs, sgd_mode='rmsprop', lr=0.1):
        self.URM = URM
        self.weights = weights

        if weights[0] != 0:
            self.S_CF_I = self.u.get_itemsim_CF(self.URM, knn[0], shrink[0], cython)

        if weights[1] != 0:
            self.S_CF_U = self.u.get_usersim_CF(self.URM, knn[1], shrink[1], cython)

        if weights[2] != 0:
            self.S_CB = self.u.get_itemsim_CB(knn[2], shrink[2], cython)

        if weights[3] != 0:
            self.S_SVD = self.u.get_itemsim_SVD(self.URM, knn[3], k)

        if weights[4] != 0:
            slim_BPR_Cython = SLIM_BPR_Cython(self.URM, recompile_cython=False, positive_threshold=0, sparse_weights=False)
            slim_BPR_Cython.fit(epochs=epochs, validate_every_N_epochs=1, batch_size=1, sgd_mode=sgd_mode, learning_rate=lr,
                                topK=knn[4])
            self.S_Slim = slim_BPR_Cython.S

    def recommend(self, target_playlist):
        row_cb = 0
        row_cf_i = 0
        row_cf_u = 0
        row_svd = 0
        row_slim = 0

        if self.weights[0] != 0:
            row_cf_i = (self.URM[target_playlist].dot(self.S_CF_I)) * self.weights[0]

        if self.weights[1] != 0:
            row_cf_u = (self.S_CF_U[target_playlist].dot(self.URM)) * self.weights[1]

        if self.weights[2] != 0:
            row_cb = (self.URM[target_playlist].dot(self.S_CB)) * self.weights[2]

        if self.weights[3] != 0:
            row_svd = (self.URM[target_playlist].dot(self.S_SVD)) * self.weights[3]

        if self.weights[4] != 0:
            row_slim = (self.URM[target_playlist].dot(self.S_Slim)) * self.weights[4]

        row = (row_cf_i + row_cf_u + row_cb + row_svd + row_slim).toarray().ravel()

        return self.u.get_top_10(self.URM, target_playlist, row)
