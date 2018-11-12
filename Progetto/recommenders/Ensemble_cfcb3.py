import numpy as np

class Ensemble_cfcb3(object):

    def __init__(self, u):
        self.u = u
        self.S_CB = None
        self.S_CF_I = None
        self.S_CF_U = None
        self.URM = None
        self.weights = None

    def fit(self, URM, knn1, knn2, knn3, shrink, weights, weight, cython):
        self.URM = URM
        self.weights = weights
        self.S_CF_I = self.u.get_itemsim_CF(self.URM, knn1, shrink, cython)
        self.S_CF_U = self.u.get_usersim_CF(self.URM, knn2, shrink, cython)
        self.S_CB = self.u.get_itemsim_CB(knn3, shrink, weight, cython)

    def recommend(self, target_playlist):
        row_cb = (self.URM[target_playlist].dot(self.S_CB))
        row_cf_i = (self.URM[target_playlist].dot(self.S_CF_I))
        row_cf_u = (self.S_CF_U[target_playlist].dot(self.URM))
        top10_cb = self.u.get_top_10(self.URM, target_playlist, row_cb.toarray().ravel())
        top10_cf_i = self.u.get_top_10(self.URM, target_playlist, row_cf_i.toarray().ravel())
        top10_cf_u = self.u.get_top_10(self.URM, target_playlist, row_cf_u.toarray().ravel())
        r1 = np.intersect1d(top10_cb, top10_cf_i)
        r2 = np.intersect1d(r1, top10_cf_u)
        row = ((self.weights[0] * row_cf_i) + (self.weights[1] * row_cf_u) + (self.weights[2] * row_cb))
        r = []
        r = np.append(r, r2[np.argsort(row[0,r2].todense()).tolist()[0][::-1]])
        top10 = self.u.get_top(self.URM, target_playlist, row)
        top = top10[~np.in1d(top10,r2)]
        r = np.append(r, top[:10-len(r2)]).astype(int)
        return r
