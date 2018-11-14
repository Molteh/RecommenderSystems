class Ensemble_cfcbsvd(object):

    def __init__(self, u):
        self.u = u
        self.S_CB = None
        self.S_CF_I = None
        self.S_CF_U = None
        self.S_SVD = None
        self.URM = None
        self.weights = None

    def fit(self, URM, knn1, knn2, knn3, knn4, shrink, weights, k, cython):
        self.URM = URM
        self.weights = weights
        self.S_SVD = self.u.get_itemsim_SVD(self.URM, k, knn4)
        self.S_CF_I = self.u.get_itemsim_CF(self.URM, knn1, shrink[0], cython)
        self.S_CF_U = self.u.get_usersim_CF(self.URM, knn2, shrink[1], cython)
        self.S_CB = self.u.get_itemsim_CB(knn3, shrink[2], cython)

    def recommend(self, target_playlist):
        row_cb = self.URM[target_playlist].dot(self.S_CB)
        row_cf_i = self.URM[target_playlist].dot(self.S_CF_I)
        row_cf_u = self.S_CF_U[target_playlist].dot(self.URM)
        row_svd = self.URM[target_playlist].dot(self.S_SVD)
        row = ((self.weights[0] * row_cf_i) + (self.weights[1] * row_cf_u) + (self.weights[2] * row_cb) + (
                    self.weights[3] * row_svd)).toarray().ravel()
        return self.u.get_top_10(self.URM, target_playlist, row)
