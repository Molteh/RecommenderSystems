class Ensemble_cfcb(object):

    def __init__(self, u):
        self.u = u
        self.S_CB = None
        self.S_CF_I = None
        self.S_CF_U = None
        self.URM = None
        self.URM2 = None
        self.weights = None

    def fit(self, URM, knn, shrink, weights, cython):
        self.URM = URM
        self.weights = weights

        self.S_CF_I = self.u.get_itemsim_CF(self.URM, knn[0], shrink[0], cython)
        self.S_CF_U = self.u.get_usersim_CF(self.URM, knn[1], shrink[1], cython)
        self.S_CB = self.u.get_itemsim_CB(knn[2], shrink[2], cython)

    def recommend(self, target_playlist):
        row_cb = self.URM[target_playlist].dot(self.S_CB)
        row_cf_i = self.URM[target_playlist].dot(self.S_CF_I)
        row_cf_u = self.S_CF_U[target_playlist].dot(self.URM)

        row = ((self.weights[0] * row_cf_i) + (self.weights[1] * row_cf_u) + (self.weights[2] * row_cb)).toarray().ravel()

        return self.u.get_top_10(self.URM, target_playlist, row)
