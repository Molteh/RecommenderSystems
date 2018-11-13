class Ensemble_cfcb4(object):

    def __init__(self, u):
        self.u = u
        self.S_CB = None
        self.S_CF_I = None
        self.S_CF_U = None
        self.URM = None
        self.weights = None
        self.mean_CB = None
        self.mean_CF_I = None
        self.mean_CF_U = None

    def fit(self, URM, knn1, knn2, knn3, shrink, weight, alfa, cython):
        self.URM = URM
        self.weights = weight
        self.S_CF_I = self.u.get_itemsim_CF(self.URM, knn1, shrink, cython)
        self.S_CF_U = self.u.get_usersim_CF(self.URM, knn2, shrink, cython)
        self.S_CB = self.u.get_itemsim_CB(knn3, shrink, alfa, cython)
        self.mean_CF_I = (self.URM * self.S_CF_I).data.mean()
        self.mean_CF_U = (self.S_CF_U * self.URM).data.mean()
        self.mean_CB = (self.URM * self.S_CB).data.mean()

    def recommend(self, target_playlist):
        row_cb = self.URM[target_playlist].dot(self.S_CB)
        row_cf_i = self.URM[target_playlist].dot(self.S_CF_I)
        row_cf_u = self.S_CF_U[target_playlist].dot(self.URM)
        row_cb = row_cb / (row_cb.data.mean()/self.mean_CB)
        row_cf_i = row_cf_i / (row_cf_i.data.mean()/self.mean_CF_I)
        row_cf_u = row_cf_u / (row_cf_u.data.mean()/self.mean_CF_U)
        row = ((self.weights[0]*row_cf_i) + (self.weights[1] * row_cf_u) + (self.weights[2] * row_cb)).toarray().ravel()
        return self.u.get_top_10(self.URM, target_playlist, row)
