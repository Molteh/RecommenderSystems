class Ensemble_cfcb2(object):

    def __init__(self, u):
        self.u = u
        self.S_CB = None
        self.S_CF_item = None
        self.S_CF_user = None
        self.S_user = None
        self.S_item = None
        self.URM = None
        self.weights = None

    def fit(self, URM, knn1, knn2, knn3, shrink, weights, cython):
        self.URM = URM
        self.weights = weights
        self.S_CF_item = self.u.get_itemsim_CF(self.URM, knn1, shrink[0], cython)
        self.S_user = self.u.get_usersim_CF(self.URM, knn2, shrink[1], cython)
        self.S_CB = self.u.get_itemsim_CB(knn3, shrink[2], cython)
        self.S_item = (weights[0] * self.S_CF_item) + (self.S_CB * weights[2])

    def recommend(self, target_playlist):
        row_user = self.S_user[target_playlist].dot(self.URM)
        row_item = self.URM[target_playlist].dot(self.S_item)
        row = (row_item + (self.weights[1] * row_user)).toarray().ravel()
        return self.u.get_top_10(self.URM, target_playlist, row)
