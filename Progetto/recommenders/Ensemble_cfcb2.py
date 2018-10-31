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

    def fit(self, URM, knn1, knn2, knn3, shrink, mode, normalize, weights):
        self.URM = URM
        self.weights = weights
        self.S_CF_item = self.u.get_itemsim_CF(self.URM, knn1, shrink, mode, normalize)
        self.S_user = self.u.get_usersim_CF(self.URM, knn2, shrink, mode, normalize)
        self.S_CB = self.u.get_itemsim_CB(knn3, shrink, mode, normalize)
        self.S_item = (weights[0] * self.S_CF_item) + ((1 - weights[0]) * self.S_CB)

    def recommend(self, target_playlist):
        row_user = self.S_user[target_playlist].dot(self.URM)
        row_item = self.URM[target_playlist].dot(self.S_item)
        row = ((self.weights[1] * row_item) + ((1 - self.weights[1]) * row_user)).toarray().ravel()
        return self.u.get_top_10(self.URM, target_playlist, row)

