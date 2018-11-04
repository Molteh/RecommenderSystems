class User_CFR(object):

    def __init__(self, u):
        self.u = u
        self.URM = None
        self.S = None

    def fit(self, URM, knn, shrink):
        self.URM = URM
        self.S = self.u.get_usersim_CF(self.URM, knn, shrink)

    def recommend(self, target_playlist):
        row = self.S[target_playlist].dot(self.URM).toarray().ravel()
        return self.u.get_top_10(self.URM, target_playlist, row)
