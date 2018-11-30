class User_CFR(object):

    def __init__(self, u):
        self.u = u
        self.URM = 0
        self.S_CF_U = 0

    def fit(self, URM, knn, shrink, normalize, similarity, tfidf):
        self.URM = URM
        self.S_CF_U = self.u.get_usersim_CF(self.URM, knn, shrink, normalize, similarity, tfidf)

    def recommend(self, target_playlist):
        row_cf_u = (self.S_CF_U[target_playlist].dot(self.URM)).toarray().ravel()
        return self.u.get_top_10(self.URM, target_playlist, row_cf_u)
