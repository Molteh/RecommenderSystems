class Item_CFR(object):

    def __init__(self, u):
        self.u = u
        self.URM = 0
        self.S_CF_I = 0

    def fit(self, URM, knn, shrink, normalize, similarity, tfidf):
        self.URM = URM
        self.S_CF_I = self.u.get_itemsim_CF(self.URM, knn, shrink, normalize, similarity, tfidf)

    def recommend(self, target_playlist):
        row_cf_i = (self.URM[target_playlist].dot(self.S_CF_I)).toarray().ravel()
        return self.u.get_top_10(self.URM, target_playlist, row_cf_i)
