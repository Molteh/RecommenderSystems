class ItemSVD(object):

    def __init__(self, u):
        self.S = None
        self.URM = None
        self.u = u

    def fit(self, URM, k, knn, tfidf):
        self.URM = URM

        self.S = self.u.get_itemsim_SVD(k, knn, tfidf)

    def recommend(self, target_playlist):
        row = self.URM[target_playlist].dot(self.S)
        return self.u.get_top_10(self.URM, target_playlist, row.toarray().ravel())

