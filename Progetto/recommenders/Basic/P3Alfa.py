from Progetto.recommenders.GraphBased.P3Alpha import P3alphaRecommender


class P3Alfa_R(object):

    def __init__(self, u):
        self.u = u
        self.URM = None
        self.S = None

    def fit(self, URM, knn=60, alfa=0.7, norm=True, implicit=True, min_rating=0):
        self.URM = URM

        p3 = P3alphaRecommender(URM)
        p3.fit(topK=knn, alpha=alfa, normalize_similarity=norm, implicit=implicit, min_rating=min_rating)
        self.S = p3.W_sparse

    def recommend(self, target_playlist):
        row = self.URM[target_playlist].dot(self.S)
        return self.u.get_top_10(self.URM, target_playlist, row.toarray().ravel())
