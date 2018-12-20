from Progetto.recommenders.GraphBased.RP3Beta import RP3betaRecommender

class P3Beta_R(object):

    def __init__(self, u):
        self.u = u
        self.URM = 0
        self.S = None

    def fit(self, URM, knn=100, alfa=0.7, beta=0.3, norm=True, implicit=True, min_rating=0):
        self.URM = URM

        p3 = RP3betaRecommender(URM)
        p3.fit(topK=knn, alpha=alfa, beta=beta, normalize_similarity=norm, implicit=implicit, min_rating=min_rating)
        self.S = p3.W_sparse

    def recommend(self, target_playlist):
        row = self.URM[target_playlist].dot(self.S)
        return self.u.get_top_10(self.URM, target_playlist, row.toarray().ravel())
