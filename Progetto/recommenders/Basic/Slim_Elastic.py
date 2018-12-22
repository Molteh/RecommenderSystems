from Progetto.recommenders.SlimBPR.Slim_ElasticNet import SLIMElasticNetRecommender


class Slim_Elastic(object):

    def __init__(self, u):
        self.u = u
        self.URM = 0
        self.S_Slim = 0


    def fit(self, URM, knn, l1, po):
        self.URM = URM

        slim_Elastic = SLIMElasticNetRecommender(self.URM)
        slim_Elastic.fit(topK=knn, l1_ratio=l1, positive_only=po)
        self.S_Slim = slim_Elastic.W_sparse


    def recommend(self, target_playlist):
        row_slim = (self.URM[target_playlist].dot(self.S_Slim)).toarray().ravel()
        return self.u.get_top_10(self.URM, target_playlist, row_slim)
