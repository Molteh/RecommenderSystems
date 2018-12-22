from Progetto.recommenders.SlimBPR.SLIM_BPR_Cython import SLIM_BPR_Cython
from Progetto.recommenders.GraphBased.RP3Beta import RP3betaRecommender
from Progetto.recommenders.SlimBPR.Slim_ElasticNet import SLIMElasticNetRecommender
from sklearn.utils.extmath import randomized_svd
from functools import reduce
import scipy.sparse as sp


class Ensemble_post(object):

    def __init__(self, u):
        self.u = u
        self.S_CB = 0
        self.S_CF_I = 0
        self.S_CF_U = 0
        self.U = 0
        self.s_Vt = 0
        self.S_Slim = 0
        self.S_Elastic = 0
        self.S_P3 = 0
        self.URM = 0
        self.weights = 0

    def fit(self, URM, knn, shrink, weights, epochs, tfidf, n_iter):
        self.URM = URM
        self.weights = weights

        if weights[0] != 0:
            self.S_CF_I = self.u.get_itemsim_CF(self.URM, knn[0], shrink[0], True, tfidf=tfidf)

        if weights[1] != 0:
            self.S_CF_U = self.u.get_usersim_CF(self.URM, knn[1], shrink[1], True, tfidf=tfidf)

        if weights[2] != 0:
            self.S_CB = self.u.get_itemsim_CB(knn[2], shrink[2])

        if weights[3] != 0:
            self.U, Sigma, VT = randomized_svd(self.u.okapi_BM_25(self.URM), n_components=800, n_iter=1, random_state=False)
            self.s_Vt = sp.diags(Sigma) * VT

        if weights[4] != 0:
            slim_BPR_Cython = SLIM_BPR_Cython(self.URM)
            total = []
            for i in range(n_iter):
                slim_BPR_Cython.fit(epochs=epochs, sgd_mode='adagrad', stop_on_validation=True, learning_rate=0.1,
                                    topK=knn[4],
                                    evaluator_object=None, lower_validatons_allowed=5)
                total.append(slim_BPR_Cython.W_sparse)
            self.S_Slim = reduce(lambda a, b: a + b, total) / n_iter

        if weights[5] != 0:
            p3 = RP3betaRecommender(self.URM)
            p3.fit(topK=knn[5], alpha=0.7, beta=0.3, normalize_similarity=True, implicit=True, min_rating=0)
            self.S_P3 = p3.W_sparse

        if weights[6] != 0:
            slim_Elastic = SLIMElasticNetRecommender(self.URM)
            slim_Elastic.fit(topK=knn[6], l1_ratio=0.00001, positive_only=True)
            self.S_Elastic = slim_Elastic.W_sparse

    def recommend(self, target_playlist):
        row_cb = 0
        row_cf_i = 0
        row_cf_u = 0
        row_svd = 0
        row_slim = 0
        row_elastic = 0
        row_p3 = 0

        if self.weights[0] != 0:
            row_cf_i = (self.URM[target_playlist].dot(self.S_CF_I))*self.weights[0]

        if self.weights[1] != 0:
            row_cf_u = (self.S_CF_U[target_playlist].dot(self.URM))*self.weights[1]

        if self.weights[2] != 0:
            row_cb = (self.URM[target_playlist].dot(self.S_CB))*self.weights[2]

        if self.weights[3] != 0:
            row_svd = sp.csr_matrix(self.U[target_playlist].dot(self.s_Vt)*self.weights[3])

        if self.weights[4] != 0:
            row_slim = self.URM[target_playlist].dot(self.S_Slim)*self.weights[4]

        if self.weights[5] != 0:
            row_p3 = self.URM[target_playlist].dot(self.S_P3)*self.weights[5]

        if self.weights[6] != 0:
            row_elastic = self.URM[target_playlist].dot(self.S_Elastic)*self.weights[6]

        row = (row_cf_i + row_cf_u + row_cb + row_svd + row_slim + row_p3 + row_elastic).toarray().ravel()
        return self.u.get_top_10(self.URM, target_playlist, row)
