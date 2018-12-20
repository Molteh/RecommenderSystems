from sklearn.utils.extmath import randomized_svd
import scipy.sparse as sp


class PureSVD(object):

    def __init__(self, u):
        self.U = None
        self.s_Vt = None
        self.URM = None
        self.u = u

    def fit(self, URM, k, n_iter, random_state, bm25=True, K1=2, B=0.9):
        self.URM = URM

        if bm25:
            UCM = self.u.okapi_BM_25(URM, K1, B)
        else:
            UCM = URM

        self.U, Sigma, VT = randomized_svd(UCM, n_components=k, n_iter=n_iter, random_state=random_state)
        self.s_Vt = sp.diags(Sigma) * VT

    def recommend(self, target_playlist):
        row = self.U[target_playlist].dot(self.s_Vt)
        return self.u.get_top_10(self.URM, target_playlist, row)

