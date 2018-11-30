from sklearn.utils.extmath import randomized_svd
import scipy.sparse as sp


class PureSVD(object):

    def __init__(self, u):
        self.U = None
        self.s_Vt = None
        self.URM = None
        self.u = u

    def fit(self, URM, k, n_iter, random_state):
        self.URM = URM
        self.U, Sigma, VT = randomized_svd(URM, n_components=k, n_iter=n_iter, random_state=random_state)
        self.s_Vt = sp.diags(Sigma) * VT

    def recommend(self, target_playlist):
        row = self.U[target_playlist].dot(self.s_Vt)
        return self.u.get_top_10(self.URM, target_playlist, row)

