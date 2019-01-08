from Progetto.recommenders.MF.ALS import IALS_numpy
import scipy.sparse as sp


class ALS(object):

    def __init__(self, u):
        self.S_user = None
        self.S_item = None
        self.URM = None
        self.u = u

    def fit(self, URM, k, n_iter, reg, mode="linear"):
        self.URM = URM

        als = IALS_numpy(k,reg,n_iter, scaling=mode)
        als.fit(URM.astype('float64'))
        self.S_user = sp.csr_matrix(als.X)
        self.S_item = sp.csr_matrix(als.Y).T

    def recommend(self, target_playlist):
        row = self.S_user[target_playlist].dot(self.S_item)
        return self.u.get_top_10(self.URM, target_playlist, row.toarray().ravel())

