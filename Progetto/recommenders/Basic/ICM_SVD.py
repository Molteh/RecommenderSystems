import scipy.sparse as sp

class ItemSVD(object):

    def __init__(self, u):
        self.S = None
        self.URM = None
        self.u = u

    def fit(self, URM, k=3000, knn=150, evaluate=False):
        self.URM = URM

        if evaluate:
            self.S = self.u.get_itemsim_SVD(k, knn)
            sp.save_npz("./s_itemsvd_new.npz", self.S)
        else:
            self.S = sp.load_npz("similarities/s_itemsvd_current.npz")

    def recommend(self, target_playlist):
        row = self.URM[target_playlist].dot(self.S)
        return self.u.get_top_10(self.URM, target_playlist, row.toarray().ravel())

