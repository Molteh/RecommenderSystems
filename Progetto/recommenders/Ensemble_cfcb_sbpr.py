from Progetto.utils.BPR_utils import SlimBPR_utils


class Ensemble_cfcb_sbpr(object):

    def __init__(self, u):
        self.u = u
        self.S_CB = None
        self.S_CF_I = None
        self.S_CF_U = None
        self.S = None
        self.URM = None
        self.weights = None

    def fit(self, URM, knn1, knn2, knn3, knn4, shrink, weights):
        self.URM = URM
        self.weights = weights
        BPR_gen = SlimBPR_utils(self.URM)
        self.S = BPR_gen.get_S_SLIM_BPR(knn4)
        self.S_CF_I = self.u.get_itemsim_CF(self.URM, knn1, shrink,)
        self.S_CF_U = self.u.get_usersim_CF(self.URM, knn2, shrink,)
        self.S_CB = self.u.get_itemsim_CB(knn3, shrink)

    def recommend(self, target_playlist):
            row_R_CB = self.URM[target_playlist].dot(self.S_CB)
            row_R_CF_I = self.URM[target_playlist].dot(self.S_CF_I)
            row_R_CF_U = self.S_CF_U[target_playlist].dot(self.URM)
            row_R_Slim_BPR = self.URM[target_playlist].dot(self.S)
            row = (row_R_CF_I + (self.weights[0] * row_R_CF_U) + (self.weights[1] * row_R_CB) + (
                        self.weights[2] * row_R_Slim_BPR)).toarray().ravel()

            return self.u.get_top_10(self.URM, target_playlist, row)
