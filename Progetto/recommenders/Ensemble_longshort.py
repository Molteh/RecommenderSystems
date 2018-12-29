class Ensemble_longshort(object):

    def __init__(self, u):
        self.u = u
        self.S_CB = 0
        self.S_CB_l10 = 0
        self.S_CF_I = 0
        self.S_CF_I_l10 = 0
        self.S_CF_U = 0
        self.S_CF_U_l10 = 0
        self.URM = 0
        self.weights = 0


    def fit(self, URM, knn, shrink, weights):
        self.URM = URM
        self.weights = weights

        if weights[0] != 0:
            self.S_CF_I = self.u.get_itemsim_CF(self.URM, 150, 10, normalize=True, tfidf=True)
            self.S_CF_I_l10 = self.u.get_itemsim_CF(self.URM, knn[0], shrink[0], normalize=True, tfidf=False)

        if weights[1] != 0:
            self.S_CF_U = self.u.get_usersim_CF(self.URM, 150, 10, normalize=True, tfidf=True)
            self.S_CF_U_l10 = self.u.get_usersim_CF(self.URM, knn[1], shrink[1], normalize=True, tfidf=True)

        if weights[2] != 0:
            self.S_CB = self.u.get_itemsim_CB(100, 5, tfidf=True)
            self.S_CB_l10 = self.u.get_itemsim_CB(knn[2], shrink[2], tfidf=False)


    def recommend(self, target_playlist):
        row_cb = 0
        row_cf_i = 0
        row_cf_u = 0

        if self.weights[0] != 0:
            row_cf_i = (self.URM[target_playlist].dot(self.S_CF_I))*1.65

        if self.weights[1] != 0:
            row_cf_u = (self.S_CF_U[target_playlist].dot(self.URM))*0.55

        if self.weights[2] != 0:
            row_cb = (self.URM[target_playlist].dot(self.S_CB))*1

        row = (row_cf_i + row_cf_u + row_cb).toarray().ravel()
        return self.u.get_top_10(self.URM, target_playlist, row)


    def recommend_l10(self, target_playlist):
        row_cb = 0
        row_cf_i = 0
        row_cf_u = 0

        if self.weights[0] != 0:
            row_cf_i = (self.URM[target_playlist].dot(self.S_CF_I_l10))*self.weights[0]

        if self.weights[1] != 0:
            row_cf_u = (self.S_CF_U_l10[target_playlist].dot(self.URM))*self.weights[1]

        if self.weights[2] != 0:
            row_cb = (self.URM[target_playlist].dot(self.S_CB_l10))*self.weights[2]

        row = (row_cf_i + row_cf_u + row_cb).toarray().ravel()
        return self.u.get_top_10(self.URM, target_playlist, row)
