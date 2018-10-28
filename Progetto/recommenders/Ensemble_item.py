class Ensemble_item(object):

    def __init__(self, u):
        self.u = u
        self.S_CB = None
        self.S_CF = None
        self.URM = None
        self.alfa = None

    def fit(self, URM, knn1, knn2, shrink, mode, normalize, alfa):
        self.URM = URM
        self.alfa = alfa
        self.S_CF = self.u.get_itemsim_CF(self.URM, knn1, shrink, mode, normalize)
        self.S_CB = self.u.get_itemsim_CB(knn2, shrink, mode, normalize)

    def recommend(self, target_playlist):
        row_cb = self.URM[target_playlist].dot(self.S_CB)
        row_cf = self.URM[target_playlist].dot(self.S_CF)
        row = ((self.alfa*row_cb) + ((1-self.alfa)*row_cf)).toarray().ravel()
        return self.u.get_top_10(self.URM, target_playlist, row)
