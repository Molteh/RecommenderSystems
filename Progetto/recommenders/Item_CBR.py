class Item_CBR(object):

    def __init__(self, u):
        self.u = u
        self.URM = None
        self.S = None

    def fit(self, URM, knn, shrink, weights, cython):
        self.URM = URM
        self.S = self.u.get_itemsim_CB(knn, shrink, weights, cython)

    def recommend(self, target_playlist):
        row = self.URM[target_playlist].dot(self.S).toarray().ravel()
        return self.u.get_top_10(self.URM, target_playlist, row)
