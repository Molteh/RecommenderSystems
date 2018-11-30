class Item_CBR(object):

    def __init__(self, u):
        self.u = u
        self.URM = 0
        self.S_CB = 0

    def fit(self, URM, knn, shrink, normalize, similarity, tfidf):
        self.URM = URM
        self.S_CB = self.u.get_itemsim_CB(knn, shrink, normalize, similarity, tfidf)

    def recommend(self, target_playlist):
        row_cb = (self.URM[target_playlist].dot(self.S_CB)).toarray().ravel()
        return self.u.get_top_10(self.URM, target_playlist, row_cb)
