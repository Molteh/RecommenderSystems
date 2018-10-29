from Progetto.utils.BPR_utils import SlimBPR_utils


class Slim_BPR(object):

    def __init__(self):
        self.URM = None
        self.num_playlist_to_recommend = None
        self.S = None
        self.u = None

    def fit(self, URM, num_playlist_to_recommend,
            learning_rate, epochs, positive_item_regularization,
            negative_item_regularization, nzz, u, knn):
        self.URM = URM
        self.num_playlist_to_recommend = num_playlist_to_recommend
        self.u = u
        BPR_gen = SlimBPR_utils(self.URM)
        self.S = BPR_gen.get_S_SLIM_BPR(knn)

    def recommend(self, target_playlist):
        row = (self.URM[target_playlist].dot(self.S)).toarray().ravel()
        return self.u.get_top_10(self.URM, target_playlist, row)
