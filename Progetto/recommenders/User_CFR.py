import numpy as np
from tqdm import tqdm


class User_CFR(object):

    def __init__(self, u):
        self.u = u
        self.URM = None
        self.target_playlists = None
        self.S = None

    def fit(self, URM, target_playlists, knn, shrink, mode, normalize):
        self.URM = URM
        self.target_playlists = target_playlists
        self.S = self.u.get_usersim_CF(self.URM, knn, shrink, mode, normalize)

    def recommend(self, user_id, n=10):
        row = self.S[user_id].dot(self.URM).toarray().ravel()
        my_songs = self.URM.indices[self.URM.indptr[user_id]:self.URM.indptr[user_id + 1]]
        row[my_songs] = -np.inf
        relevant_items_partition = (-row).argpartition(n)[0:n]
        relevant_items_partition_sorting = np.argsort(-row[relevant_items_partition])
        ranking = relevant_items_partition[relevant_items_partition_sorting]
        return ranking
