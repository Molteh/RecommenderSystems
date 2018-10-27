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

    def recommend(self, target_playlist):
        row = self.S[target_playlist].dot(self.URM).toarray().ravel()
        return self.u.get_top_10(self.URM, target_playlist, row)
