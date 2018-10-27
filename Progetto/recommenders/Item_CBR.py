import pandas as pd
import numpy as np
from tqdm import tqdm


class Item_CBR(object):

    def __init__(self, u):
        self.u = u
        self.URM = None
        self.target_playlists = None
        self.S = None

    def fit(self, URM, target_playlists, knn, shrink, mode, normalize):
        self.URM = URM
        self.target_playlists = target_playlists
        self.S = self.u.get_itemsim_CB(knn, shrink, mode, normalize)

    def recommend(self, target_playlist):
        row = self.URM[target_playlist].dot(self.S).toarray().ravel()
        return self.u.get_top_10(self.URM, target_playlist, row)
