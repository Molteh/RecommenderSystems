import pandas as pd
import numpy as np
from tqdm import tqdm


class Item_CFR(object):

    def __init__(self, u):
        self.u = u
        self.URM = None
        self.target_playlists = None
        self.S = None

    def fit(self, URM, target_playlists, knn, shrink, mode, normalize):
        self.URM = URM
        self.target_playlists = target_playlists
        self.S = self.u.get_itemsim_CF(self.URM, knn, shrink, mode, normalize)

    def recommend(self, target_playlist):
        row = self.URM[target_playlist].dot(self.S).toarray().ravel()
        my_songs = self.URM.indices[self.URM.indptr[target_playlist]:self.URM.indptr[target_playlist + 1]]
        row[my_songs] = -np.inf
        relevant_items_partition = (-row).argpartition(10)[0:10]
        relevant_items_partition_sorting = np.argsort(-row[relevant_items_partition])
        ranking = relevant_items_partition[relevant_items_partition_sorting]
        return ranking
