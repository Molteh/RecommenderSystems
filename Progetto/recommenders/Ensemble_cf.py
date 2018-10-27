import pandas as pd
import numpy as np
from tqdm import tqdm


class Ensemble_cf(object):

    def __init__(self, u):
        self.u = u
        self.S_CF_I = None
        self.S_CF_U = None
        self.target_playlists = None
        self.URM = None
        self.alfa = None

    def fit(self, URM, target_playlists, knn1, knn2, shrink, mode, normalize, alfa):
        self.URM = URM
        self.alfa = alfa
        self.target_playlists = target_playlists
        self.S_CF_I = self.u.get_itemsim_CF(self.URM, knn1, shrink, mode, normalize)
        self.S_CF_U = self.u.get_usersim_CF(self.URM, knn2, shrink, mode, normalize)

    def recommend(self, target_playlist):
        row_cf_i = self.URM[target_playlist].dot(self.S_CF_I)
        row_cf_u = self.S_CF_U[target_playlist].dot(self.URM)
        row = ((self.alfa * row_cf_i) + ((1-self.alfa) * row_cf_u)).toarray().ravel()
        my_songs = self.URM.indices[self.URM.indptr[target_playlist]:self.URM.indptr[target_playlist + 1]]
        row[my_songs] = -np.inf
        relevant_items_partition = (-row).argpartition(10)[0:10]
        relevant_items_partition_sorting = np.argsort(-row[relevant_items_partition])
        ranking = relevant_items_partition[relevant_items_partition_sorting]
        return ranking
