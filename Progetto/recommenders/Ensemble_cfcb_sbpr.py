import pandas as pd
import numpy as np
from tqdm import tqdm


class Ensemble_cfcb_sbpr(object):

    def __init__(self, u):
        self.u = u
        self.S_CB = None
        self.S_CF_I = None
        self.S_CF_U = None
        self.S_Slim_BPR = None
        self.target_playlists = None
        self.URM = None
        self.weights = None

    def fit(self, URM, S_Slim_BPR, target_playlists, knn1, knn2, knn3, shrink, mode, normalize, weights):
        self.URM = URM
        self.S_Slim_BPR = S_Slim_BPR
        self.weights = weights
        self.target_playlists = target_playlists
        self.S_CF_I = self.u.get_itemsim_CF(self.URM, knn1, shrink, mode, normalize)
        self.S_CF_U = self.u.get_usersim_CF(self.URM, knn2, shrink, mode, normalize)
        self.S_CB = self.u.get_itemsim_CB(knn3, shrink, mode, normalize)

    def recommend(self, target_playlist):
            row_R_CB = self.URM[target_playlist].dot(self.S_CB)
            row_R_CF_I = self.URM[target_playlist].dot(self.S_CF_I)
            row_R_CF_U = self.S_CF_U[target_playlist].dot(self.URM)
            row_R_Slim_BPR = self.URM[target_playlist].dot(self.S_Slim_BPR)
            row = (self.weights[0] * row_R_CF_I) + (self.weights[1] * row_R_CF_U) + (self.weights[2] * row_R_CB) + (
                        self.weights[3] * row_R_Slim_BPR).toarray().ravel()

            my_songs = self.URM.indices[self.URM.indptr[target_playlist]:self.URM.indptr[target_playlist + 1]]
            row[my_songs] = -np.inf
            relevant_items_partition = (-row).argpartition(10)[0:10]
            relevant_items_partition_sorting = np.argsort(-row[relevant_items_partition])
            ranking = relevant_items_partition[relevant_items_partition_sorting]
            return ranking
