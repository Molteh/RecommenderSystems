from sklearn.preprocessing import normalize
import pandas as pd
import numpy as np
from tqdm import tqdm


class MyEvaluator(object):

    def __init__(self, URM, target_playlists, norm=True, weights=(0,0,0,0,0), e=None):
        self.S_ICF = 0
        self.S_UCF = 0
        self.S_CBR = 0
        self.S_SVD = 0
        self.S_SLIM_BPR = 0
        self.URM = URM
        self.target_playlists = target_playlists
        self.norm = norm
        self.weights = weights
        self.e = e

    def setWeights(self, weights):
        self.weights = weights

    def update(self, S_ICF=None, S_UCF=None, S_CBR=None, S_SVD=None, S_SLIM_BPR=None):

        if S_ICF is not None:
            self.S_ICF = S_ICF

        if S_UCF is not None:
            self.S_UCF = S_UCF

        if S_CBR is not None:
            self.S_CBR = S_CBR

        if S_SVD is not None:
            self.S_SVD = S_SVD

        if S_SLIM_BPR is not None:
            self.S_SLIM_BPR = S_SLIM_BPR

    def recommend(self, target_playlist):
        row_cb = 0
        row_cf_i = 0
        row_cf_u = 0
        row_svd = 0
        row_slim = 0

        if self.weights[0] != 0:
            row_cf_i = (self.URM[target_playlist].dot(self.S_ICF))
            if self.norm:
                row_cf_i = normalize(row_cf_i, axis=1, norm='l2')

        if self.weights[1] != 0:
            row_cf_u = (self.S_UCF[target_playlist].dot(self.URM))
            if self.norm:
                row_cf_u = normalize(row_cf_u, axis=1, norm='l2')

        if self.weights[2] != 0:
            row_cb = (self.URM[target_playlist].dot(self.S_CBR))
            if self.norm:
                row_cb = normalize(row_cb, axis=1, norm='l2')

        if self.weights[3] != 0:
            row_svd = (self.URM[target_playlist].dot(self.S_SVD))
            if self.norm:
                row_svd = normalize(row_svd, axis=1, norm='l2')

        if self.weights[4] != 0:
            row_slim = (self.URM[target_playlist].dot(self.S_SLIM_BPR))
            if self.norm:
                row_slim = normalize(row_slim, axis=1, norm='l2')

        row_cf_i = row_cf_i * self.weights[0]
        row_cf_u = row_cf_u * self.weights[1]
        row_cb = row_cb * self.weights[2]
        row_svd = row_svd * self.weights[3]
        row_slim = row_slim * self.weights[4]

        row = (row_cf_i + row_cf_u + row_cb + row_svd + row_slim).toarray().ravel()

        return self.get_top_10(self.URM, target_playlist, row)

    def rec_and_evaluate(self):
        result = self.evaluate(self.target_playlists)
        return self.e.MAP(result, self.e.get_target_tracks())

    def evaluate(self, target_playlists):
        final_result = pd.DataFrame(index=range(target_playlists.shape[0]), columns=('playlist_id', 'track_ids'))

        for i, target_playlist in tqdm(enumerate(np.array(target_playlists))):
            result_tracks = self.recommend(int(target_playlist))
            final_result['playlist_id'][i] = int(target_playlist)
            final_result['track_ids'][i] = result_tracks
        return final_result

    def get_top_10(self, URM, target_playlist, row):
        my_songs = URM.indices[URM.indptr[target_playlist]:URM.indptr[target_playlist + 1]]
        row[my_songs] = -np.inf
        relevant_items_partition = (-row).argpartition(10)[0:10]
        relevant_items_partition_sorting = np.argsort(-row[relevant_items_partition])
        ranking = relevant_items_partition[relevant_items_partition_sorting]
        return ranking
