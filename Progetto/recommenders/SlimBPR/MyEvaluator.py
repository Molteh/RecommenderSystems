from sklearn.preprocessing import normalize
import pandas as pd
import numpy as np
from tqdm import tqdm


class MyEvaluator(object):

    def __init__(self, URM, norm=False, e=None):
        self.S_SLIM_BPR = 0
        self.URM = URM
        self.norm = norm
        self.e = e

    def update(self, S_SLIM_BPR=None):
        if S_SLIM_BPR is not None:
            self.S_SLIM_BPR = S_SLIM_BPR

    def recommend(self, target_playlist):
        row_slim = (self.URM[target_playlist].dot(self.S_SLIM_BPR))
        if self.norm:
            row_slim = normalize(row_slim, axis=1, norm='l2')
        return self.get_top_10(self.URM, target_playlist, row_slim.toarray().ravel())

    def rec_and_evaluate(self):
        target_playlists = self.e.get_target_playlists()
        result = self.evaluate(target_playlists)
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
