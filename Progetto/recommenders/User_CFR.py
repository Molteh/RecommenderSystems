import pandas as pd
import numpy as np
from tqdm import tqdm


class User_CFR(object):

    def __init__(self, u):
        self.u = u
        self.URM = None
        self.target_playlists = None
        self.S = None

    def fit(self, URM, target_playlists, knn, shrink, mode):
        self.URM = URM
        self.target_playlists = target_playlists
        self.S = self.u.get_usersim_CF(self.URM, knn, shrink, mode)

    def recommend(self, is_test):
        print("Recommending", flush=True)

        final_result = pd.DataFrame(index=range(self.target_playlists.shape[0]), columns=('playlist_id', 'track_ids'))

        for i, target_playlist in tqdm(enumerate(np.array(self.target_playlists))):

            URM_row = self.S[target_playlist, :] * self.URM

            result_tracks = self.u.get_top10_tracks(self.URM, target_playlist[0], URM_row)
            string_rec = ' '.join(map(str, result_tracks.reshape(1, 10)[0]))
            final_result['playlist_id'][i] = int(target_playlist)
            if is_test:
                final_result['track_ids'][i] = result_tracks
            else:
                final_result['track_ids'][i] = string_rec

        return final_result