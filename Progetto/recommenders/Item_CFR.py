import pandas as pd
import numpy as np
from tqdm import tqdm


class Item_CFR(object):

    def __init__(self, u):
        self.u = u
        self.URM = None
        self.target_playlists = None
        self.S = None

    def fit(self, URM, target_playlists, knn):
        self.URM = URM
        self.target_playlists = target_playlists
        self.S = self.u.get_itemsim_CF(self.URM, knn)

    def recommend(self, is_test):
        print("Recommending", flush=True)
        R = self.URM * self.S
        final_result = pd.DataFrame(index=range(self.target_playlists.shape[0]), columns=('playlist_id', 'track_ids'))

        for i, target_playlist in tqdm(enumerate(np.array(self.target_playlists))):
            result_tracks = self.u.get_top10_tracks(self.URM, target_playlist[0], R[target_playlist[0]])
            string_rec = ' '.join(map(str, result_tracks.reshape(1, 10)[0]))
            final_result['playlist_id'][i] = int(target_playlist)
            if is_test:
                final_result['track_ids'][i] = result_tracks
            else:
                final_result['track_ids'][i] = string_rec

        return final_result