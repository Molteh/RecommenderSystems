import pandas as pd
import numpy as np
from tqdm import tqdm


class Hybrid(object):

    def __init__(self, u):
        self.u = u
        self.S_CB = None
        self.S_CF_item = None
        self.S_CF_user = None
        self.S_user = None
        self.target_playlists = None
        self.URM = None

    def fit(self, URM, target_playlists, knn1, knn2, knn3, shrink, mode, normalize):
        self.URM = URM
        self.target_playlists = target_playlists
        self.S_CF_item = self.u.get_itemsim_CF(self.URM, knn1, shrink, mode, normalize)
        self.S_user = self.u.get_usersim_CF(self.URM, knn2, shrink, mode, normalize)
        self.S_CB = self.u.get_itemsim_CB(knn3, shrink, mode, normalize)

    def recommend(self, is_test, weights):
        print("Recommending", flush=True)
        alfa = weights[0]
        beta = weights[1]
        S_item = (alfa * self.S_CF_item) + ((1 - alfa) * self.S_CB)
        final_result = pd.DataFrame(index=range(self.target_playlists.shape[0]), columns=('playlist_id', 'track_ids'))

        for i, target_playlist in tqdm(enumerate(np.array(self.target_playlists))):
            row_user = self.S_user[target_playlist].dot(self.URM)
            row_item = self.URM[target_playlist].dot(S_item)
            row = (beta * row_item) + ((1 - beta) * row_user)

            result_tracks = self.u.get_top10_tracks(self.URM, target_playlist[0], row)
            string_rec = ' '.join(map(str, result_tracks.reshape(1, 10)[0]))
            final_result['playlist_id'][i] = int(target_playlist)
            if is_test:
                final_result['track_ids'][i] = result_tracks
            else:
                final_result['track_ids'][i] = string_rec

        return final_result
