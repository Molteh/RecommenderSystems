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

    def fit(self, URM, S_Slim_BPR, target_playlists, knn1, knn2, knn3):
        self.URM = URM
        self.S_Slim_BPR = S_Slim_BPR
        self.target_playlists = target_playlists
        self.S_CF_I = self.u.get_itemsim_CF(self.URM, knn1, 100, "cosine")
        self.S_CF_U = self.u.get_usersim_CF(self.URM, knn2, 100, "cosine")
        self.S_CB = self.u.get_itemsim_CB(knn3, 100, "cosine")

    def recommend(self, is_test, weights):
        print("Recommending", flush=True)

        final_result = pd.DataFrame(index=range(self.target_playlists.shape[0]), columns=('playlist_id', 'track_ids'))

        for i, target_playlist in tqdm(enumerate(np.array(self.target_playlists))):

            R_CB_row = self.URM[target_playlist, :] * self.S_CB
            R_CF_I_row = self.URM[target_playlist, :] * self.S_CF_I
            R_CF_U_row = self.S_CF_U[target_playlist, :] * self.URM
            R_Slim_BPR_row = self.URM[target_playlist, :] * self.S_Slim_BPR

            R_row = (weights[0] * R_CF_I_row) + (weights[1] * R_CF_U_row) + (weights[2] * R_CB_row) + (
                        weights[3] * R_Slim_BPR_row)

            result_tracks = self.u.get_top10_tracks(self.URM, target_playlist[0], R_row)
            string_rec = ' '.join(map(str, result_tracks.reshape(1, 10)[0]))
            final_result['playlist_id'][i] = int(target_playlist)
            if is_test:
                final_result['track_ids'][i] = result_tracks
            else:
                final_result['track_ids'][i] = string_rec

        return final_result
