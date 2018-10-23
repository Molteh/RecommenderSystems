import pandas as pd
import numpy as np
from tqdm import tqdm


class SlimBPR(object):

    def __init__(self):
        self.URM = None
        self.target_playlists = None
        self.num_playlist_to_recommend = None
        self.Slim = None
        self.u = None

    def fit(self, URM, Slim, target_playlists, num_playlist_to_recommend,
            learning_rate, epochs, positive_item_regularization,
            negative_item_regularization, nzz, u):
        self.URM = URM
        self.target_playlists = target_playlists
        self.num_playlist_to_recommend = num_playlist_to_recommend
        self.Slim = Slim
        self.u = u

    def recommend(self, is_test):
        self.is_test = is_test

        final_result = pd.DataFrame(index=range(self.target_playlists.shape[0]), columns=('playlist_id', 'track_ids'))

        print('Predicting...', flush=True)
        for j, i in tqdm(enumerate(np.array(self.target_playlists))):
            row = self.URM[i].dot(self.Slim)

            # Make prediction
            result_tracks = self.u.get_top10_tracks(self.URM, i[0], row)
            string_rec = ' '.join(map(str, result_tracks.reshape(1, 10)[0]))
            final_result['playlist_id'][j] = int(i)
            if is_test:
                final_result['track_ids'][j] = result_tracks
            else:
                final_result['track_ids'][j] = string_rec

        return final_result
