import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import MultiLabelBinarizer, normalize
from Progetto.utils.cosine_similarity import Cosine_Similarity

class Utils(object):

    def __init__(self, train, tracks, target_playlists):
        self.train = train
        self.tracks = tracks
        self.target_playlists = target_playlists

    def get_target_playlists(self):
        return self.target_playlists

    @staticmethod
    def get_top10_tracks(URM, my_id, row):
        my_indices = URM.indices[URM.indptr[my_id]:URM.indptr[my_id + 1]]
        target_indices = list(np.intersect1d(row.indices, my_indices))
        row[0, target_indices] = 0
        row.eliminate_zeros()
        top10_tracks = row.toarray().flatten().argsort()[-10:][::-1]
        return top10_tracks

    def get_URM(self):
        grouped = self.train.groupby('playlist_id', as_index=True).apply((lambda playlist: list(playlist['track_id'])))
        URM = MultiLabelBinarizer(classes=self.tracks['track_id'].unique(), sparse_output=True).fit_transform(grouped)
        return URM

    def get_UCM(self, URM):
        UCM = TfidfTransformer().fit_transform(URM.T).T
        return normalize(UCM, 'l2', 0).tocsr()

    def get_ICM(self):  # returns Item Content Matrix
        grouped = self.tracks.groupby('track_id', as_index=True).apply((lambda track: list(track['artist_id'])))

        ICM_artists = MultiLabelBinarizer(classes=self.tracks['artist_id'].unique(), sparse_output=True).fit_transform(
            grouped)
        ICM_artists = ICM_artists * 0.8  # best weight for the artis feature
        ICM_artists = TfidfTransformer().fit_transform(ICM_artists.T).T

        grouped = self.tracks.groupby('track_id', as_index=True).apply((lambda track: list(track['album_id'])))
        ICM_albums = MultiLabelBinarizer(classes=self.tracks['album_id'].unique(), sparse_output=True).fit_transform(
            grouped)
        ICM_albums = TfidfTransformer().fit_transform(ICM_albums.T).T

        ICM = sp.hstack((ICM_artists, ICM_albums))
        return normalize(ICM, 'l2', 0).tocsr()

    def get_itemsim_CB(self, knn):
        ICM = self.get_ICM()

        similarity = Cosine_Similarity(dataMatrix=ICM.T, normalize=True, shrink=100, mode='cosine',
                                                         topK = knn)
        S = similarity.compute_similarity()

        return S.tocsr()

    def get_itemsim_CF(self, URM, knn):
        UCM = self.get_UCM(URM)

        similarity = Cosine_Similarity(dataMatrix=UCM, normalize=True, shrink=200, mode='cosine',
                                                         topK = knn)
        S = similarity.compute_similarity()

        return S.tocsr()

    def get_usersim_CF(self, URM, knn):
        UCM = self.get_UCM(URM)

        similarity = Cosine_Similarity(dataMatrix=UCM.T, normalize=True, shrink=100, mode='cosine',
                                                         topK = knn)
        S = similarity.compute_similarity()

        return S.tocsr()
