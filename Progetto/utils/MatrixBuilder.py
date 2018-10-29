import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import MultiLabelBinarizer
from Progetto.utils.cosine_similarity import Compute_Similarity_Python as Cosine_Similarity


class Utils(object):

    def __init__(self, train, tracks, target_playlists):
        self.train = train
        self.tracks = tracks
        self.target_playlists = target_playlists
        self.URM = self.build_URM()

    def get_target_playlists(self):
        return self.target_playlists

    @staticmethod
    def get_top_10(URM, target_playlist, row):
        my_songs = URM.indices[URM.indptr[target_playlist]:URM.indptr[target_playlist + 1]]
        row[my_songs] = -np.inf
        relevant_items_partition = (-row).argpartition(10)[0:10]
        relevant_items_partition_sorting = np.argsort(-row[relevant_items_partition])
        ranking = relevant_items_partition[relevant_items_partition_sorting]
        return ranking

    @staticmethod
    def get_similarity(matrix, normalize, knn, shrink, mode):
        if normalize is False:
            similarity = Cosine_Similarity(dataMatrix=matrix, normalize=False, similarity=mode, topK=knn)
        else:
            similarity = Cosine_Similarity(dataMatrix=matrix, normalize=True, shrink=shrink, similarity=mode, topK=knn)
        S = similarity.compute_similarity()
        return S.tocsr()

    @staticmethod
    def get_UCM(URM):
        UCM = TfidfTransformer().fit_transform(URM.T).T
        return UCM

    def build_URM(self):
        grouped = self.train.groupby('playlist_id', as_index=True).apply((lambda playlist: list(playlist['track_id'])))
        URM = MultiLabelBinarizer(classes=self.tracks['track_id'].unique(), sparse_output=True).fit_transform(grouped)
        return URM.tocsr()

    def get_URM(self):
        return self.URM

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
        return ICM

    def get_itemsim_CB(self, knn, shrink, mode, normalize):
        ICM = self.get_ICM()
        return self.get_similarity(ICM.T, normalize, knn, shrink, mode)

    def get_itemsim_CF(self, URM, knn, shrink, mode, normalize):
        return self.get_similarity(URM, normalize, knn, shrink, mode)

    def get_usersim_CF(self, URM, knn, shrink, mode, normalize):
        return self.get_similarity(URM.T, normalize, knn, shrink, mode)
