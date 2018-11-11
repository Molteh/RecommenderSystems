import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import MultiLabelBinarizer

try:
    from Progetto.utils.cython.cosine_similarity import Cosine_Similarity as Cython_Cosine_Similarity
except ImportError:
    print("Unable to load Cython Cosine_Similarity, reverting to Python")

from Progetto.utils.cosine_similarity import Compute_Similarity_Python as Cosine_Similarity


class Utils(object):

    def __init__(self, train, tracks, target_playlists, train_sequential):
        self.train = train
        self.tracks = tracks
        self.target_playlists = target_playlists
        self.train_sequential = train_sequential
        self.URM = self.build_URM()

    def get_target_playlists(self):
        return self.target_playlists

    def get_train_sequential(self):
        return self.train_sequential

    @staticmethod
    def get_top_10(URM, target_playlist, row):
        my_songs = URM.indices[URM.indptr[target_playlist]:URM.indptr[target_playlist + 1]]
        row[my_songs] = -np.inf
        relevant_items_partition = (-row).argpartition(10)[0:10]
        relevant_items_partition_sorting = np.argsort(-row[relevant_items_partition])
        ranking = relevant_items_partition[relevant_items_partition_sorting]
        return ranking

    @staticmethod
    def get_similarity(matrix, knn, shrink, cython):
        if cython:
            similarity = Cython_Cosine_Similarity(matrix, normalize=True, shrink=shrink, mode='cosine', topK=knn)
        else:
            similarity = Cosine_Similarity(dataMatrix=matrix, normalize=True, shrink=shrink, similarity='cosine', topK=knn)

        return similarity.compute_similarity().tocsr()

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

    def get_ICM(self, alfa):  # returns Item Content Matrix
        grouped = self.tracks.groupby('track_id', as_index=True).apply((lambda track: list(track['artist_id'])))

        ICM_artists = MultiLabelBinarizer(classes=self.tracks['artist_id'].unique(), sparse_output=True).fit_transform(
            grouped)
        ICM_artists = TfidfTransformer().fit_transform(ICM_artists.T).T

        grouped = self.tracks.groupby('track_id', as_index=True).apply((lambda track: list(track['album_id'])))
        ICM_albums = MultiLabelBinarizer(classes=self.tracks['album_id'].unique(), sparse_output=True).fit_transform(
            grouped)
        ICM_albums = TfidfTransformer().fit_transform(ICM_albums.T).T

        ICM = sp.hstack((alfa*ICM_artists, ICM_albums))
        return ICM

    def get_itemsim_CB(self, knn, shrink, alfa, cython):
        ICM = self.get_ICM(alfa)
        return self.get_similarity(ICM.T, knn, shrink, cython)

    def get_itemsim_CF(self, URM, knn, shrink, cython):
        UCM = self.get_UCM(URM)
        return self.get_similarity(UCM, knn, shrink, cython)

    def get_usersim_CF(self, URM, knn, shrink, cython):
        UCM = self.get_UCM(URM.T)
        return self.get_similarity(UCM, knn, shrink, cython)
