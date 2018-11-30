import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import MultiLabelBinarizer


from Progetto.utils.cython.Compute_Similarity_Cython import Compute_Similarity_Cython as Cython_Cosine_Similarity


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
    def get_top(URM, target_playlist, row):
        row = row.tolil()
        my_songs = URM.indices[URM.indptr[target_playlist]:URM.indptr[target_playlist + 1]]
        row[0, my_songs] = -np.inf
        row = row.tocsr()
        row = row.indices[np.argsort(row.data)][::-1]
        return row

    @staticmethod
    def get_similarity(matrix, knn, shrink, normalize, similarity):
        similarity = Cython_Cosine_Similarity(matrix, normalize=normalize, shrink=shrink, similarity=similarity,
                                              topK=knn)
        return similarity.compute_similarity().tocsr()

    @staticmethod
    def get_UCM(URM, tfidf):
        if not tfidf:
            return URM
        else:
            UCM = TfidfTransformer().fit_transform(URM.T).T
            return UCM

    def build_URM(self):
        grouped = self.train.groupby('playlist_id', as_index=True).apply((lambda playlist: list(playlist['track_id'])))
        URM = MultiLabelBinarizer(classes=self.tracks['track_id'].unique(), sparse_output=True).fit_transform(grouped)
        return URM.tocsr()

    def get_URM(self):
        return self.URM

    def get_ICM(self, tfidf):  # returns Item Content Matrix
        grouped = self.tracks.groupby('track_id', as_index=True).apply((lambda track: list(track['artist_id'])))

        ICM_artists = MultiLabelBinarizer(classes=self.tracks['artist_id'].unique(), sparse_output=True).fit_transform(
            grouped)
        if tfidf:
            ICM_artists = TfidfTransformer().fit_transform(ICM_artists.T).T

        grouped = self.tracks.groupby('track_id', as_index=True).apply((lambda track: list(track['album_id'])))
        ICM_albums = MultiLabelBinarizer(classes=self.tracks['album_id'].unique(), sparse_output=True).fit_transform(
            grouped)
        if tfidf:
            ICM_albums = TfidfTransformer().fit_transform(ICM_albums.T).T

        ICM = sp.hstack((ICM_artists, ICM_albums))
        return ICM

    def get_itemsim_CB(self, knn, shrink, normalize=True, similarity='cosine', tfidf=True):
        ICM = self.get_ICM(tfidf)
        return self.get_similarity(ICM.T, knn, shrink, normalize, similarity)

    def get_itemsim_CF(self, URM, knn, shrink, normalize=True, similarity='cosine', tfidf=True):
        UCM = self.get_UCM(URM, tfidf)
        return self.get_similarity(UCM, knn, shrink, normalize, similarity)

    def get_usersim_CF(self, URM, knn, shrink, normalize=True, similarity='cosine', tfidf=True):
        UCM = self.get_UCM(URM.T, tfidf)
        return self.get_similarity(UCM, knn, shrink, normalize, similarity)

