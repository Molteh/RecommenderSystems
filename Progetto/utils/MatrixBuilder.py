import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import MultiLabelBinarizer, normalize
from Progetto.utils.cosine_similarity import Compute_Similarity_Python as Cosine_Similarity

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
        indices = np.intersect1d(my_indices, row.indices)
        row[0, indices] = -np.inf
        top10_tracks = row.toarray().flatten().argsort()[-10:][::-1]
        return top10_tracks

    @staticmethod
    def get_similarity_normalized(matrix, knn, shrink, mode):
        similarity = Cosine_Similarity(dataMatrix=matrix, normalize=True, shrink=shrink, similarity=mode, topK=knn)
        S = similarity.compute_similarity()
        return S.tocsr()

    @staticmethod
    def get_similarity(matrix, knn):
        result = []
        matrix = matrix.tocsr()
        T = matrix.T.tocsr()

        for row in matrix:
            new_row = row.dot(T)
            indices = new_row.data.argsort()[:-knn]
            new_row.data[indices] = 0
            sp.csr_matrix.eliminate_zeros(new_row)
            result.append(new_row)

        S = sp.vstack(result).tolil()
        S.setdiag(0)
        return S.tocsr()

    def get_URM(self):
        grouped = self.train.groupby('playlist_id', as_index=True).apply((lambda playlist: list(playlist['track_id'])))
        URM = MultiLabelBinarizer(classes=self.tracks['track_id'].unique(), sparse_output=True).fit_transform(grouped)
        return URM

    def get_UCM(self, URM):
        UCM = TfidfTransformer().fit_transform(URM.T).T
        return UCM

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
        if normalize:
            return self.get_similarity_normalized(ICM.T, knn, shrink, mode)
        else:
            return self.get_similarity(ICM.T, knn)

    def get_itemsim_CF(self, URM, knn, shrink, mode, normalize):
        UCM = self.get_UCM(URM)
        if normalize:
            return self.get_similarity_normalized(UCM, knn, shrink, mode)
        else:
            return self.get_similarity(UCM.T, knn)

    def get_usersim_CF(self, URM, knn, shrink, mode, normalize):
        UCM = self.get_UCM(URM)
        if normalize:
            return self.get_similarity_normalized(UCM.T, knn, shrink, mode)
        else:
            return self.get_similarity(UCM, knn)
