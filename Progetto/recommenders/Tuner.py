from Progetto.Run import Recommender
from Progetto.recommenders.SlimBPR.SLIM_BPR_Cython import SLIM_BPR_Cython
from sklearn.preprocessing import normalize
from sklearn.utils.extmath import randomized_svd
import scipy.sparse as sp
import pandas as pd
import numpy as np


class Tuner(object):

    def __init__(self):
        self.norm = False
        self.run = None
        self.target_playlists = None
        self.target_tracks = None
        self.S_CB = 0
        self.S_CF_I = 0
        self.S_CF_U = 0
        self.U = 0
        self.s_Vt = 0
        self.S_Slim_I = 0
        self.S_Slim_U = 0
        self.URM = 0

    def fit(self, shrink=(10, 10, 5), knn=(150, 150, 150, 250, 250, 250), k=500, epochs=10, sgd_mode='adagrad', lr=0.1,
            consider=(True, True, True, True, True, True)):
        self.run = Recommender()
        self.URM = self.run.e.get_URM_train()
        self.target_playlists = self.run.e.get_target_playlists()
        self.target_tracks = self.run.e.get_target_tracks()

        print("Building model")

        if consider[0]:
            self.S_CF_I = self.run.u.get_itemsim_CF(self.URM, knn[0], shrink[0])

        if consider[1]:
            self.S_CF_U = self.run.u.get_usersim_CF(self.URM, knn[1], shrink[1])

        if consider[2]:
            self.S_CB = self.run.u.get_itemsim_CB(knn[2], shrink[2])

        if consider[3]:
            self.U, Sigma, VT = randomized_svd(self.URM, n_components=k, n_iter=1, random_state=False)
            self.s_Vt = sp.diags(Sigma) * VT

        if consider[4] != 0:
            slim_BPR_Cython = SLIM_BPR_Cython(self.URM)
            slim_BPR_Cython.fit(epochs=epochs, sgd_mode=sgd_mode, stop_on_validation=True, learning_rate=lr,
                                topK=knn[4])
            self.S_Slim_I = slim_BPR_Cython.W_sparse

        if consider[5] != 0:
            slim_BPR_Cython = SLIM_BPR_Cython(self.URM.T)
            slim_BPR_Cython.fit(epochs=epochs, sgd_mode=sgd_mode, stop_on_validation=True, learning_rate=lr,
                                topK=knn[4])
            self.S_Slim_U = slim_BPR_Cython.W_sparse

        print("Finished building model")

    def recommend(self, target_playlist, weights):
        row_cb = 0
        row_cf_i = 0
        row_cf_u = 0
        row_svd = 0
        row_slim_i = 0
        row_slim_u = 0

        if weights[0] != 0:
            row_cf_i = (self.URM[target_playlist].dot(self.S_CF_I))
            if self.norm:
                row_cf_i = normalize(row_cf_i, axis=1, norm='l2')

        if weights[1] != 0:
            row_cf_u = (self.S_CF_U[target_playlist].dot(self.URM))
            if self.norm:
                row_cf_u = normalize(row_cf_u, axis=1, norm='l2')

        if weights[2] != 0:
            row_cb = (self.URM[target_playlist].dot(self.S_CB))
            if self.norm:
                row_cb = normalize(row_cb, axis=1, norm='l2')

        if weights[3] != 0:
            row_svd = self.U[target_playlist].dot(self.s_Vt)
            if self.norm:
                row_svd = normalize(row_svd, axis=1, norm='l2')

        if weights[4] != 0:
            row_slim_i = (self.URM[target_playlist].dot(self.S_Slim_I))
            if self.norm:
                row_slim_i = normalize(row_slim_i, axis=1, norm='l2')

        if weights[5] != 0:
            row_slim_u = (self.S_Slim_U[target_playlist].dot(self.URM))
            if self.norm:
                row_slim_u = normalize(row_slim_u, axis=1, norm='l2')

        row_cf_i = row_cf_i * weights[0]
        row_cf_u = row_cf_u * weights[1]
        row_cb = row_cb * weights[2]
        row_svd = sp.csr_matrix(row_svd * weights[3])
        row_slim_i = row_slim_i * weights[4]
        row_slim_u = row_slim_u * weights[5]

        row = (row_cf_i + row_cf_u + row_cb + row_svd + row_slim_i + row_slim_u).toarray().ravel()
        return self.run.u.get_top_10(self.URM, target_playlist, row)

    def tune(self, weights=(1.65, 0.55, 1, 0.3, 0.05, 0.001)):
        print("Recommending")
        final_result = pd.DataFrame(index=range(self.target_playlists.shape[0]), columns=('playlist_id', 'track_ids'))

        for i, target_playlist in enumerate(np.array(self.target_playlists)):
            result_tracks = self.recommend(int(target_playlist), weights)
            final_result['playlist_id'][i] = int(target_playlist)
            final_result['track_ids'][i] = result_tracks
        return self.run.e.MAP(final_result, self.target_tracks)
