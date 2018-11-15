from Progetto.utils.MatrixBuilder import Utils
from Progetto.utils.Evaluation import Eval
from Progetto.recommenders.Ensemble_pre import Ensemble_pre
from Progetto.recommenders.Ensemble_post import Ensemble_post
from Progetto.recommenders.Slim_BPR_Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
#from Progetto.recommenders.MFBPR import MFBPR
import pandas as pd
import numpy as np
from tqdm import tqdm


class Recommender(object):

    def __init__(self, n=0):
        self.train = pd.read_csv("data/train.csv")
        self.tracks = pd.read_csv("data/tracks.csv")
        self.target_playlists = pd.read_csv("data/target_playlists.csv")
        self.train_sequential = pd.read_csv("data/train_sequential.csv")
        self.u = Utils(self.train, self.tracks, self.target_playlists, self.train_sequential)
        self.e = Eval(self.u, (np.random.choice(range(10000), 5000, replace=False)).tolist())
        self.URM_full = self.preprocess_URM(self.u.get_URM(), self.target_playlists, n)
        self.URM_train = self.preprocess_URM(self.e.get_URM_train(), self.e.get_target_playlists(), n)

    @staticmethod
    def evaluate(recommender, is_test, target_playlists):
        final_result = pd.DataFrame(index=range(target_playlists.shape[0]), columns=('playlist_id', 'track_ids'))

        for i, target_playlist in tqdm(enumerate(np.array(target_playlists))):
            result_tracks = recommender.recommend(int(target_playlist))
            string_rec = ' '.join(map(str, result_tracks.reshape(1, 10)[0]))
            final_result['playlist_id'][i] = int(target_playlist)
            if is_test:
                final_result['track_ids'][i] = result_tracks
            else:
                final_result['track_ids'][i] = string_rec
        return final_result

    @staticmethod
    def preprocess_URM(URM, target_playlists, n):
        if n == 0:
            return URM
        URM_new = URM.copy().tolil()
        total_users = URM.shape[0]
        possible_playlists = [i for i in range(total_users) if len(
            URM.indices[URM.indptr[i]:URM.indptr[i + 1]]) <= n]
        discard = np.setdiff1d(np.array(possible_playlists), target_playlists['playlist_id'])
        URM_new[discard, :] = 0
        return URM_new.tocsr()

    def rec_and_evaluate(self, rec, target_playlists):
        result = self.evaluate(rec, True, target_playlists)
        return self.e.MAP(result, self.e.get_target_tracks())

    def rec_and_save(self, rec, target_playlists, path):
        result = self.evaluate(rec, False, target_playlists)
        result.to_csv(path, index=False)

    def recommend_SlimBPR_Cython(self, is_test, recompile=False, epochs=5, learning_rate=0.1, knn=250, sparse_weights=False,
                                 pt=0, sgd='rmsprop'):
        if is_test:
            rec = SLIM_BPR_Cython(self.URM_train, recompile_cython=recompile, positive_threshold=pt, sparse_weights=sparse_weights, sgd_mode=sgd)
            target_playlists = self.e.get_target_playlists()
            rec.fit(epochs=epochs, batch_size=1, learning_rate=learning_rate, topK=knn)
            return self.rec_and_evaluate(rec, target_playlists)
        else:
            rec = SLIM_BPR_Cython(self.URM_full, recompile_cython=recompile, positive_threshold=pt, sparse_weights=sparse_weights, sgd_mode=sgd)
            target_playlists = self.u.get_target_playlists()
            rec.fit(epochs=epochs, batch_size=1, learning_rate=learning_rate, topK=knn)
            self.rec_and_save(rec, target_playlists, "predictions/slim_BPR.csv")

    def recommend_ensemble_pre(self, is_test, knn=(150, 150, 150, 250, 250), shrink=(10, 10, 5),
                                   weights=(1.65, 0.55, 1, 0.1, 0.005), k=300, cython=True, epochs=5):
        rec = Ensemble_pre(self.u)
        if is_test:
            target_playlists = self.e.get_target_playlists()
            rec.fit(self.URM_train, knn, shrink, weights, k, cython, epochs)
            return self.rec_and_evaluate(rec, target_playlists)
        else:
            target_playlists = self.u.get_target_playlists()
            rec.fit(self.URM_full, knn, shrink, weights, k, cython, epochs)
            self.rec_and_save(rec, target_playlists, "predictions/ensemble_pre.csv")

    def recommend_ensemble_post(self, is_test, knn=(150, 150, 150, 150, 250), shrink=(10, 10, 5),
                                   weights=(1.65, 0.55, 1, 0.1, 0.005), k=300, cython=True, epochs=5):
        rec = Ensemble_post(self.u)
        if is_test:
            target_playlists = self.e.get_target_playlists()
            rec.fit(self.URM_train, knn, shrink, weights, k, cython, epochs)
            return self.rec_and_evaluate(rec, target_playlists)
        else:
            target_playlists = self.u.get_target_playlists()
            rec.fit(self.URM_full, knn, shrink, weights, k, cython, epochs)
            self.rec_and_save(rec, target_playlists, "predictions/ensemble_post.csv")

    #OUTDATED!
    def recommend_MFBPR(self, is_test, epochs=10):

        if is_test:
            rec = MFBPR(self.URM_train, self.u, epochs=epochs)
            target_playlists = self.e.get_target_playlists()
            rec.fit()
            return self.rec_and_evaluate(rec, target_playlists)
        else:
            rec = MFBPR(self.URM_train, self.u, epochs=epochs)
            target_playlists = self.u.get_target_playlists()
            rec.fit()
            self.rec_and_save(rec, target_playlists, "predictions/MFBPR.csv")


if __name__ == '__main__':
    run = Recommender(n=5)
    run.recommend_ensemble_post(False, weights=(1.55, 0.65, 1, 0.1, 0.005), epochs=50)











