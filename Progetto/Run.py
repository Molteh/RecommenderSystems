from Progetto.utils.MatrixBuilder import Utils
from Progetto.utils.Evaluation import Eval
from Progetto.recommenders.Ensemble_post import Ensemble_post
from Progetto.recommenders.Ensemble_list import Ensemble_list
from Progetto.recommenders.MF_BPR import MF_BPR
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
            final_result['playlist_id'][i] = int(target_playlist)
            if is_test:
                final_result['track_ids'][i] = result_tracks
            else:
                string_rec = ' '.join(map(str, result_tracks.reshape(1, 10)[0]))
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

    def recommend_ensemble_post(self, is_test, knn=(150, 150, 150, 250, 250), shrink=(10, 10, 5),
                                weights=(1.65, 0.55, 1, 0.1, 0.005), k=300, epochs=5, normalize=False,
                                lr=0.1):
        rec = Ensemble_post(self.u)
        if is_test:
            target_playlists = self.e.get_target_playlists()
            rec.fit(self.URM_train, knn, shrink, weights, k, epochs, normalize, lr)
            return self.rec_and_evaluate(rec, target_playlists)
        else:
            target_playlists = self.u.get_target_playlists()
            rec.fit(self.URM_full, knn, shrink, weights, k, epochs, normalize, lr)
            self.rec_and_save(rec, target_playlists, "predictions/ensemble_post.csv")

    def recommend_ensemble_list(self, is_test, knn=(150, 150, 150, 250), shrink=(10, 10, 5),
                                weights=(1.65, 0.55, 1, 1), epochs=5, lr=0.1):
        rec = Ensemble_list(self.u)
        if is_test:
            target_playlists = self.e.get_target_playlists()
            rec.fit(self.URM_train, knn, shrink, weights, epochs, lr)
            return self.rec_and_evaluate(rec, target_playlists)
        else:
            target_playlists = self.u.get_target_playlists()
            rec.fit(self.URM_full, knn, shrink, weights, epochs, lr)
            self.rec_and_save(rec, target_playlists, "predictions/ensemble_post.csv")

    #OUTDATED!
    def recommend_MFBPR(self, is_test, k=100, epochs=10, sgd_mode='adagrad', lr=0.1):

        if is_test:
            rec = MF_BPR(self.u)
            target_playlists = self.e.get_target_playlists()
            rec.fit(self.URM_train, k, epochs, sgd_mode, lr)
            return self.rec_and_evaluate(rec, target_playlists)
        else:
            rec = MF_BPR(self.u)
            target_playlists = self.u.get_target_playlists()
            rec.fit(self.URM_full, k, epochs, sgd_mode, lr)
            self.rec_and_save(rec, target_playlists, "predictions/MF_BPR.csv")


if __name__ == '__main__':
    run = Recommender(n=0)
    run.recommend_ensemble_post(False, k=100, epochs=1, weights=(0,0,0,0,1))















