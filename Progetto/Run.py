from Progetto.utils.MatrixBuilder import Utils
from Progetto.utils.Evaluation import Eval
from Progetto.recommenders.Ensemble_post import Ensemble_post
from Progetto.recommenders.Basic.MF import MF_BPR
from Progetto.recommenders.Basic.Pure_SVD import PureSVD
from Progetto.recommenders.SlimBPR.MyEvaluator import MyEvaluator
from Progetto.recommenders.Basic.Item_CFR import Item_CFR
from Progetto.recommenders.Basic.Item_CBR import Item_CBR
from Progetto.recommenders.Basic.User_CFR import User_CFR
from Progetto.recommenders.Basic.Slim_BPR import Slim_BPR
from Progetto.recommenders.Basic.Slim_BPR_U import Slim_BPR_U
import pandas as pd
import numpy as np
from tqdm import tqdm


class Recommender(object):

    def __init__(self):
        self.train = pd.read_csv("data/train.csv")
        self.tracks = pd.read_csv("data/tracks.csv")
        self.target_playlists = pd.read_csv("data/target_playlists.csv")
        self.train_sequential = pd.read_csv("data/train_sequential.csv")
        self.u = Utils(self.train, self.tracks, self.target_playlists, self.train_sequential)
        self.e = Eval(self.u, (np.random.choice(range(10000), 5000, replace=False)).tolist())
        self.URM_full = self.u.get_URM()
        self.URM_train = self.e.get_URM_train()
        self.myEvaluator = MyEvaluator(URM=self.URM_train, norm=False, e=self.e)

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

    def applyReduction(self, n=0):
        self.URM_full = self.preprocess_URM(self.u.get_URM(), self.target_playlists, n)
        self.URM_train = self.preprocess_URM(self.e.get_URM_train(), self.e.get_target_playlists(), n)

    def rec_and_evaluate(self, rec, target_playlists):
        result = self.evaluate(rec, True, target_playlists)
        return self.e.MAP(result, self.e.get_target_tracks())

    def rec_and_save(self, rec, target_playlists, path):
        result = self.evaluate(rec, False, target_playlists)
        result.to_csv(path, index=False)

    def recommend_itemCBR(self, knn=150, shrink=5, normalize=True, similarity='cosine', tfidf=True):
        rec = Item_CBR(self.u)
        target_playlists = self.e.get_target_playlists()
        rec.fit(self.URM_train, knn, shrink, normalize, similarity, tfidf)
        return self.rec_and_evaluate(rec, target_playlists)

    def recommend_itemCFR(self, knn=150, shrink=10, normalize=True, similarity='cosine', tfidf=True):
        rec = Item_CFR(self.u)
        target_playlists = self.e.get_target_playlists()
        rec.fit(self.URM_train, knn, shrink, normalize, similarity, tfidf)
        return self.rec_and_evaluate(rec, target_playlists)

    def recommend_userCFR(self, knn=150, shrink=10, normalize=True, similarity='cosine', tfidf=True):
        rec = User_CFR(self.u)
        target_playlists = self.e.get_target_playlists()
        rec.fit(self.URM_train, knn, shrink, normalize, similarity, tfidf)
        return self.rec_and_evaluate(rec, target_playlists)

    def recommend_SlimBPR(self, knn=250, epochs=150, sgd_mode='adagrad', lr=0.1, lower=5, es=False):
        rec = Slim_BPR(self.u)
        if es:
            ev = self.myEvaluator
        else:
            ev = None
        target_playlists = self.e.get_target_playlists()
        rec.fit(self.URM_train, knn, epochs, sgd_mode, lr, lower, ev)
        return self.rec_and_evaluate(rec, target_playlists)

    def recommend_SlimBPR_U(self, knn=250, epochs=150, sgd_mode='adagrad', lr=0.1, lower=5, es=False):
        rec = Slim_BPR_U(self.u)
        if es:
            ev = self.myEvaluator
        else:
            ev = None
        target_playlists = self.e.get_target_playlists()
        rec.fit(self.URM_train, knn, epochs, sgd_mode, lr, lower, ev)
        return self.rec_and_evaluate(rec, target_playlists)

    def recommend_MF(self, k=50, epochs=150, sgd_mode='sgd', lr=0.001):
        rec = MF_BPR(self.u)
        target_playlists = self.e.get_target_playlists()
        rec.fit(self.URM_train, k, epochs, sgd_mode, lr)
        return self.rec_and_evaluate(rec, target_playlists)

    def recommend_PureSVD(self, k=700, n_iter=1, random_state=False):
        rec = PureSVD(self.u)
        target_playlists = self.e.get_target_playlists()
        rec.fit(self.URM_train, k, n_iter, random_state)
        return self.rec_and_evaluate(rec, target_playlists)

    def recommend_ensemble_post(self, is_test, knn=(150, 150, 150, 250, 250), shrink=(10, 10, 5),
                                weights=(1.65, 0.55, 1, 0.1, 0.005), k=500, epochs=5,
                                lr=0.1, sgd_mode='adagrad', lower=5, es=True):
        rec = Ensemble_post(self.u)
        if is_test:
            if es:
                ev = self.myEvaluator
            else:
                ev = None
            target_playlists = self.e.get_target_playlists()
            rec.fit(self.URM_train, knn, shrink, weights, k, epochs, lr, sgd_mode, lower, ev)
            return self.rec_and_evaluate(rec, target_playlists)
        else:
            target_playlists = self.u.get_target_playlists()
            rec.fit(self.URM_full, knn, shrink, weights, k, epochs, lr, sgd_mode, lower, None)
            self.rec_and_save(rec, target_playlists, "predictions/ensemble_post.csv")


if __name__ == '__main__':
    run = Recommender()
    #run.recommend_itemCFR()



















