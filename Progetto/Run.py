from Progetto.utils.MatrixBuilder import Utils
from Progetto.utils.Evaluation import Eval
from Progetto.recommenders.Item_CFR import Item_CFR
from Progetto.recommenders.User_CFR import User_CFR
from Progetto.recommenders.Item_CBR import Item_CBR
from Progetto.recommenders.Ensemble_cfcb import Ensemble_cfcb
from Progetto.recommenders.Ensemble_cfcb2 import Ensemble_cfcb2
from Progetto.recommenders.Ensemble_cfcb3 import Ensemble_cfcb3
from Progetto.recommenders.Slim_BPR import Slim_BPR
from Progetto.recommenders.Ensemble_cfcb_sbpr import Ensemble_cfcb_sbpr
from Progetto.recommenders.Slim_BPR_Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
import pandas as pd
import numpy as np
from tqdm import tqdm


class Recommender(object):

    def __init__(self, n=5):
        self.train = pd.read_csv("data/train.csv")
        self.tracks = pd.read_csv("data/tracks.csv")
        self.target_playlists = pd.read_csv("data/target_playlists.csv")
        self.train_sequential = pd.read_csv("data/train_sequential.csv")
        self.u = Utils(self.train, self.tracks, self.target_playlists, self.train_sequential)
        self.e = Eval(self.u)
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

    def recommend_itemCBR(self, is_test, knn=150, shrink=10, alfa=1, cython=True):
        rec = Item_CBR(self.u)
        if is_test:
            target_playlists = self.e.get_target_playlists()
            rec.fit(self.URM_train, knn, shrink, alfa, cython)
            return self.rec_and_evaluate(rec, target_playlists)
        else:
            target_playlists = self.u.get_target_playlists()
            rec.fit(self.URM_full, knn, shrink, alfa, cython)
            self.rec_and_save(rec, target_playlists, "predictions/item_cbr.csv")

    def recommend_itemCFR(self, is_test, knn=250, shrink=10, cython=True):
        rec = Item_CFR(self.u)
        if is_test:
            target_playlists = self.e.get_target_playlists()
            rec.fit(self.URM_train, knn, shrink, cython)
            return self.rec_and_evaluate(rec, target_playlists)
        else:
            target_playlists = self.u.get_target_playlists()
            rec.fit(self.URM_full, knn, shrink, cython)
            self.rec_and_save(rec, target_playlists, "predictions/item_cfr.csv")

    def recommend_userCFR(self, is_test, knn=250, shrink=10, cython=True):
        rec = User_CFR(self.u)
        if is_test:
            target_playlists = self.e.get_target_playlists()
            rec.fit(self.URM_train, knn, shrink, cython)
            return self.rec_and_evaluate(rec, target_playlists)
        else:
            target_playlists = self.u.get_target_playlists()
            rec.fit(self.URM_full, knn, shrink, cython)
            self.rec_and_save(rec, target_playlists, "predictions/user_cfr1.csv")

    def recommend_slimBPR(self, is_test, knn=800):
        rec = Slim_BPR()
        if is_test:
            target_playlists = self.e.get_target_playlists()
            rec.fit(self.URM_train, 10000,
                    learning_rate=0.1, epochs=1, positive_item_regularization=1.0,
                    negative_item_regularization=1.0, nzz=1, u=self.u, knn=knn)
            return self.rec_and_evaluate(rec, target_playlists)
        else:
            target_playlists = self.u.get_target_playlists()
            rec.fit(self.URM_full, 10000,
                    learning_rate=0.1, epochs=1, positive_item_regularization=1.0,
                    negative_item_regularization=1.0, nzz=1, u=self.u, knn=knn)
            self.rec_and_save(rec, target_playlists, "predictions/slimBPR.csv")

    def recommend_ensemble_cfcb(self, is_test, weights=(1.5, 0.4), knn1=150, knn2=150, knn3=150, shrink=10, alfa=1, cython=True):
        rec = Ensemble_cfcb(self.u)
        if is_test:
            target_playlists = self.e.get_target_playlists()
            rec.fit(self.URM_train, knn1, knn2, knn3, shrink, weights, alfa, cython)
            return self.rec_and_evaluate(rec, target_playlists)
        else:
            target_playlists = self.u.get_target_playlists()
            rec.fit(self.URM_full, knn1, knn2, knn3, shrink, weights, alfa, cython)
            self.rec_and_save(rec, target_playlists, "predictions/ensemble_cfcb.csv")

    def recommend_ensemble_cfcb2(self, is_test, weights=(1.5, 0.4), knn1=150, knn2=150, knn3=150, shrink=10, alfa=1, cython=True):
        rec = Ensemble_cfcb2(self.u)
        if is_test:
            target_playlists = self.e.get_target_playlists()
            rec.fit(self.URM_train, knn1, knn2, knn3, shrink, weights, alfa, cython)
            return self.rec_and_evaluate(rec, target_playlists)
        else:
            target_playlists = self.u.get_target_playlists()
            rec.fit(self.URM_full, knn1, knn2, knn3, shrink, weights, alfa, cython)
            self.rec_and_save(rec, target_playlists, "predictions/hybrid.csv")

    def recommend_ensemble_cfcb3(self, is_test, weights=(1.5, 0.4, 1), knn1=150, knn2=150, knn3=150, shrink=10, alfa=1,
                                 cython=True):
        rec = Ensemble_cfcb3(self.u)
        if is_test:
            target_playlists = self.e.get_target_playlists()
            rec.fit(self.URM_train, knn1, knn2, knn3, shrink, weights, alfa, cython)
            return self.rec_and_evaluate(rec, target_playlists)
        else:
            target_playlists = self.u.get_target_playlists()
            rec.fit(self.URM_full, knn1, knn2, knn3, shrink, weights, alfa, cython)
            self.rec_and_save(rec, target_playlists, "predictions/ensemble_cfcb3.csv")

    def recommend_ensemble_cfcb_SlimBPR(self, is_test, weights=(0, 0.4, 0, 0.5), knn1=150, knn2=150, knn3=150,
                                        knn4=500, shrink=10, cython=True):
        rec = Ensemble_cfcb_sbpr(self.u)
        if is_test:
            target_playlists = self.e.get_target_playlists()
            rec.fit(self.URM_train, knn1, knn2, knn3, knn4, shrink, weights, cython)
            return self.rec_and_evaluate(rec, target_playlists)
        else:
            target_playlists = self.u.get_target_playlists()
            rec.fit(self.URM_full, knn1, knn2, knn3, knn4, shrink, weights, cython)
            self.rec_and_save(rec, target_playlists, "predictions/ensemble_cfcb_bpr.csv")

    def recommend_Cython_SlimBPR(self, is_test, recompile=False, epochs=2, val_every_n_ep=1, sgd_mode='rmsprop',
                                 learning_rate=1e-4):
        if is_test:
            rec = SLIM_BPR_Cython(self.URM_train, recompile_cython=recompile, positive_threshold=0, sparse_weights=True)
            target_playlists = self.e.get_target_playlists()
            rec.fit(epochs=epochs, validate_every_N_epochs=val_every_n_ep, batch_size=1, sgd_mode=sgd_mode, learning_rate=learning_rate)
            return self.rec_and_evaluate(rec, target_playlists)
        else:
            rec = SLIM_BPR_Cython(self.URM_full, recompile_cython=recompile, positive_threshold=0, sparse_weights=True)
            target_playlists = self.u.get_target_playlists()
            rec.fit(epochs=epochs, validate_every_N_epochs=val_every_n_ep, batch_size=1, sgd_mode=sgd_mode, learning_rate=learning_rate)
            self.rec_and_save(rec, target_playlists, "predictions/cython_Slim_BPR.csv")


if __name__ == '__main__':
    run = Recommender(n=5)
    run.recommend_ensemble_cfcb3(False)








