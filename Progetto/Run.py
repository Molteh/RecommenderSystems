from Progetto.utils.MatrixBuilder import Utils
from Progetto.utils.Evaluation import Eval
from Progetto.utils.SlimBPR_utils import SlimBPR_utils
from Progetto.recommenders.Item_CFR import Item_CFR
from Progetto.recommenders.User_CFR import User_CFR
from Progetto.recommenders.Item_CBR import Item_CBR
from Progetto.recommenders.Ensemble_cfcb import Ensemble_cfcb
from Progetto.recommenders.Ensemble_item import Ensemble_item
from Progetto.recommenders.Hybrid import Hybrid
from Progetto.recommenders.SlimBPR import SlimBPR
from Progetto.recommenders.Ensemble_cf import Ensemble_cf
from Progetto.recommenders.Ensemble_cfcb_sbpr import Ensemble_cfcb_sbpr
import pandas as pd
import numpy as np
from tqdm import tqdm


class Recommender(object):

    def __init__(self):
        self.train = pd.read_csv("data/train.csv")
        self.tracks = pd.read_csv("data/tracks.csv")
        self.target_playlists = pd.read_csv("data/target_playlists.csv")
        self.u = Utils(self.train, self.tracks, self.target_playlists)
        self.e = Eval(self.u)
        self.URM_full = self.u.get_URM()
        self.URM_train = self.e.get_URM_train()

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

    def recommend_itemCBR(self, is_test, knn=150, shrink=10, mode='cosine', normalize=True):
        rec = Item_CBR(self.u)
        if is_test:
            target_playlists = self.e.get_target_playlists()
            rec.fit(self.URM_train, target_playlists, knn, shrink, mode, normalize)
            result = self.evaluate(rec, True, target_playlists)
            self.e.MAP(result, self.e.get_target_tracks())
        else:
            target_playlists = self.u.get_target_playlists()
            rec.fit(self.URM_full, target_playlists, knn, shrink, mode, normalize)
            result = self.evaluate(rec, False, target_playlists)
            result.to_csv("predictions/item_cbr.csv", index=False)

    def recommend_itemCFR(self, is_test, knn=150, shrink=10, mode='cosine', normalize=True):
        rec = Item_CFR(self.u)
        if is_test:
            target_playlists = self.e.get_target_playlists()
            rec.fit(self.URM_train, target_playlists, knn, shrink, mode, normalize)
            result = self.evaluate(rec, True, target_playlists)
            self.e.MAP(result, self.e.get_target_tracks())
        else:
            target_playlists = self.u.get_target_playlists()
            rec.fit(self.URM_full, target_playlists, knn, shrink, mode, normalize)
            result = self.evaluate(rec, False, target_playlists)
            result.to_csv("predictions/item_cfr.csv", index=False)

    def recommend_userCFR(self, is_test, knn=250, shrink=10, mode='cosine', normalize=True):
        rec = User_CFR(self.u)
        if is_test:
            target_playlists = self.e.get_target_playlists()
            rec.fit(self.URM_train, target_playlists, knn, shrink, mode, normalize)
            result = self.evaluate(rec, True, target_playlists)
            self.e.MAP(result, self.e.get_target_tracks())
        else:
            target_playlists = self.u.get_target_playlists()
            rec.fit(self.URM_full, target_playlists, knn, shrink, mode, normalize)
            result = self.evaluate(rec, False, target_playlists)
            result.to_csv("predictions/user_cfr1.csv", index=False)

    def recommend_ensemble_item(self, is_test, alfa=0.6, knn1=150, knn2=250, shrink=10, mode='cosine', normalize=True):
        rec = Ensemble_item(self.u)
        if is_test:
            target_playlists = self.e.get_target_playlists()
            rec.fit(self.URM_train, target_playlists, knn1, knn2, shrink, mode, normalize, alfa)
            result = self.evaluate(rec, True, target_playlists)
            self.e.MAP(result, self.e.get_target_tracks())
        else:
            target_playlists = self.u.get_target_playlists()
            rec.fit(self.URM_full, target_playlists, knn1, knn2, shrink, mode, normalize, alfa)
            result = self.evaluate(rec, False, target_playlists)
            result.to_csv("predictions/ensemble_item.csv", index=False)

    def recommend_ensemble_cf(self, is_test, alfa=0.6, knn1=150, knn2=150, shrink=10, mode='cosine', normalize=True):
        rec = Ensemble_cf(self.u)
        if is_test:
            target_playlists = self.e.get_target_playlists()
            rec.fit(self.URM_train, target_playlists, knn1, knn2, shrink, mode, normalize, alfa)
            result = self.evaluate(rec, True, target_playlists)
            self.e.MAP(result, self.e.get_target_tracks())
        else:
            target_playlists = self.u.get_target_playlists()
            rec.fit(self.URM_full, target_playlists, knn1, knn2, shrink, mode, normalize, alfa)
            result = self.evaluate(rec, False, target_playlists)
            result.to_csv("predictions/ensemble_cf.csv", index=False)

    def recommend_ensemble_cfcb(self, is_test, weights=[0.6, 0.4, 0.5], knn1=150, knn2=150, knn3=200, shrink=10,
                                mode='cosine', normalize=True):
        rec = Ensemble_cfcb(self.u)
        if is_test:
            target_playlists = self.e.get_target_playlists()
            rec.fit(self.URM_train, target_playlists, knn1, knn2, knn3, shrink, mode, normalize, weights)
            result = self.evaluate(rec, True, target_playlists)
            self.e.MAP(result, self.e.get_target_tracks())
        else:
            target_playlists = self.u.get_target_playlists()
            rec.fit(self.URM_full, target_playlists, knn1, knn2, knn3, shrink, mode, normalize, weights)
            result = self.evaluate(rec, False, target_playlists)
            result.to_csv("predictions/ensemble_cfcb.csv", index=False)

    def recommend_hybrid(self, is_test, weights=[0.5, 0.7], knn1=150, knn2=150, knn3=200, shrink=10, mode='cosine',
                         normalize=True):
        rec = Hybrid(self.u)
        if is_test:
            target_playlists = self.e.get_target_playlists()
            rec.fit(self.URM_train, target_playlists, knn1, knn2, knn3, shrink, mode, normalize, weights)
            result = self.evaluate(rec, True, target_playlists)
            self.e.MAP(result, self.e.get_target_tracks())
        else:
            target_playlists = self.u.get_target_playlists()
            rec.fit(self.URM_full, target_playlists, knn1, knn2, knn3, shrink, mode, normalize, weights)
            result = self.evaluate(rec, False, target_playlists)
            result.to_csv("predictions/hybrid.csv", index=False)

    def recommend_slimBPR(self, is_test, knn=100):
        rec = SlimBPR()
        if is_test:
            BPR_gen = SlimBPR_utils(self.URM_train)
            S_bpr = BPR_gen.get_S_SLIM_BPR(knn)
            target_playlists = self.e.get_target_playlists()
            rec.fit(self.URM_train, S_bpr, target_playlists, 10000,
                    learning_rate=0.1, epochs=1, positive_item_regularization=1.0,
                    negative_item_regularization=1.0, nzz=1, u=self.u)
            result = rec.recommend(True)
            self.e.MAP(result, self.e.get_target_tracks())
        else:
            BPR_gen = SlimBPR_utils(self.URM_full)
            S_bpr = BPR_gen.get_S_SLIM_BPR(knn)
            target = self.u.get_target_playlists()
            rec.fit(self.URM_full, S_bpr, target, 10000,
                    learning_rate=0.1, epochs=1, positive_item_regularization=1.0,
                    negative_item_regularization=1.0, nzz=1, u=self.u)
            result = rec.recommend(False)
            result.to_csv("predictions/slimBPR.csv", index=False)

    def recommend_ensemble_cfcb_SlimBPR(self, is_test, weights=[0.6, 0.5, 0.5, 0.6], knn1=150, knn2=150, knn3=200,
                                        knn4=800, shrink=10, mode='cosine', normalize=True):
        rec = Ensemble_cfcb_sbpr(self.u)
        if is_test:
            BPR_gen = SlimBPR_utils(self.URM_train)
            S_bpr = BPR_gen.get_S_SLIM_BPR(knn4)
            target_playlists = self.e.get_target_playlists()
            rec.fit(self.URM_train, S_bpr, target_playlists, knn1, knn2, knn3, shrink, mode, normalize, weights)
            result = self.evaluate(rec, True, target_playlists)
            self.e.MAP(result, self.e.get_target_tracks())
        else:
            BPR_gen = SlimBPR_utils(self.URM_full)
            S_bpr = BPR_gen.get_S_SLIM_BPR(knn4)
            target_playlists = self.e.get_target_playlists()
            rec.fit(self.URM_train, S_bpr, target_playlists, knn1, knn2, knn3, shrink, mode, normalize, weights)
            result = self.evaluate(rec, False, target_playlists)
            result.to_csv("predictions/ensemble_cfcb_bpr.csv", index=False)



if __name__ == '__main__':
    run = Recommender()
    run.recommend_hybrid(True, weights=[0.5, 0.7])






