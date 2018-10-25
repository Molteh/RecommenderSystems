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
import scipy.sparse as sp


class Recommender(object):

    def __init__(self):
        self.train = pd.read_csv("data/train.csv")
        self.tracks = pd.read_csv("data/tracks.csv")
        self.target_playlists = pd.read_csv("data/target_playlists.csv")
        self.u = Utils(self.train, self.tracks, self.target_playlists)
        self.e = Eval(self.u)
        self.URM_full = self.u.get_URM()
        self.URM_train = self.e.get_URM_train()

    def recommend_itemCBR(self, is_test, knn=300, shrink=300, mode='cosine', normalize=True):
        rec = Item_CBR(self.u)
        if is_test:
            target_playlists = self.e.get_target_playlists()
            rec.fit(self.URM_train, target_playlists, knn, shrink, mode, normalize)
            result = rec.recommend(True)
            self.e.MAP(result, self.e.get_target_tracks())
        else:
            target = self.u.get_target_playlists()
            rec.fit(self.URM_full, target, knn, shrink, mode, normalize)
            result = rec.recommend(False)
            result.to_csv("predictions/item_cbr.csv", index=False)

    def recommend_itemCFR(self, is_test, knn=500, shrink=300, mode='cosine', normalize=True):
        rec = Item_CFR(self.u)
        if is_test:
            target_playlists = self.e.get_target_playlists()
            rec.fit(self.URM_train, target_playlists, knn, shrink, mode, normalize)
            result = rec.recommend(True)
            self.e.MAP(result, self.e.get_target_tracks())
        else:
            target = self.u.get_target_playlists()
            rec.fit(self.URM_full, target, knn, shrink, mode, normalize)
            result = rec.recommend(False)
            result.to_csv("predictions/item_cfr.csv", index=False)

    def recommend_userCFR(self, is_test, knn=400, shrink=200, mode='cosine', normalize=True):
        rec = User_CFR(self.u)
        if is_test:
            target_playlists = self.e.get_target_playlists()
            rec.fit(self.URM_train, target_playlists, knn, shrink, mode, normalize)
            result = rec.recommend(True)
            self.e.MAP(result, self.e.get_target_tracks())
        else:
            target = self.u.get_target_playlists()
            rec.fit(self.URM_full, target, knn, shrink, mode, normalize)
            result = rec.recommend(False)
            result.to_csv("predictions/user_cfr.csv", index=False)

    def recommend_ensemble_item(self, is_test, alfa=0.7, knn1=400, knn2=400, shrink=200, mode='cosine', normalize=True):
        rec = Ensemble_item(self.u)
        if is_test:
            target_playlists = self.e.get_target_playlists()
            rec.fit(self.URM_train, target_playlists, knn1, knn2, shrink, mode, normalize)
            result = rec.recommend(True, alfa)
            self.e.MAP(result, self.e.get_target_tracks())
        else:
            target = self.u.get_target_playlists()
            rec.fit(self.URM_full, target, knn1, knn2, shrink, mode, normalize)
            result = rec.recommend(False, alfa)
            result.to_csv("predictions/item_avg.csv", index=False)

    def recommend_ensemble_cf(self, is_test, alfa=0.6, knn1=400, knn2=400, shrink=200, mode='cosine', normalize=True):
        rec = Ensemble_cf(self.u)
        if is_test:
            target_playlists = self.e.get_target_playlists()
            rec.fit(self.URM_train, target_playlists, knn1, knn2, shrink, mode, normalize)
            result = rec.recommend(True, alfa)
            self.e.MAP(result, self.e.get_target_tracks())
        else:
            target = self.u.get_target_playlists()
            rec.fit(self.URM_full, target, knn1, knn2, shrink, mode, normalize)
            result = rec.recommend(False, alfa)
            result.to_csv("predictions/ensemble_cf.csv", index=False)

    def recommend_ensemble_cfcb(self, is_test, weights=[0.6, 0.4, 0.5], knn1=400, knn2=400, knn3=300, shrink=200, mode='cosine', normalize=True):
        rec = Ensemble_cfcb(self.u)
        if is_test:
            target_playlists = self.e.get_target_playlists()
            rec.fit(self.URM_train, target_playlists, knn1, knn2, knn3, shrink, mode, normalize)
            result = rec.recommend(True, weights)
            self.e.MAP(result, self.e.get_target_tracks())
        else:
            target = self.u.get_target_playlists()
            rec.fit(self.URM_full, target, knn1, knn2, knn3, shrink, mode, normalize)
            result = rec.recommend(False, weights)
            result.to_csv("predictions/ensemble_cfcb.csv", index=False)

    def recommend_hybrid(self, is_test, weights=[0.7, 0.65], knn1=400, knn2=400, knn3=300, shrink=200, mode='cosine', normalize=True):
        rec = Hybrid(self.u)
        if is_test:
            target_playlists = self.e.get_target_playlists()
            rec.fit(self.URM_train, target_playlists, knn1, knn2, knn3, shrink, mode, normalize)
            result = rec.recommend(True, weights)
            self.e.MAP(result, self.e.get_target_tracks())
        else:
            target = self.u.get_target_playlists()
            rec.fit(self.URM_full, target, knn1, knn2, knn3, shrink, mode, normalize)
            result = rec.recommend(False, weights)
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
            BPR_gen = SlimBPR(self.URM_full)
            S_bpr = BPR_gen.get_S_SLIM_BPR(knn)
            target = self.u.get_target_playlists()
            rec.fit(self.URM_full, S_bpr, target, 10000,
                    learning_rate=0.1, epochs=1, positive_item_regularization=1.0,
                    negative_item_regularization=1.0, nzz=1, u=self.u)
            result = rec.recommend(False)
            result.to_csv("predictions/slimBPR.csv", index=False)

    def recommend_ensemble_cfcb_SlimBPR(self, is_test, weights=[0.6, 0.5, 0.5, 0.6], knn1=400, knn2=400, knn3=300, knn4=500, normalize=True):
        rec = Ensemble_cfcb_sbpr(self.u)
        if is_test:
            BPR_gen = SlimBPR_utils(self.URM_train)
            S_bpr = BPR_gen.get_S_SLIM_BPR(knn4)
            target_playlists = self.e.get_target_playlists()
            rec.fit(self.URM_train, S_bpr, target_playlists, knn1, knn2, knn3, normalize)
            result = rec.recommend(True, weights)
            self.e.MAP(result, self.e.get_target_tracks())
        else:
            BPR_gen = SlimBPR_utils(self.URM_full)
            S_bpr = BPR_gen.get_S_SLIM_BPR(knn4)
            target = self.u.get_target_playlists()
            rec.fit(self.URM_full, S_bpr, target, knn1, knn2, knn3, normalize)
            result = rec.recommend(False, weights)
            result.to_csv("predictions/ensemble_cfcb_bpr.csv", index=False)


if __name__ == '__main__':
    run = Recommender()
    run.recommend_ensemble_cfcb_SlimBPR(True, normalize=False)





