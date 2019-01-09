from Progetto.utils.MatrixBuilder import Utils
from Progetto.utils.Evaluation import Eval
from Progetto.recommenders.Basic.ALS import ALS
from Progetto.recommenders.Basic.Pure_SVD import PureSVD
from Progetto.recommenders.Basic.ICM_SVD import ItemSVD
from Progetto.recommenders.Basic.Item_CFR import Item_CFR
from Progetto.recommenders.Basic.Item_CBR import Item_CBR
from Progetto.recommenders.Basic.User_CFR import User_CFR
from Progetto.recommenders.Basic.P3Beta import P3Beta_R
from Progetto.recommenders.Basic.Slim_BPR import Slim_BPR
from Progetto.recommenders.Basic.Slim_Elastic import Slim_Elastic
from Progetto.recommenders.Ensemble_longshort import Ensemble_longshort
import pandas as pd


class Recommender(object):

    def __init__(self, holdout="standard"):
        self.train = pd.read_csv("data/train.csv")
        self.tracks = pd.read_csv("data/tracks.csv")
        self.target_playlists = pd.read_csv("data/target_playlists.csv")
        self.train_sequential = pd.read_csv("data/train_sequential.csv")
        self.u = Utils(self.train, self.tracks, self.target_playlists, self.train_sequential)
        self.e = Eval(self.u, holdout)
        self.URM_full = self.u.URM
        self.URM_train = self.e.URM_train

    def generate_result(self, recommender, path, is_test=True, longshort=False):
        if is_test:
            if longshort:
                return self.e.evaluate_algorithm_longshort(recommender)
            else:
                return self.e.evaluate_algorithm(recommender)
        else:
            if longshort:
                return self.e.generate_predictions_longshort(recommender, path)
            else:
                return self.e.generate_predictions(recommender, path)

    def recommend_itemCBR(self, knn=150, shrink=5, normalize=True, similarity='cosine', tfidf=True):
        rec = Item_CBR(self.u)
        rec.fit(self.URM_train, knn, shrink, normalize, similarity, tfidf)
        return self.generate_result(rec, None)

    def recommend_itemCFR(self, knn=150, shrink=10, normalize=True, similarity='cosine', tfidf=True):
        rec = Item_CFR(self.u)
        rec.fit(self.URM_train, knn, shrink, normalize, similarity, tfidf)
        return self.generate_result(rec, None)

    def recommend_userCFR(self, knn=150, shrink=10, normalize=True, similarity='cosine', tfidf=True):
        rec = User_CFR(self.u)
        rec.fit(self.URM_train, knn, shrink, normalize, similarity, tfidf)
        return self.generate_result(rec, None)

    def recommend_SlimBPR(self, knn=250, epochs=15, sgd_mode='adagrad', lr=0.1, lower=5, n_iter=10):
        rec = Slim_BPR(self.u)
        rec.fit(self.URM_full, knn, epochs, sgd_mode, lr, lower, n_iter)
        return self.generate_result(rec, "./predictions/slim_bpr", is_test=False)

    def recommend_SlimElastic(self, knn=250, l1=1, po=True):
        rec = Slim_Elastic(self.u)
        rec.fit(self.URM_train, knn, l1, po)
        return self.generate_result(rec, None)

    def recommend_PureSVD(self, k=800, n_iter=1, random_state=False, bm25=True, K1=2, B=0.9):
        rec = PureSVD(self.u)
        rec.fit(self.URM_train, k, n_iter, random_state, bm25, K1, B)
        return self.generate_result(rec, None)

    def recommend_ALS(self, k=50, n_iter=1, reg=0.015, mode="linear"):
        rec = ALS(self.u)
        rec.fit(self.URM_train, k, n_iter, reg, mode)
        return self.generate_result(rec, None)

    def recommend_ItemSVD(self, k=300, knn=150, evaluate=False):
        rec = ItemSVD(self.u)
        rec.fit(self.URM_train, k, knn, evaluate)
        return self.generate_result(rec, None)

    def recommend_P3B(self, knn=100, alfa=0.7, beta=0.3):
        rec = P3Beta_R(self.u)
        rec.fit(self.URM_full, knn, alfa, beta)
        return self.generate_result(rec, path=None)

    def recommend_ensemble_longshort(self, is_test=True, knn=(150,150,50,250,100,50), shrink=(10, 10, 10),
                                     weights=(1, 1, 1, 0, 1, 1, 1, 0)):
        rec = Ensemble_longshort(self.u)
        if is_test:
            rec.fit(self.URM_train, knn, shrink, weights)
        else:
            rec.fit(self.URM_full, knn, shrink, weights)
        return self.generate_result(rec, "./predictions/ensemble_longshort.csv", is_test, longshort=True)


if __name__ == '__main__':
    run = Recommender()
    run.recommend_ensemble_longshort(is_test=False)








    







