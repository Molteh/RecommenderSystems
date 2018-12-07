
from SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from SLIM_ElasticNet.SLIMElasticNetRecommender import SLIMElasticNetRecommender

from MatrixFactorization.Cython.MatrixFactorization_Cython import MatrixFactorization_BPR_Cython, MatrixFactorization_FunkSVD_Cython
from MatrixFactorization.PureSVD import PureSVDRecommender

from Base.NonPersonalizedRecommender import TopPop, Random

from KNN.UserKNNCFRecommender import UserKNNCFRecommender
from KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender
from GraphBased.RP3betaRecommender import RP3betaRecommender
from GraphBased.P3alphaRecommender import P3alphaRecommender
from Ensemble.EnsembleRecommender import EnsembleRecommender

from data.Movielens_10M.Movielens10MReader import Movielens10MReader

import pandas as pd
import numpy as np

import traceback, os
from Utils.MatrixBuilder import Utils
from Utils.Evaluation import Eval


if __name__ == '__main__':


    #dataReader = Movielens10MReader()

    #URM_train = dataReader.get_URM_train()
    #URM_validation = dataReader.get_URM_validation()
    #URM_test = dataReader.get_URM_test()

    train = pd.read_csv("data/train.csv")
    tracks = pd.read_csv("data/tracks.csv")
    target_playlists = pd.read_csv("data/target_playlists.csv")
    train_sequential = pd.read_csv("data/train_sequential.csv")
    u = Utils(train, tracks, target_playlists, train_sequential)
    e = Eval(u, (np.random.choice(range(10000), 5000, replace=False)).tolist())
    e.splitTestValidation()
    ICM = u.get_ICM(tfidf=True)
    URM_test = e.get_URM_test()
    URM_train = e.get_URM_train()
    URM_validation = e.get_URM_validation()

    from Base.Evaluation.Evaluator import SequentialEvaluator
    evaluator = SequentialEvaluator(URM_test, [5, 10, 20], exclude_seen=True)

    '''ICFR = ItemKNNCFRecommender(URM_train)
    ICFR.fit(topK=150, shrink=10, tfidf=True)
    results_run, results_run_string = evaluator.evaluateRecommender(ICFR)
    print("Algorithm: {}, results: \n{}".format(ICFR.__class__, results_run_string))

    ICFR3 = ItemKNNCFRecommender(URM_train)
    ICFR3.fit(topK=150, shrink=10, tfidf=False)
    results_run, results_run_string = evaluator.evaluateRecommender(ICFR3)
    print("Algorithm: {}, results: \n{}".format(ICFR3.__class__, results_run_string))'''

    '''UCFR = UserKNNCFRecommender(URM_train)
    UCFR.fit()
    results_run, results_run_string = evaluator.evaluateRecommender(UCFR)
    print("Algorithm: {}, results: \n{}".format(UCFR.__class__, results_run_string))

    UCFR3 = UserKNNCFRecommender(URM_train)
    UCFR3.fit(topK=150, shrink=10, tfidf=False)
    results_run, results_run_string = evaluator.evaluateRecommender(UCFR3)
    print("Algorithm: {}, results: \n{}".format(UCFR3.__class__, results_run_string))'''

    Ens = EnsembleRecommender(URM_train)
    Ens.fit(tfidf1=True, tfidf2=True)
    results_run, results_run_string = evaluator.evaluateRecommender(Ens)
    print("Algorithm: {}, results: \n{}".format(Ens.__class__, results_run_string))

    Ens = EnsembleRecommender(URM_train)
    Ens.fit(tfidf1=True, tfidf2=False)
    results_run, results_run_string = evaluator.evaluateRecommender(Ens)
    print("Algorithm: {}, results: \n{}".format(Ens.__class__, results_run_string))

    Ens = EnsembleRecommender(URM_train)
    Ens.fit(tfidf1=False, tfidf2=True)
    results_run, results_run_string = evaluator.evaluateRecommender(Ens)
    print("Algorithm: {}, results: \n{}".format(Ens.__class__, results_run_string))

    Ens = EnsembleRecommender(URM_train)
    Ens.fit(tfidf1=False, tfidf2=False)
    results_run, results_run_string = evaluator.evaluateRecommender(Ens)
    print("Algorithm: {}, results: \n{}".format(Ens.__class__, results_run_string))

    recommender_list = [
        #Random,
        #TopPop,
        #P3alphaRecommender,
        #RP3betaRecommender,
        #ItemKNNCFRecommender,
        EnsembleRecommender,
        #ItemKNNCBFRecommender,
        UserKNNCFRecommender,
        #MatrixFactorization_BPR_Cython,
        #MatrixFactorization_FunkSVD_Cython,
        #PureSVDRecommender,
        #SLIM_BPR_Cython,
        #SLIMElasticNetRecommender
        ]


    from Base.Evaluation.Evaluator import SequentialEvaluator

    evaluator = SequentialEvaluator(URM_test, [5, 10, 20], exclude_seen=True)


    output_root_path = "result_experiments/"

    # If directory does not exist, create
    if not os.path.exists(output_root_path):
        os.makedirs(output_root_path)


    logFile = open(output_root_path + "result_all_algorithms.txt", "a")

    print("I'm here")


    for recommender_class in recommender_list:

        try:

            print("Algorithm: {}".format(recommender_class))


            recommender = recommender_class(URM_train)
            recommender.fit()

            results_run, results_run_string = evaluator.evaluateRecommender(recommender)

            print("Algorithm: {}, results: \n{}".format(recommender.__class__, results_run_string))
            logFile.write("Algorithm: {}, results: \n{}\n".format(recommender.__class__, results_run_string))
            logFile.flush()

        except Exception as e:
            traceback.print_exc()
            logFile.write("Algorithm: {} - Exception: {}\n".format(recommender_class, str(e)))
            logFile.flush()
