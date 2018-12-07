#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 23/10/17

@author: Maurizio Ferrari Dacrema
"""

import scipy.sparse as sp
from Base.Recommender import Recommender
from Base.Recommender_utils import check_matrix
from Base.SimilarityMatrixRecommender import SimilarityMatrixRecommender
from sklearn.feature_extraction.text import TfidfTransformer


from Base.Similarity.Compute_Similarity import Compute_Similarity


class ItemKNNCFRecommender(SimilarityMatrixRecommender, Recommender):
    """ ItemKNN recommender"""

    RECOMMENDER_NAME = "ItemKNNCFRecommender"

    def __init__(self, URM_train, sparse_weights=True):
        super(ItemKNNCFRecommender, self).__init__()

        # CSR is faster during evaluation
        self.URM_train = check_matrix(URM_train, 'csr')

        self.dataset = None

        self.sparse_weights = sparse_weights

    def fit(self, topK=200, shrink=20, similarity='cosine', normalize=True, tfidf=True, **similarity_args):

        self.topK = topK
        self.shrink = shrink


        if not tfidf:
            self.URM_train = self.URM_train
        else:
            self.URM_train = TfidfTransformer().fit_transform(self.URM_train.T).T.tocsr()

        similarity = Compute_Similarity(self.URM_train, shrink=shrink, topK=topK, normalize=normalize, similarity = similarity, **similarity_args)


        if self.sparse_weights:
            self.W_sparse = similarity.compute_similarity()
        else:
            self.W = similarity.compute_similarity()
            self.W = self.W.toarray()

        sp.save_npz("ICFMatrix.npz", self.W_sparse)

