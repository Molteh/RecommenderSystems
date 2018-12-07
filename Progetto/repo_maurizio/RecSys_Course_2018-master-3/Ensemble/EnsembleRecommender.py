from Base.Recommender import Recommender
from Base.Recommender_utils import check_matrix
from Base.SimilarityMatrixRecommender import SimilarityMatrixRecommender
from KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from KNN.UserKNNCFRecommender import UserKNNCFRecommender

class EnsembleRecommender(SimilarityMatrixRecommender, Recommender):

    RECOMMENDER_NAME = "EnsembleRecommender"

    def __init__(self,  URM_train, sparse_weights=True):
        super(EnsembleRecommender, self).__init__()

        # CSR is faster during evaluation
        self.URM_train = check_matrix(URM_train, 'csr')

        self.sparse_weights = sparse_weights

        self.compute_item_score = self.compute_score_ensemble_based

    def fit(self, alpha=1.65, beta=0.55, tfidf1=True, tfidf2=True):

        '''ItemCFRecommender'''
        ICFRec = ItemKNNCFRecommender(self.URM_train)
        ICFRec.fit(tfidf=tfidf1)
        self.S_ICF = ICFRec.W_sparse
        self.S_ICF = alpha * self.S_ICF

        '''UserCFRecommender'''
        UCFRec = UserKNNCFRecommender(self.URM_train)
        UCFRec.fit(tfidf=tfidf2)
        self.S_UCF = UCFRec.W_sparse
        self.S_UCF = beta * self.S_UCF

