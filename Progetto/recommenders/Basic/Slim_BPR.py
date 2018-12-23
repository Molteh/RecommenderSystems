from Progetto.recommenders.SlimBPR.SLIM_BPR_Cython import SLIM_BPR_Cython
from functools import reduce
import scipy.sparse as sp


class Slim_BPR(object):

    def __init__(self, u):
        self.u = u
        self.URM = 0
        self.S_Slim = 0

    def fit(self, URM, knn, epochs, sgd_mode, lr, lower, n_iter):
        self.URM = URM

        '''slim_BPR_Cython = SLIM_BPR_Cython(self.URM)
        total = []

        for i in range(n_iter):
            slim_BPR_Cython.fit(epochs=epochs, sgd_mode=sgd_mode, stop_on_validation=True, learning_rate=lr, topK=knn,
                                evaluator_object=None, lower_validatons_allowed=lower)
            total.append(slim_BPR_Cython.W_sparse)

        self.S_Slim = reduce(lambda a, b: a + b, total) / n_iter
        sp.save_npz("./s_slim.npz", self.S_Slim)'''
        self.S_Slim = sp.load_npz("./s_slim.npz")


    def recommend(self, target_playlist):
        row_slim = (self.URM[target_playlist].dot(self.S_Slim)).toarray().ravel()
        return self.u.get_top_10(self.URM, target_playlist, row_slim)
