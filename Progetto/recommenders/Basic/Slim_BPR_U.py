from Progetto.recommenders.SlimBPR.SLIM_BPR_Cython import SLIM_BPR_Cython


class Slim_BPR_U(object):

    def __init__(self, u):
        self.u = u
        self.URM = 0
        self.S_Slim = 0

    def fit(self, URM, knn, epochs, sgd_mode, lr, lower, evaluator):
        self.URM = URM

        slim_BPR_Cython = SLIM_BPR_Cython(self.URM.T)
        slim_BPR_Cython.fit(epochs=epochs, sgd_mode=sgd_mode, stop_on_validation=True, learning_rate=lr, topK=knn,
                            evaluator_object=evaluator, lower_validatons_allowed=lower)
        self.S_Slim = slim_BPR_Cython.W_sparse

    def recommend(self, target_playlist):
        row_slim = (self.S_Slim[target_playlist].dot(self.URM)).toarray().ravel()
        return self.u.get_top_10(self.URM, target_playlist, row_slim)
