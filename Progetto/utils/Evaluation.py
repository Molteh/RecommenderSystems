import pandas as pd
import numpy as np
import scipy.sparse as sp


class Eval(object):

    def __init__(self, u, mask):
        self.mask = mask
        self.URM = u.get_URM()
        self.train_sequential = u.get_train_sequential()
        self.target_playlists = u.get_target_playlists()
        self.split = pd.read_csv("data/split.csv")
        self.URM_train = None
        self.URM_test = None
        self.URM_valid = None
        self.test_playlists = None
        self.validation_playlists = None
        self.build_URM_train2()


    def build_URM_train(self):
        target_seq = list(self.train_sequential['playlist_id'].unique()[:5000])
        target = target_seq
        for length in self.split['length']:
            possible_playlists = [i for i in range(self.URM.shape[0]) if len(
                self.URM.indices[self.URM.indptr[i]:self.URM.indptr[i + 1]]) == length]
            possible_playlists = np.setdiff1d(possible_playlists, target_seq)
            target_random = np.random.choice(possible_playlists,
                                             list(self.split[self.split['length'] == length]['number']), replace=False)
            target = np.concatenate((target, target_random))
        self.URM_train = self.URM.copy().tolil()
        URM_target = sp.lil_matrix(self.URM.shape)

        for idx in target[:5000]:
            length = int(len(self.URM[idx].indices) * 0.2)
            target_songs = np.array(
                self.train_sequential[self.train_sequential['playlist_id'] == idx]['track_id'][-length:])
            self.URM_train[idx, target_songs] = 0
            URM_target[idx, target_songs] = 1

        for idx in target[-5000:]:
            length = int(len(self.URM[idx].indices) * 0.2)
            target_songs = np.random.choice(self.URM[idx].indices, length, replace=False)
            self.URM_train[idx, target_songs] = 0
            URM_target[idx, target_songs] = 1

        target = pd.DataFrame(target, columns=['playlist_id'])

        not_mask = np.setdiff1d(np.arange(10000), self.mask)
        self.test_playlists = target.filter(self.mask, axis=0).sort_index()
        self.validation_playlists = target.filter(not_mask, axis=0)

        self.URM_valid = URM_target.copy().tolil()
        self.URM_valid[self.test_playlists['playlist_id'], :] = 0

        self.URM_test = URM_target.copy().tolil()
        self.URM_test[self.validation_playlists['playlist_id'], :] = 0

        self.URM_test = self.URM_test.tocsr()
        self.URM_valid = self.URM_valid.tocsr()
        self.URM_train = self.URM_train.tocsr()

        assert self.URM_valid.nnz + self.URM_test.nnz + self.URM_train.nnz == self.URM.nnz


    def build_URM_train2(self):
        target = self.target_playlists['playlist_id']
        self.URM_train = self.URM.copy().tolil()
        URM_target = sp.lil_matrix(self.URM.shape)

        for idx in target[:5000]:
            length = int(len(self.URM[idx].indices) * 0.2)
            target_songs = np.array(
                self.train_sequential[self.train_sequential['playlist_id'] == idx]['track_id'][-length:])
            self.URM_train[idx, target_songs] = 0
            URM_target[idx, target_songs] = 1

        for idx in target[-5000:]:
            length = int(len(self.URM[idx].indices) * 0.2)
            target_songs = np.random.choice(self.URM[idx].indices, length, replace=False)
            self.URM_train[idx, target_songs] = 0
            URM_target[idx, target_songs] = 1

        target = pd.DataFrame(target, columns=['playlist_id'])

        '''not_mask = np.setdiff1d(np.arange(10000), self.mask)
        self.test_playlists = target.filter(self.mask, axis=0).sort_index()
        self.validation_playlists = target.filter(not_mask, axis=0)

        self.URM_valid = URM_target.copy().tolil()
        self.URM_valid[self.test_playlists['playlist_id'], :] = 0

        self.URM_test = URM_target.copy().tolil()
        self.URM_test[self.validation_playlists['playlist_id'], :] = 0

        self.URM_test = self.URM_test.tocsr()
        self.URM_valid = self.URM_valid.tocsr()
        self.URM_train = self.URM_train.tocsr()

        assert self.URM_valid.nnz + self.URM_test.nnz + self.URM_train.nnz == self.URM.nnz'''

        self.test_playlists = target
        self.URM_test = URM_target.tocsr()
        self.URM_train = self.URM_train.tocsr()


    def get_URM_train(self):
        return self.URM_train


    @staticmethod
    def AP(recommended_items, relevant_items):
        relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)
        p_at_k = relevant * np.cumsum(relevant, dtype=np.float32) / (1 + np.arange(relevant.shape[0]))
        map_score = np.sum(p_at_k) / np.min([relevant_items.shape[0], relevant.shape[0]])
        return map_score


    def evaluate_algorithm(self, recommender):
        cumulative_MAP = 0.0
        num_eval = 0

        for user_id in self.test_playlists['playlist_id']:

            relevant_items = self.URM_test[user_id].indices

            if len(relevant_items) > 0:
                recommended_items = recommender.recommend(user_id)
                num_eval += 1

                cumulative_MAP += self.AP(recommended_items, relevant_items)

        cumulative_MAP /= num_eval

        print("Recommender performance is: {:.8f}".format(cumulative_MAP))
