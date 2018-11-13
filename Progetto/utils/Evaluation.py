import pandas as pd
import numpy as np


class Eval(object):

    def __init__(self, u, target_playlists):
        self.URM = u.get_URM()
        self.train_sequential = u.get_train_sequential()
        self.target_playlists = target_playlists
        self.target = None
        self.target_tracks = None
        self.URM_train = None
        self.build_URM_test4()

    def build_URM_test(self):
        possible_playlists = [i for i in range(self.URM.shape[0]) if len(
            self.URM.indices[self.URM.indptr[i]:self.URM.indptr[i + 1]]) > 10]  # playlists with more than 10 songs
        self.target = np.random.choice(possible_playlists, 10000, replace=False)
        self.URM_train = self.URM.copy().tolil()
        self.target_tracks = []

        for idx in self.target:
            length = int(len(self.URM[idx].indices) * 0.2)
            target_songs = np.random.choice(self.URM[idx].indices, length, replace=False)
            self.URM_train[idx, target_songs] = 0
            self.target_tracks.append(target_songs)

        self.target_tracks = np.array(self.target_tracks)
        self.target = pd.DataFrame(self.target, columns=['playlist_id'])
        self.URM_train = self.URM_train.tocsr()

    def build_URM_test2(self):
        possible_playlists = [i for i in range(self.URM.shape[0]) if len(
            self.URM.indices[self.URM.indptr[i]:self.URM.indptr[i + 1]]) > 10]  # playlists with more than 10 songs
        self.target = np.random.choice(possible_playlists, 10000, replace=False)
        self.URM_train = self.URM.copy().tolil()
        self.target_tracks = []

        for idx in self.target[:5000]:
            length = int(len(self.URM[idx].indices) * 0.2)
            target_songs = self.URM[idx].indices[-length:]
            self.URM_train[idx, target_songs] = 0
            self.target_tracks.append(target_songs)

        for idx in self.target[-5000:]:
            length = int(len(self.URM[idx].indices) * 0.2)
            target_songs = np.random.choice(self.URM[idx].indices, length, replace=False)
            self.URM_train[idx, target_songs] = 0
            self.target_tracks.append(target_songs)

        self.target_tracks = np.array(self.target_tracks)
        self.target = pd.DataFrame(self.target, columns=['playlist_id'])
        self.URM_train = self.URM_train.tocsr()

    def build_URM_test3(self):
        target_seq = list(self.train_sequential['playlist_id'][:5000])
        possible_playlists = [i for i in range(self.URM.shape[0]) if len(
            self.URM.indices[self.URM.indptr[i]:self.URM.indptr[i + 1]]) > 7]  # playlists with more than 10 songs
        possible_playlists = np.setdiff1d(possible_playlists, target_seq)
        target_random = np.random.choice(possible_playlists, 5000, replace=False)
        self.target = np.concatenate((target_seq, target_random))
        self.URM_train = self.URM.copy().tolil()
        self.target_tracks = []

        for idx in self.target[:5000]:
            length = int(len(self.URM[idx].indices) * 0.2)
            target_songs = np.array(self.train_sequential[self.train_sequential['playlist_id'] == idx]['track_id'][-length:])
            self.URM_train[idx, target_songs] = 0
            self.target_tracks.append(target_songs)

        for idx in self.target[-5000:]:
            length = int(len(self.URM[idx].indices) * 0.2)
            target_songs = np.random.choice(self.URM[idx].indices, length, replace=False)
            self.URM_train[idx, target_songs] = 0
            self.target_tracks.append(target_songs)

        self.target_tracks = np.array(self.target_tracks)
        self.target = pd.DataFrame(self.target, columns=['playlist_id'])
        self.URM_train = self.URM_train.tocsr()

    def build_URM_test4(self):
        target_seq = list(self.train_sequential['playlist_id'][:5000])
        self.target = target_seq


        for x in range(20):
            length = len([i for i in self.target_playlists['playlist_id'][5000:] if (len(
                self.URM.indices[self.URM.indptr[i]:self.URM.indptr[i + 1]]) >= (x * 5)) &
                          (len(self.URM.indices[self.URM.indptr[i]:self.URM.indptr[i + 1]]) < ((x + 1) * 5))])
            possible_playlists = [i for i in range(self.URM.shape[0]) if len(
                self.URM.indices[self.URM.indptr[i]:self.URM.indptr[i + 1]]) >= (x * 5) &
                                  len(self.URM.indices[self.URM.indptr[i]:self.URM.indptr[i + 1]]) < ((x + 1) * 5)]
            possible_playlists = np.setdiff1d(possible_playlists, target_seq)
            target_random = np.random.choice(possible_playlists, length, replace=False)
            self.target = np.concatenate((self.target, target_random))
        print(sum)
        self.URM_train = self.URM.copy().tolil()
        self.target_tracks = []

        for idx in self.target[:5000]:
            length = int(len(self.URM[idx].indices) * 0.2)
            target_songs = np.array(self.train_sequential[self.train_sequential['playlist_id'] == idx]['track_id'][-length:])
            self.URM_train[idx, target_songs] = 0
            self.target_tracks.append(target_songs)

        for idx in self.target[-5000:]:
            length = int(len(self.URM[idx].indices) * 0.2)
            target_songs = np.random.choice(self.URM[idx].indices, length, replace=False)
            self.URM_train[idx, target_songs] = 0
            self.target_tracks.append(target_songs)

        self.target_tracks = np.array(self.target_tracks)
        self.target = pd.DataFrame(self.target, columns=['playlist_id'])
        self.URM_train = self.URM_train.tocsr()

    def get_URM_train(self):
        return self.URM_train

    def get_target_playlists(self):
        return self.target

    def get_target_tracks(self):
        return self.target_tracks

    @staticmethod
    def AP(recommended_items, relevant_items):
        relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)
        p_at_k = relevant * np.cumsum(relevant, dtype=np.float32) / (1 + np.arange(relevant.shape[0]))
        map_score = np.sum(p_at_k) / np.min([relevant_items.shape[0], relevant.shape[0]])
        return map_score

    # input has to be the URM and the dataframe returned by the recommender
    # NB: the songs in the dataframe must be a list (or ndarray), not a string!
    def MAP(self, df, relevant_items):
        print("Evaluating", flush=True)
        MAP = 0.0
        num_eval = 0

        targets = np.random.choice(df.shape[0], 5000, replace=False)

        for i in targets:
            relevant = relevant_items[i]
            if len(relevant_items) > 0:
                recommended_items = df['track_ids'][i]
                num_eval += 1
                MAP += self.AP(recommended_items, relevant)

        MAP /= num_eval
        print("Recommender performance is {:.8f}".format(MAP))
        return MAP

    @staticmethod
    def result_diff(result_dfs):

        # load  all results form various recommenders
        # for file in files:
        #   results.append(pd.read_csv(file))

        for i, result in enumerate(result_dfs):
            for j, result_2 in enumerate(result_dfs):
                tot_diff = 0
                for row, row_2 in zip(result['track_ids'], result_2['track_ids']):
                    row, row_2 = list(row), list(row_2)
                    row = [el for el in row if el != ' ']
                    row_2 = [el for el in row_2 if el != ' ']
                    tot_diff += [1 for x, y in zip(row, row_2) if x != y].count(1)
                print('Total differences between res %d and res %d are: %d' % (i, j, tot_diff))
