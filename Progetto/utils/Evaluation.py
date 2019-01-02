import pandas as pd
import numpy as np
import scipy.sparse as sp
from tqdm import tqdm


class Eval(object):

    def __init__(self, u, holdout):
        self.l10 = pd.read_csv("./data/longshort/l10.csv")
        self.l30 = pd.read_csv("./data/longshort/l30.csv")
        self.l60 = pd.read_csv("./data/longshort/l60.csv")
        self.g60 = pd.read_csv("./data/longshort/g60.csv")
        self.URM = u.URM
        self.train_sequential = u.train_sequential
        self.target_playlists = u.target_playlists
        self.URM_train = None
        self.URM_test = None
        self.test_playlists = None
        self.build_URM_train(holdout)


    def build_URM_train(self, holdout):
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

        self.test_playlists = self.get_test_playlists(holdout)
        self.URM_test = URM_target.tocsr()
        self.URM_train = self.URM_train.tocsr()


    def get_test_playlists(self, holdout):
        if holdout== "standard":
            print("Generated standard hold-out")
            return np.random.choice(self.target_playlists['playlist_id'], 5000, replace=False)
        elif holdout== "l10":
            print("Generated l10 hold-out")
            return self.l10['playlist_id']
        elif holdout== "l30":
            print("Generated l30 hold-out")
            return self.l30['playlist_id']
        elif holdout== "l60":
            print("Generated l60 hold-out")
            return self.l60['playlist_id']
        else:
            print("Generated g60 hold-out")
            return self.g60['playlist_id']


    @staticmethod
    def AP(recommended_items, relevant_items):
        relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)
        p_at_k = relevant * np.cumsum(relevant, dtype=np.float32) / (1 + np.arange(relevant.shape[0]))
        map_score = np.sum(p_at_k) / np.min([relevant_items.shape[0], relevant.shape[0]])
        return map_score


    def evaluate_algorithm(self, recommender):
        cumulative_MAP = 0.0
        num_eval = 0

        for user_id in self.test_playlists:

            relevant_items = self.URM_test[user_id].indices

            if len(relevant_items) > 0:
                recommended_items = recommender.recommend(user_id)
                num_eval += 1

                cumulative_MAP += self.AP(recommended_items, relevant_items)

        cumulative_MAP /= num_eval

        print("Recommender performance is: {:.8f}".format(cumulative_MAP))


    def evaluate_algorithm_longshort(self, recommender):
        cumulative_MAP = 0.0
        num_eval = 0
        l10 = self.l10['playlist_id'].unique()
        l30 = self.l30['playlist_id'].unique()
        l60 = self.l60['playlist_id'].unique()
        g60 = self.g60['playlist_id'].unique()

        for user_id in self.test_playlists:

            relevant_items = self.URM_test[user_id].indices

            if len(relevant_items) > 0:
                recommended_items = 0

                if user_id in l10:
                    recommended_items = recommender.recommend_l10(user_id)
                elif user_id in l30:
                    recommended_items = recommender.recommend_l30(user_id)
                elif user_id in l60:
                    recommended_items = recommender.recommend_l60(user_id)
                elif user_id in g60:
                    recommended_items = recommender.recommend_g60(user_id)
                else:
                    print(user_id)
                    print("Playlist not in the split")

                num_eval += 1
                cumulative_MAP += self.AP(recommended_items, relevant_items)

        cumulative_MAP /= num_eval

        print("Recommender performance is: {:.8f}".format(cumulative_MAP))


    def generate_predictions(self, recommender, path):
        target_playlists = self.target_playlists
        final_result = pd.DataFrame(index=range(target_playlists.shape[0]), columns=('playlist_id', 'track_ids'))

        for i, user_id in tqdm(enumerate(np.array(target_playlists))):
            recommended_items = recommender.recommend(int(user_id))

            if len(recommended_items) != 10:
                print(len(recommended_items))

            final_result['playlist_id'][i] = int(user_id)
            string_rec = ' '.join(map(str, recommended_items.reshape(1, 10)[0]))
            final_result['track_ids'][i] = string_rec

        final_result.to_csv(path, index=False)


    def generate_predictions_longshort(self, recommender, path):
        target_playlists = self.target_playlists
        final_result = pd.DataFrame(index=range(target_playlists.shape[0]), columns=('playlist_id', 'track_ids'))
        l10 = self.l10['playlist_id'].unique()
        l30 = self.l30['playlist_id'].unique()
        l60 = self.l60['playlist_id'].unique()
        g60 = self.g60['playlist_id'].unique()

        for i, user_id in tqdm(enumerate(np.array(target_playlists))):
            recommended_items = 0

            if user_id in l10:
                recommended_items = recommender.recommend_l10(int(user_id))
            elif user_id in l30:
                recommended_items = recommender.recommend_l30(int(user_id))
            elif user_id in l60:
                recommended_items = recommender.recommend_l60(int(user_id))
            elif user_id in g60:
                recommended_items = recommender.recommend_g60(int(user_id))
            else:
                print(user_id)
                print("Playlist not in the split")

            if len(recommended_items) != 10:
                print(len(recommended_items))

            final_result['playlist_id'][i] = int(user_id)
            string_rec = ' '.join(map(str, recommended_items.reshape(1, 10)[0]))
            final_result['track_ids'][i] = string_rec

        final_result.to_csv(path, index=False)
