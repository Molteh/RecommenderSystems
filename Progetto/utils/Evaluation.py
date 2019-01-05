import pandas as pd
import numpy as np
import scipy.sparse as sp
from tqdm import tqdm
from Progetto.recommenders.Ensemble_post import Ensemble_post


class Eval(object):

    def __init__(self, u, holdout):
        self.c1 = pd.read_csv("./data/longshort/0-10.csv")
        self.c2 = pd.read_csv("./data/longshort/10-15.csv")
        self.c3 = pd.read_csv("./data/longshort/15-25.csv")
        self.c4 = pd.read_csv("./data/longshort/25-40.csv")
        self.c5 = pd.read_csv("./data/longshort/40-60.csv")
        self.c6 = pd.read_csv("./data/longshort/60-100.csv")
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
        elif holdout== "0-10":
            print("Generated 0-10 hold-out")
            return self.c1['playlist_id']
        elif holdout== "10-15":
            print("Generated 10-15 hold-out")
            return self.c2['playlist_id']
        elif holdout== "15-25":
            print("Generated 15-25 hold-out")
            return self.c3['playlist_id']
        elif holdout== "25-40":
            print("Generated 25-40 hold-out")
            return self.c4['playlist_id']
        elif holdout== "40-60":
            print("Generated 40-60 hold-out")
            return self.c5['playlist_id']
        elif holdout== "60-100":
            print("Generated 60-100 hold-out")
            return self.c6['playlist_id']
        else:
            print("Wrong holdout")


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
        c1 = self.c1['playlist_id'].unique()
        c2 = self.c2['playlist_id'].unique()
        c3 = self.c3['playlist_id'].unique()
        c4 = self.c4['playlist_id'].unique()
        c5 = self.c5['playlist_id'].unique()
        c6 = self.c6['playlist_id'].unique()

        for user_id in self.test_playlists:

            relevant_items = self.URM_test[user_id].indices

            if len(relevant_items) > 0:
                recommended_items = 0

                if user_id in c1:
                    recommended_items = recommender.recommend_c1(int(user_id))
                elif user_id in c2:
                    recommended_items = recommender.recommend_c2(int(user_id))
                elif user_id in c3:
                    recommended_items = recommender.recommend_c3(int(user_id))
                elif user_id in c4:
                    recommended_items = recommender.recommend_c4(int(user_id))
                elif user_id in c5:
                    recommended_items = recommender.recommend_c5(int(user_id))
                elif user_id in c6:
                    recommended_items = recommender.recommend_c6(int(user_id))
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

    def generate_predictions_clustered(self, path):
        target_playlists = self.target_playlists
        final_result = pd.DataFrame(index=range(target_playlists.shape[0]), columns=('playlist_id', 'track_ids'))

        c1 = self.c1['playlist_id'].unique()
        c2 = self.c2['playlist_id'].unique()
        c3 = self.c3['playlist_id'].unique()
        c4 = self.c4['playlist_id'].unique()
        c5 = self.c5['playlist_id'].unique()
        c6 = self.c6['playlist_id'].unique()

        clusters = [c1, c2, c3, c4, c5, c6]
        i = 0

        for c in clusters:

            rec = Ensemble_post()
            rec.fit() #pass params
            for user_id in c:
                final_result['playlist_id'][i] = int(user_id)
                recommended_items = rec.recommend(user_id)
                string_rec = ' '.join(map(str, recommended_items.reshape(1, 10)[0]))
                final_result['track_ids'][i] = string_rec
                i += 1

        final_result.to_csv(path, index=False)


    def generate_predictions_longshort(self, recommender, path):
        target_playlists = self.target_playlists
        final_result = pd.DataFrame(index=range(target_playlists.shape[0]), columns=('playlist_id', 'track_ids'))
        c1 = self.c1['playlist_id'].unique()
        c2 = self.c2['playlist_id'].unique()
        c3 = self.c3['playlist_id'].unique()
        c4 = self.c4['playlist_id'].unique()
        c5 = self.c5['playlist_id'].unique()
        c6 = self.c6['playlist_id'].unique()

        for i, user_id in tqdm(enumerate(np.array(target_playlists))):
            recommended_items = 0

            if user_id in c1:
                recommended_items = recommender.recommend_c1(int(user_id))
            elif user_id in c2:
                recommended_items = recommender.recommend_c2(int(user_id))
            elif user_id in c3:
                recommended_items = recommender.recommend_c3(int(user_id))
            elif user_id in c4:
                recommended_items = recommender.recommend_c4(int(user_id))
            elif user_id in c5:
                recommended_items = recommender.recommend_c5(int(user_id))
            elif user_id in c6:
                recommended_items = recommender.recommend_c6(int(user_id))
            else:
                print(user_id)
                print("Playlist not in the split")

            if len(recommended_items) != 10:
                print(len(recommended_items))

            final_result['playlist_id'][i] = int(user_id)
            string_rec = ' '.join(map(str, recommended_items.reshape(1, 10)[0]))
            final_result['track_ids'][i] = string_rec

        final_result.to_csv(path, index=False)
