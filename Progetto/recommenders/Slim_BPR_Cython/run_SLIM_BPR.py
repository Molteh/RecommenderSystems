
from SLIM_BPR_Cython import SLIM_BPR_Cython
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import MultiLabelBinarizer

train = pd.read_csv("../../data/train.csv")
tracks = pd.read_csv("../../data/tracks.csv")
target_playlists = pd.read_csv("../../data/target_playlists.csv")


def buildAndGetURM():
    grouped = train.groupby('playlist_id', as_index=True).apply((lambda playlist: list(playlist['track_id'])))
    URM = MultiLabelBinarizer(classes=tracks['track_id'].unique(), sparse_output=True).fit_transform(grouped)
    URM = URM.tocsr()
    return URM


def run_SLIM():

    print('Trying to build URM')

    URM_train = buildAndGetURM()

    print('URM built')

    print('Trying to build recommender')

    recommender = SLIM_BPR_Cython(URM_train, recompile_cython=True, positive_threshold=0, sparse_weights=True)

    print('Recommender built')

    logFile = open("Result_log.txt", "a")

    print('Trying to fit recommender')

    recommender.fit(,

    print('Recommender fit')
    #results_run = recommender.evaluateRecommendations(URM_test, at=5)
    #print(results_run)


    final_result = pd.DataFrame(index=range(target_playlists.shape[0]), columns=('playlist_id', 'track_ids'))

    for i, target_playlist in tqdm(enumerate(np.array(target_playlists))):
        result_tracks = recommender.recommend(int(target_playlist))
        string_rec = ' '.join(map(str, result_tracks.reshape(1, 10)[0]))
        final_result['playlist_id'][i] = int(target_playlist)
        final_result['track_ids'][i] = string_rec
    final_result.to_csv('fuckin_Cython_SBPR.csv', index=False)


run_SLIM()
