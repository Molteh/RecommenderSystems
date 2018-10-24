import numpy as np
import scipy.sparse as sp
from tqdm import tqdm

class SlimBPR(object):

    def __init__(self,
                 URM,
                 learning_rate=0.01,
                 epochs=1,
                 positive_item_regularization=1.0,
                 negative_item_regularization=1.0,
                 nnz=1):
        self.URM = URM
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.positive_item_regularization = positive_item_regularization
        self.negative_item_regularization = negative_item_regularization
        self.nnz = nnz
        self.n_playlist = self.URM.shape[0]
        self.n_track = self.URM.shape[1]

        self.similarity_matrix = sp.lil_matrix((self.n_track, self.n_track))

    def sample(self):

        playlist_id = np.random.choice(self.n_playlist)

        # get tracks in playlist and choose one
        tracks = self.URM[playlist_id, :].indices
        pos_track_id = np.random.choice(tracks)

        negTrackSelected = False

        while not negTrackSelected:
            neg_track_id = np.random.choice(self.n_track)
            if neg_track_id not in tracks:
                negTrackSelected = True
        return playlist_id, pos_track_id, neg_track_id

    def epochIteration(self):

        numPosInteractions = int(self.URM.nnz * self.nnz)

        # sampling without replacement
        # tqdm performs range op with progress visualization
        for num_sample in tqdm(range(numPosInteractions)):

            playlist_id, pos_track_id, neg_track_id = self.sample()

            tracks = self.URM[playlist_id, :].indices

            # Prediction
            x_i = self.similarity_matrix[pos_track_id, tracks].sum()
            x_j = self.similarity_matrix[neg_track_id, tracks].sum()

            # Gradient
            x_ij = x_i - x_j

            gradient = 1 / (1 + np.exp(x_ij))

            for i in tracks:
                # dp and dn outside for?
                dp = gradient - self.positive_item_regularization * x_i
                self.similarity_matrix[pos_track_id, i] = self.similarity_matrix[
                                                              pos_track_id, i] + self.learning_rate * dp
                dn = gradient - self.negative_item_regularization * x_j
                self.similarity_matrix[neg_track_id, i] = self.similarity_matrix[
                                                              neg_track_id, i] - self.learning_rate * dn

            self.similarity_matrix[pos_track_id, pos_track_id] = 0
            self.similarity_matrix[pos_track_id, pos_track_id] = 0

    def get_S_SLIM_BPR(self, knn):
        print('get S Slim BPR...')

        for numEpoch in range(self.epochs):
            print('Epoch: ', numEpoch)
            self.epochIteration()

        # replace with our own knn methods
        print('Keeping only knn =', knn, '...')
        similarity_matrix_csr = self.similarity_matrix.tocsr()

        for row in tqdm(range(0, similarity_matrix_csr.shape[0])):
            ordered_indices = similarity_matrix_csr[row, :].data.argsort()[:-knn]
            similarity_matrix_csr[row, :].data[ordered_indices] = 0
        sp.csr_matrix.eliminate_zeros(similarity_matrix_csr)

        return similarity_matrix_csr