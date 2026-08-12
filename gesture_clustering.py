import torch
import os
from gesture import load_gesture
from collections import defaultdict

class GestureClustering():
    def __init__(self, gesture_path, frame_skip, uncertainty):
        self.path = gesture_path
        self.frame_skip = frame_skip
        self.uncertainty = uncertainty
        self.gesture_map = {}
        self.gesture_order = []

        self.embeddings = torch.empty()
        self.embedding_row_map = torch.empty()
        self.last_embedding_mask = torch.empty()
        self.timestep_mask = []

        self._load_gesture_embeddings()
    
    def _load_gesture_embeddings(self):

        if os.path.exists(self.path):
            gesture_lengths = defaultdict(int)
            gesture_dict = {}

            # load all gestures in
            for ind, gesture_file in enumerate(os.listdir(self.path)):
                file_path = os.path.join(gesture_file, file_path)
                gesture = load_gesture(file_path)
                gesture.set_index(ind)
                self.gesture_map[ind] = gesture
                gesture_lengths[len(gesture)] += 1

            # order by longest gesture embedding
            self.gesture_order = sorted(gesture_dict.keys(), key=lambda x: len(gesture_dict[x]), reverse=True)

            total_keypoints = sum(gesture_lengths.values())

            # pre allocate memory
            self.embeddings = torch.zeros([total_keypoints, 42], dtype=torch.float32)
            embedding_lst = []

            ind_lst = []
            self.embedding_row_map = torch.zeros([total_keypoints, 1], dtype=torch.uint8)

            last_embedding = torch.zeros([total_keypoints, 1], dtype=torch.bool)

            timestep_lst = []
            c = 0

            # create matrix of all embeddings
            # each block matrix is the kth timestep of each gesture with at least k timesteps
            # block matrix order is set by gesture_order
            for embedding_ind in range(max(gesture_lengths.keys())):
                tmp_emb_lst = []
                tmp_ind_lst = []

                # counts block matrix row indices
                timestep_c = 0

                # creates the block matrices and row map for each block matrix stacked together
                for ind in self.gesture_order:
                    timestep_c += 1
                    gesture = gesture_dict[ind]
                    if len(gesture) - 1 < embedding_ind:
                        break

                    elif len(gesture) - 1 == embedding_ind:
                        last_embedding[c, :] = 1

                    keypoints = gesture.keypoints.reshape(1, 42)
                    tmp_emb_lst.append(keypoints)
                    tmp_ind_lst.append(ind)
                    c += 1

                timestep_lst.append(timestep_c)
                embedding_lst.append(tmp_emb_lst)
                ind_lst.append(tmp_ind_lst)

            self.timestep_mask = timestep_lst
            self.embeddings = torch.tensor(embedding_lst)
            self.embedding_row_map = torch.tensor(ind_lst).reshape(-1)

            assert c == total_keypoints, f'Incorrect number of keypoints; should\'ve had {total_keypoints}. Instead had {c}'

        else:
            raise FileNotFoundError('Gestures folder does not exist.')

    def real_time_clustering(self, keypoints, mask, embed_inds):

        current_embeddings = self.embeddings[embed_inds[0]:embed_inds[1], :][mask, :]

        row_map = self.embedding_row_map[embed_inds[0]:embed_inds[1], :][mask, :]

        norm_dist = torch.linalg.vector_norm(current_embeddings - keypoints, dim=1)

        confidence_mask = norm_dist < self.uncertainty

        invalid_indices = row_map[~confidence_mask]

        if confidence_mask.sum().item() > 0: 
            gesture_index = row_map[torch.argmin(norm_dist), :]
            return gesture_index, invalid_indices
        else:
            return -1, torch.empty(0)

    def start_clustering(self):

        # while True:
        #     gesture_ind = self.real_time_clustering()

        #     if gesture_ind == -1:
        #         continue
        pass

def preprocess_keypoints():
    pass

def create_boolean_mask(ind_array, valid_inds):
    mask = torch.isin(ind_array, valid_inds, assume_unique=True)
    return mask