import torch
import os
from gesture import load_gesture
from collections import defaultdict

class GestureClustering():
    def __init__(self, gesture_path, frame_skip):
        self.path = gesture_path
        self.frame_skip = frame_skip
        self.gesture_map = {}
        self.gesture_order = []

        self.embeddings = torch.empty(1)
        self.embedding_row_map = torch.empty(1)
        self.last_embedding_mask = torch.empty(1)
        # self.

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

            self.gesture_order = sorted(gesture_dict.keys(), key=lambda x: len(gesture_dict[x]))

            total_keypoints = sum(gesture_lengths.values())

            self.embeddings = torch.zeros([total_keypoints, 42], dtype=torch.float32)
            embedding_lst = []

            ind_lst = []
            self.embedding_row_map = torch.zeros([total_keypoints, 1], dtype=torch.uint8)

            last_embedding = torch.zeros([total_keypoints, 1], dtype=torch.bool)
            c = 0

            for embedding_ind in range(max(gesture_lengths.keys())):
                tmp_emb_lst = []
                tmp_ind_lst = []
                for ind in self.gesture_order:
                    gesture = gesture_dict[ind]
                    if len(gesture) - 1 < embedding_ind:
                        break

                    elif len(gesture) - 1 == embedding_ind:
                        last_embedding[c, :] = 1

                    keypoints = gesture.keypoints.reshape(1, 42)
                    tmp_emb_lst.append(keypoints)
                    tmp_ind_lst.append(ind)
                    c += 1

                embedding_lst.append(tmp_emb_lst)
                ind_lst.append(tmp_ind_lst)

            self.embeddings = torch.tensor(embedding_lst)
            self.embedding_row_map = torch.tensor(ind_lst).reshape(-1)

            assert c == total_keypoints, f'Incorrect number of keypoints; should\'ve had {total_keypoints}. Instead had {c}'

        else:
            raise FileNotFoundError('Gestures folder does not exist.')

    def real_time_clustering(self):

        pass

    def start_clustering(self):

        # while True:
        #     gesture_ind = self.real_time_clustering()

        #     if gesture_ind == -1:
        #         continue
        pass