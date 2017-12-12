import numpy as np


class Retrieval:
    def __init__(self, features, labels, clutter=None):
        self.features = features
        self.labels = labels
        self.clutter = clutter
        self.distance_index = None
        self.groundtruth = None
        self.sample_num = self.features.shape[0]

    def preprocess(self):
        self.features = self.features / np.sqrt(np.sum(np.square(self.features), axis=1))

    def cal_distance(self):
        distance = self.get_distance(self.features, self.features)
        self.distance_index = np.argsort(distance)
        self.groundtruth = np.zeros(self.distance_index.shape)
        for i in range(self.features.shape[0]):
            self.groundtruth[i] = (self.labels[self.distance_index[i]] == self.labels[i]).astype(np.int)

    def retrieval_for_all(self, k):
        k_num = len(k)
        precision = np.zeros([self.sample_num, k_num])
        recall = np.zeros([self.sample_num, k_num])
        F1 = np.zeros([self.sample_num, k_num])
        MRR = np.zeros([self.sample_num, k_num])
        for i in range(self.sample_num):
            if self.labels[i] == self.clutter:
                continue
            for j in range(k_num):
                precision[i, j], recall[i, j], F1[i, j], MRR[i, j] = self.get_top_K(self.groundtruth[i], k[j])
        precision = np.mean(precision, axis=0)
        recall = np.mean(recall, axis=0)
        F1 = np.mean(F1, axis=0)
        MRR = np.mean(MRR, axis=0)
        return precision, recall, F1, MRR

    def get_top_K(self, truth, k):
        precision = np.sum(truth[:k])/k
        recall = np.sum(truth[:k])/50
        F1 = 2*precision*recall/(precision+recall)
        truth = truth[:k] * np.arange(1, 1+k)
        c= np.sum(1/truth[(np.where(truth > 0))[0]])
        MRR = np.sum(1/truth[(np.where(truth > 0))[0]]) / k
        return precision, recall, F1, MRR

    @staticmethod
    def get_distance(a, b):
        a_num = a.shape[0]
        b_num = b.shape[0]
        a_sq = np.tile(np.sum(np.square(a), 1), (b_num, 1)).T
        b_sq = np.tile(np.sum(np.square(b), 1), (a_num, 1))
        result = a_sq + b_sq - 2 * np.dot(a, b.T)
        return result
