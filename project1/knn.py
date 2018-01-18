import numpy as np

class KNN:
    def __init__(self, data, label):
        self.data = data
        self.label = label
        self.fold_num = 5
        self.data_fold = []
        self.label_fold = []

    def split_data(self, num):
        self.fold_num = num
        np.random.seed(1)
        perm = np.arange(self.data.shape[0])
        np.random.shuffle(perm)
        data_now = self.data[perm]
        label_now = self.label[perm]
        self.data_fold = np.array_split(data_now, num)
        self.label_fold = np.array_split(label_now, num)

    def run_knn(self, k=1, normalization = False, pca=0, norm=2):
        true_count = 0
        for i in range(self.fold_num):
            val_data = self.data_fold[i]
            val_label = self.label_fold[i]
            train_data = []
            train_label = []
            for j in range(self.fold_num):
                if i != j:
                    train_data.append(self.data_fold[j])
                    train_label.append(self.label_fold[j])
            train_data = np.concatenate(train_data)
            train_label = np.concatenate(train_label)
            if normalization:
                tmp_max = np.max(train_data, axis=0)
                train_data = train_data / tmp_max
                val_data = val_data / tmp_max
            if pca != 0:
                tmp_mean = np.mean(train_data, axis=0)
                train_data = train_data - tmp_mean
                val_data = val_data - tmp_mean
                eigVals, eigVects = np.linalg.eig(np.cov(train_data, rowvar=False))
                eigVects = eigVects[:, np.argsort(-eigVals)[:pca]]
                train_data = np.dot(train_data, eigVects)
                val_data = np.dot(val_data, eigVects)
            if norm==2:
                distance = self.get_distance(val_data, train_data)
            elif norm==1:
                distance = self.get_1_norm_distance(val_data, train_data)
            index = np.argsort(distance)
            for i in range(index.shape[0]):
                count = np.zeros(4)
                for j in range(k):
                    count[train_label[index[i, j]]] += 1
                if np.argmax(count) == val_label[i]:
                    true_count += 1
        return true_count/self.data.shape[0]

    @staticmethod
    def get_distance(a, b):
        a_num = a.shape[0]
        b_num = b.shape[0]
        a_sq = np.tile(np.sum(np.square(a), 1), (b_num, 1)).T
        b_sq = np.tile(np.sum(np.square(b), 1), (a_num, 1))
        result = a_sq + b_sq - 2 * np.dot(a, b.T)
        return result

    @staticmethod
    def get_1_norm_distance(a, b):
        result = np.zeros([a.shape[0], b.shape[0]])
        for i in range(a.shape[0]):
            for j in range(b.shape[0]):
                result[i, j] = np.sum(np.abs(a[i]-b[j]))
        return result
