# coding=utf-8
import pickle
import os
import numpy as np


class Data_Loader:
    def __init__(self):
        self.genre_map, self.movie_genre = self.load_movie_info()
        self.age_map = [1, 18, 25, 35, 45, 50, 56]
        self.user_info = self.load_user_info()
        self.user_rating = self.load_user_rating()
        self.movie_feature, self.genre_feature, self.genre_mask_feature= self.gen_features()
        self.perm = self.gen_cross_val()

    def load_movie_info(self):
        if os.path.exists('data/genre_map.npy') and os.path.exists('data/movie_genre.npy'):
            genre_map = np.load('data/genre_map.npy').tolist()
            movie_genre = np.load('data/movie_genre.npy')
            return genre_map, movie_genre
        movie_info = []
        id = []
        with open('proj3_data/task1/movies.dat', 'r', encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip().split('::')
                assert len(line) == 3
                line[0] = int(line[0])-1
                id.append(line[0])
                movie_info.append(line[2].split('|'))
        genre_map = []
        for movie in movie_info:
            for genre in movie:
                if genre not in genre_map:
                    genre_map.append(genre)
        genre_map.sort()

        movie_genre = np.zeros([max(id)+1, len(genre_map)])
        for i, movie in enumerate(movie_info):
            for genre in movie:
                movie_genre[id[i], genre_map.index(genre)] = 1
        np.save('data/genre_map.npy', genre_map)
        np.save('data/movie_genre.npy', movie_genre)
        return genre_map, movie_genre

    def load_user_info(self):
        if os.path.exists('data/user_matrix.npy'):
            user_matrix = np.load('data/user_matrix.npy')
            return user_matrix
        user_info = []
        for i in range(10):
            f = open('proj3_data/task1/users.dat{:d}'.format(i), 'r', encoding='utf-8')
            for line in f.readlines():
                line = line.strip().split('::')
                assert line[1] in ['M', 'F']
                if line[1]=='M':
                    line[1]='0'
                else:
                    line[1]='1'
                line = [int(x) for x in line[:-1]]
                line[2] = self.age_map.index(line[2])
                user_info.append(line)
            f.close()
        user_matrix = np.zeros([len(user_info), len(user_info[0])-1])
        for user in user_info:
            user_matrix[user[0]-1] = np.array(user[1:])
        np.save('data/user_matrix.npy', user_matrix)
        return user_matrix

    def load_user_rating(self):
        if os.path.exists('data/user_rating.dict'):
            with open('data/user_rating.dict', 'rb') as input:
                user_rating = pickle.load(input)
            return dict(user_rating)
        rating_info = []
        with open('proj3_data/task1/ratings.dat', 'r', encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip().split('::')
                line = [int(x) for x in line[:-1]]
                line[0] -=1
                line[1] -=1
                rating_info.append(line)
        user_rating = {}
        for rate in rating_info:
            if rate[0] not in user_rating.keys():
                user_rating[rate[0]] = {}
            user_rating[rate[0]][rate[1]] = rate[2]
        with open('data/user_rating.dict', 'wb') as output:
            pickle.dump(user_rating, output)
        return user_rating

    def gen_features(self):
        movie_feature = np.zeros([self.user_info.shape[0], self.movie_genre.shape[0]])
        genre_feature = np.zeros([self.user_info.shape[0], self.movie_genre.shape[1]])
        genre_mask_feature = genre_feature.copy()
        for user in self.user_rating.keys():
            for movie in self.user_rating[user].keys():
                rating = self.user_rating[user][movie]
                movie_feature[user, movie] = rating
                genre_feature[user] += self.movie_genre[movie] * rating
                genre_mask_feature[user] += self.movie_genre[movie]
        genre_mask = genre_mask_feature.copy()
        genre_mask[np.where(genre_mask==0)] = 1
        genre_feature = genre_feature / genre_mask
        return movie_feature, genre_feature, genre_mask_feature

    def gen_cross_val(self, fold=10):
        perm = np.arange(self.user_info.shape[0])
        np.random.seed(1103)
        np.random.shuffle(perm)
        perm = np.split(perm, fold)
        return perm

    def get_fold(self, fold_index):
        train_index = []
        val_index = []
        for i in range(len(self.perm)):
            if i == fold_index:
                val_index = np.array(self.perm[i])
            else:
                train_index.append(np.array(self.perm[i]))
        train_index = np.concatenate(train_index)
        train = {'user_info': self.user_info[train_index],
                 'movie_feature': self.movie_feature[train_index],
                 'genre_feature': self.genre_feature[train_index],
                 'genre_mask_feature': self.genre_mask_feature[train_index]}
        val =  {'user_info': self.user_info[val_index],
                 'movie_feature': self.movie_feature[val_index],
                 'genre_feature': self.genre_feature[val_index],
                 'genre_mask_feature': self.genre_mask_feature[val_index]}
        return train, val


class User_and_Movie:
    def __init__(self):
        self.dl = Data_Loader()

    def estimator(self, fold, mode='genre_mask_feature', target = 'gender'):
        train, val = self.dl.get_fold(fold)
        target_map = {'gender': 0, 'age': 1}
        if '+' not in mode:
            train_feature = train[mode]
            val_feature = val[mode]
        else:
            train_feature = []
            val_feature = []
            for part in mode.split('+'):
                train_feature.append(train[part])
                val_feature.append(val[part])
            train_feature = np.concatenate(train_feature, axis=1)
            val_feature = np.concatenate(val_feature, axis=1)
        train_label = train['user_info'][:, target_map[target]]
        val_label = val['user_info'][:, target_map[target]]
        from sklearn.svm import SVC
        import time
        clf = SVC()
        time_1 = time.time()
        clf.fit(train_feature, train_label)
        time_2 = time.time()
        predict = clf.predict(val_feature)
        time_3 = time.time()
        error = self.cal_error_rate(predict, val_label)
        train_time = time_2-time_1
        test_time = time_3-time_2
        return error, train_time, test_time

    @staticmethod
    def cal_error_rate(predict, label):
        total = predict.shape[0]
        error = np.sum(predict!=label)
        return error/total

    @staticmethod
    def cal_weight_error_rate(predict, label):
        total = predict.shape[0]
        error = np.sum(np.abs(predict-label))
        return error/total