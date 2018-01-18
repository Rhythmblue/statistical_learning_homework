import numpy as np


class Binary_Classification:
    def __init__(self, train_x, train_y, test_x, test_p, test_y, k=5):
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_p = test_p
        self.test_y = test_y
        self.cross_val_index = self.get_split_index(k)

    def get_split_index(self, k=5):
        test_num = int(self.train_x.shape[0]*0.2)
        split_index = []
        perm = np.arange(self.train_x.shape[0])
        np.random.seed(1103)
        np.random.shuffle(perm)
        perm = np.split(perm, k)
        for i in range(k):
            test_index = perm[i]
            train_index = []
            for j in range(k):
                if j!=i:
                    train_index.append(perm[j])
            train_index = np.concatenate(train_index)
            split_index.append([train_index, test_index])
        return split_index

    def run_ridge_regression(self, order=1, lmd=0, mode='val'):
        if mode == 'val':
            error_rate = []
            for index in self.cross_val_index:
                train_x_poly = self.get_poly(self.train_x[index[0]], order)
                val_x_poly = self.get_poly(self.train_x[index[1]], order)
                val_y = self.train_y[index[1]]
                w = self.ridge_regression(lmd, train_x_poly, self.train_y[index[0]])
                predict_val = np.dot(val_x_poly, w) >=0.5
                val_p = np.full(val_y.shape, 1/val_y.shape[0])
                error_rate.append(self.cal_error_rate(predict_val, val_p, val_y))
            return np.mean(np.array(error_rate))
        elif mode == 'test':
            train_x_poly = self.get_poly(self.train_x, order)
            test_x_poly = self.get_poly(self.test_x, order)
            w = self.ridge_regression(lmd, train_x_poly, self.train_y)
            predict = np.dot(test_x_poly, w) >= 0.5
            return self.cal_error_rate(predict, self.test_p, self.test_y)
        elif mode == 'ml':
            train_x_poly = self.get_poly(self.train_x, order)
            w = self.ridge_regression(0, train_x_poly, self.train_y)
            y = np.dot(train_x_poly, w)
            y[np.where(y<0)] = 0
            y[np.where(y>1)] = 1
            ll = np.sum(np.log(self.train_y * y + (1-self.train_y) * (1-y) +1e-5))
            k = w.shape[0]
            n = train_x_poly.shape[0]
            return ll - 0.5 * k * np.log(n)
        else:
            return

    def run_lasso_regression(self, order=1, lmd=0.1):
        from sklearn.linear_model import Lasso
        lasso = Lasso(max_iter=1000, alpha=lmd)
        train_x_poly = self.get_poly(self.train_x, order)
        test_x_poly = self.get_poly(self.test_x, order)
        predict = lasso.fit(train_x_poly, self.train_y).predict(test_x_poly)
        predict = predict >= 0.5
        return self.cal_error_rate(predict, self.test_p, self.test_y)

    def run_logistic_regression(self, order, iteration, learning_rate, mode):
        if mode == 'val':
            train_x_poly = self.get_poly(self.train_x, order)
            w = np.zeros(train_x_poly.shape[1])
            for step in range(iteration):
                w = self.gradient_descent(w, train_x_poly, self.train_y, learning_rate)
            bic = self.bic(w, train_x_poly, self.train_y)
            predict = np.dot(train_x_poly, w) >= 0
            train_p = np.full(self.train_y.shape, 1 / self.train_y.shape[0])
            error_rate = self.cal_error_rate(predict, train_p, self.train_y)
            return bic, error_rate
        elif mode == 'test':
            train_x_poly = self.get_poly(self.train_x, order)
            test_x_poly = self.get_poly(self.test_x, order)
            w = np.zeros(train_x_poly.shape[1])
            for step in range(iteration):
                w = self.gradient_descent(w, train_x_poly, self.train_y, learning_rate)
            predict = np.dot(test_x_poly, w) >=0
            return self.cal_error_rate(predict, self.test_p, self.test_y)

    def run_newton_raphson(self, order, iteration):
        train_x_poly = self.get_poly(self.train_x, order)
        test_x_poly = self.get_poly(self.test_x, order)
        w = np.zeros(train_x_poly.shape[1])
        for step in range(iteration):
            w = self.newton_raphson(w, train_x_poly, self.train_y)
        predict = np.dot(test_x_poly, w) >= 0
        return self.cal_error_rate(predict, self.test_p, self.test_y)

    def run_fisher_lda(self, order):
        index_0 = np.where(self.train_y==0)
        index_1 = np.where(self.train_y==1)
        train_x_poly = self.get_poly(self.train_x, order)[:, 1:]
        test_x_poly = self.get_poly(self.test_x, order)[:, 1:]
        train_x_0 = train_x_poly[index_0]
        train_x_1 = train_x_poly[index_1]
        m_0 = np.mean(train_x_0, axis=0)
        m_1 = np.mean(train_x_1, axis=0)
        sw = np.dot((train_x_0-m_0).T, train_x_0-m_0) +np.dot((train_x_1-m_1).T, train_x_1-m_1)
        w = np.zeros(2*order+1)
        w[1:] = np.linalg.inv(sw).dot(m_1-m_0)
        w[0] = -np.mean(np.dot(train_x_poly, w[1:]))
        if (m_1.dot(w[1:]) + w[0]) > 0:
            predict = (np.dot(test_x_poly, w[1:]) + w[0]) >= 0
        else:
            predict = (np.dot(test_x_poly, w[1:]) + w[0]) < 0
        return w, self.cal_error_rate(predict, self.test_p, self.test_y)
    
    def run_multi_class(self, order, n_clusters):
        from sklearn.cluster import KMeans
        from sklearn.svm import SVC
        from sklearn.multiclass import  OneVsRestClassifier
        train_x_poly = self.get_poly(self.train_x, order)
        test_x_poly = self.get_poly(self.test_x, order)
        clf = KMeans(n_clusters)
        clf.fit(train_x_poly[:, 1:3])
        train_label = clf.labels_
        label_convert = {}
        for i in range(n_clusters):
            index = np.where(train_label==i)
            label_convert[i] = int(np.mean(self.train_y[index])>=0.5)
        model = OneVsRestClassifier(SVC())
        model.fit(train_x_poly, train_label)
        test_label = model.predict(test_x_poly)
        predict = np.zeros(test_label.shape)
        for i in label_convert.keys():
            predict[np.where(test_label==i)] = label_convert[i]
        return train_label, test_label, predict, self.cal_error_rate(predict==1, self.test_p, self.test_y)

    @staticmethod
    def cal_error_rate(predict, test_p, test_y):
        predict_0 = np.where(predict == False)
        predict_1 = np.where(predict == True)
        error_rate = np.sum(test_p[predict_0] * test_y[predict_0]) \
                     + np.sum(test_p[predict_1] *(1-test_y[predict_1]))
        return error_rate

    @staticmethod
    def get_poly(raw, order):
        assert order>=1
        poly = [np.ones([raw.shape[0],1])]
        for i in range(1, order+1):
            poly.append(raw**i)
        poly = np.concatenate(poly, axis=1)
        return poly

    @staticmethod
    def ridge_regression(lmd, x, y):
        xTx = np.dot(x.T, x)
        m, n = xTx.shape
        return np.dot(np.linalg.inv(xTx + lmd * np.eye(m, n)), np.dot(x.T, y))

    @staticmethod
    def gradient_descent(w, x, t, learning_rate):
        n = x.shape[0]
        delta_w = (1/n) * (sigmoid(np.dot(x, w))-t).dot(x)
        return w - learning_rate*delta_w

    @staticmethod
    def newton_raphson(w, x, t):
        def hessian(x, y):
            n = y.shape[0]
            y = np.reshape(y, [n, 1])
            r = y.dot(y.T) * np.eye(n)
            return (x.T).dot(r).dot(x)
        y = sigmoid(np.dot(x, w))
        return w - np.linalg.inv(hessian(x, y)).dot(x.T).dot(y-t)

    @staticmethod
    def bic(w, x, t):
        k = w.shape[0]  # parameter num
        n = x.shape[0]  # sample num
        l = likelihood(sigmoid(np.dot(x, w)), t)
        bic = l - 0.5 * k * np.log(n)
        return bic

def likelihood(x, t):
    # first = (t - (x > 0)) * x
    # second = np.log(1 + np.exp(x - 2 * x * (x > 0)))
    # return np.sum(first - second) / x.shape[0]
    return np.sum(np.log(t * x + (1-t) * (1-x) +1e-5))

def sigmoid(x):
    return 1 / (1 + np.e ** -x)


