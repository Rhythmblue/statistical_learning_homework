import numpy as np
import matplotlib.pyplot as plt

class Iris:
    def __init__(self, file, is_np=False):
        self.data, self.target, self.label_map \
            = load_data(file)
        if is_np:
            self.data = np.array(self.data)
        self.attribute = ['sepal length', 'sepal width', 'petal length', 'petal width']

def load_data(file):
    data = []
    target = []
    label_map = []
    with open(file, 'r') as f:
        for line in f.readlines():
            tmp = line.strip().split(',')
            if len(tmp) != 5:
                break
            tmp[:4] = [float(x) for x in tmp[:4]]
            data.append(tmp[:4])
            if tmp[4] not in label_map:
                label_map.append(tmp[4])
            target.append(label_map.index(tmp[4]))
    return data, target, label_map

def main():
    iris = Iris('bezdekIris.data', True)

    iris_cov = np.cov(iris.data.T)
    print('Iris\'covariance matrix:')
    print(iris_cov)

    iris_corrcoef = np.corrcoef(iris.data.T)
    print('Iris\'correlation coefficient matrix:')
    print(iris_corrcoef)

    eigen_value, eigen_vectors = np.linalg.eig(iris_cov)
    KL_matrix = np.dot(iris.data, eigen_vectors)
    print('eigen_values:')
    print(eigen_value)
    print('eigen_vectors')
    print(eigen_vectors)
    print('After K-L transform:')
    print(KL_matrix)
    print(KL_matrix.shape)


    plt.figure()
    plt.boxplot(iris.data, labels=iris.attribute)
    plt.ylabel('Values (centimeters)')
    plt.show()

if __name__ == '__main__':
    main()
