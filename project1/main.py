import numpy as np
from knn import KNN
import matplotlib.pyplot as plt

data_dir = './proj1_data/wine.data'
wine_data = []
wine_label = []
with open(data_dir, 'r') as f:
    for line in f.readlines():
        data_now = line.strip().split(',')
        wine_data.append(np.array([float(x) for x in data_now[1:]]))
        wine_label.append(int(data_now[0]))
    wine_data = np.stack(wine_data)
    wine_label = np.array(wine_label)

wine_knn = KNN(wine_data, wine_label)
wine_knn.split_data(5)

# k-NN
accuracy = []
for k in range(1, 141):
    accuracy.append(wine_knn.run_knn(k))
plt.figure(1)
plt.title('k-NN classifier')
plt.plot(list(range(1, 141)), accuracy, '-_')
plt.xlabel('k')
plt.ylabel('accuracy')
plt.grid(True)

# k-NN with pca
accuracy = []
for k in range(1, 141):
    accuracy.append(wine_knn.run_knn(k, pca=6))
plt.figure(2)
plt.title('k-NN with pca')
plt.plot(list(range(1, 141)), accuracy, '-_')
plt.xlabel('k')
plt.ylabel('accuracy')
plt.grid(True)

# k-NN with normalization
accuracy = []
for k in range(1, 141):
    accuracy.append(wine_knn.run_knn(k, normalization=True))
plt.figure(3)
plt.title('k-NN with normalization')
plt.plot(list(range(1, 141)), accuracy, '-_')
plt.xlabel('k')
plt.ylabel('accuracy')
plt.grid(True)

plt.show()