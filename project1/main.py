import numpy as np
from knn import KNN

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
accuracy = wine_knn.run_knn(5)
print(accuracy)