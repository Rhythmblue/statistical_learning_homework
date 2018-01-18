import numpy as np
from knn import KNN
import matplotlib.pyplot as plt
import pandas as pd
import pandas.plotting as pd_plot

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
plt.title('k-NN')
plt.plot(list(range(1, 141)), accuracy, '-_', label='2-norm')
plt.xlabel('k')
plt.ylabel('accuracy')
plt.grid(True)
accuracy = np.array(accuracy)
print('1.K-NN: k:{:d}, accuracy:{:.3f}'.format(np.argmax(accuracy)+1, np.max(accuracy)))


# k-NN with pca
accuracy = []
for k in range(1, 141):
    accuracy.append(wine_knn.run_knn(k, pca=6))
plt.figure(1)
#plt.title('k-NN with pca')
plt.plot(list(range(1, 141)), accuracy, '-_', label='2-norm+PCA')
plt.xlabel('k')
plt.ylabel('accuracy')
plt.grid(True)
accuracy = np.array(accuracy)
print('2.K-NN with pca: k:{:d}, accuracy:{:.3f}'.format(np.argmax(accuracy)+1, np.max(accuracy)))

# visualize the data
wine_plot = pd.read_csv('proj1_data/wine.csv', index_col=False)
plt.figure(3)
plt.title('andrew\'s curves')
pd_plot.andrews_curves(wine_plot, 'Class')

tmp = wine_plot[wine_plot.columns[1:]]
wine_plot[wine_plot.columns[1:]] = tmp.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
plt.figure(4)
plt.title('parallel coordinates')
pd_plot.parallel_coordinates(wine_plot, 'Class')

# k-NN with normalization
accuracy = []
for k in range(1, 141):
    accuracy.append(wine_knn.run_knn(k, normalization=True))
plt.figure(1)
#plt.title('k-NN with normalization')
plt.plot(list(range(1, 141)), accuracy, '-_', label='2-norm+normalization')
plt.xlabel('k')
plt.ylabel('accuracy')
plt.grid(True)
accuracy = np.array(accuracy)
print('3.K-NN with normalization: k:{:d}, accuracy:{:.3f}'.format(np.argmax(accuracy)+1, np.max(accuracy)))

# k-NN with 1-norm
accuracy = []
for k in range(1, 141):
    accuracy.append(wine_knn.run_knn(k, normalization=False, norm=1))
plt.figure(1)
#plt.title('k-NN with 1-norm')
plt.plot(list(range(1, 141)), accuracy, '-_', label='1-norm')
plt.xlabel('k')
plt.ylabel('accuracy')
plt.grid(True)
accuracy = np.array(accuracy)
print('4.K-NN with 1-norm: k:{:d}, accuracy:{:.3f}'.format(np.argmax(accuracy)+1, np.max(accuracy)))

# k-NN with 1-norm and normalization
accuracy = []
for k in range(1, 141):
    accuracy.append(wine_knn.run_knn(k, normalization=True,norm=1))
plt.figure(1)
#plt.title('k-NN with 1-norm and normalization')
plt.plot(list(range(1, 141)), accuracy, '-_', label='1-norm+normalization')
plt.xlabel('k')
plt.ylabel('accuracy')
plt.grid(True)
accuracy = np.array(accuracy)
print('5.K-NN with 1-norm and normalization: k:{:d}, accuracy:{:.3f}'.format(np.argmax(accuracy)+1, np.max(accuracy)))

plt.legend()
plt.show()