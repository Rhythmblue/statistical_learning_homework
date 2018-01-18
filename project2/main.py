import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from Binary_Classification import Binary_Classification

train_x = []
train_y = []
test_x = []
test_p = []
test_y = []
with open('proj2_data/xtrain.txt', 'r') as f:
    for line in f.readlines():
        line = [float(x) for x in line.strip().split(',')]
        train_x.append(line)

with open('proj2_data/ctrain.txt', 'r') as f:
    for line in f.readlines():
        line = int(line.strip())
        train_y.append(line)

with open('proj2_data/xtest.txt', 'r') as f:
    for line in f.readlines():
        line = [float(x) for x in line.strip().split(',')]
        test_x.append(line)

with open('proj2_data/ptest.txt', 'r') as f:
    for line in f.readlines():
        line = float(line.strip())
        test_p.append(line)

with open('proj2_data/c1test.txt', 'r') as f:
    for line in f.readlines():
        line = float(line.strip())
        test_y.append(line)

train_x = np.array(train_x)
train_y = np.array(train_y)
test_x = np.array(test_x)
test_p = np.array(test_p)
test_y = np.array(test_y)



# plt.show()

bc = Binary_Classification(train_x, train_y, test_x, test_p, test_y, k=5)
order_range = np.arange(1, 21)
lmd_range = np.arange(0, 6)

# linear regression
error_rate = []
for order in order_range:
    error_rate.append(bc.run_ridge_regression(order=order, lmd=0, mode='val'))
plt.figure(1)
plt.title('linear regression')
plt.xticks(np.arange(21))
plt.xlabel('order')
plt.ylabel('error rate')
plt.plot(order_range, error_rate, '-o')
print('1. linear regression: {:.2f}'.format(min(error_rate)))

#ridge regression on val data
X, Y = np.meshgrid(order_range, lmd_range)
Z = np.zeros(X.shape)
for i, lmd in enumerate(lmd_range):
    for j, order in enumerate(order_range):
        Z[i, j] = bc.run_ridge_regression(order=order, lmd=10.0**(-lmd), mode='val')
fig = plt.figure(2)
ax = Axes3D(fig)
ax.set_title('ridge regression on val data')
surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow')
ax.set_xticks(np.arange(0, 21, 5))
ax.set_xlabel('order')
ax.set_ylabel('lambda(10^-n)')
ax.set_zlabel('error rate')
fig.colorbar(surf, shrink=0.5)

#ridge regression on test data
X, Y = np.meshgrid(order_range, lmd_range)
Z = np.zeros(X.shape)
for i, lmd in enumerate(lmd_range):
    for j, order in enumerate(order_range):
        Z[i, j] = bc.run_ridge_regression(order=order, lmd=10.0**(-lmd), mode='test')
fig = plt.figure(3)
ax = Axes3D(fig)
ax.set_title('ridge regression on test data')
surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow')
ax.set_xticks(np.arange(0, 21, 5))
ax.set_xlabel('order')
ax.set_ylabel('lambda(10^-n)')
ax.set_zlabel('error rate')
fig.colorbar(surf, shrink=0.5)
print('2. ridge regression: {:.2f}'.format(np.min(Z)))

#logistic regression to select model
bic = []
error_rate = []
for order in order_range:
    tmp = bc.run_logistic_regression(order, iteration=500, learning_rate=0.05, mode='val')
    bic.append(tmp[0])
    error_rate.append(tmp[1])
plt.figure(4)
plt.title('logistic regression under bic')
plt.plot(order_range, bic, '-o')
plt.xticks(np.arange(21))
plt.xlabel('order')
plt.ylabel('model evidence')

plt.figure(5)
plt.title('logistic regression on train data')
plt.plot(order_range, error_rate, '-o')
plt.xticks(np.arange(21))
plt.xlabel('order')
plt.ylabel('error rate')

error_rate = []
for order in order_range:
    error = bc.run_logistic_regression(order, iteration=500, learning_rate=0.05, mode='test')
    error_rate.append(error)
plt.figure(6)
plt.title('logistic regression on test data')
plt.plot(order_range, error_rate, '-o')
plt.xticks(np.arange(21))
plt.xlabel('order')
plt.ylabel('error rate')
print('3. logistic regression: {:.2f}'.format(min(error_rate)))

# newton_raphson
error_rate = []
for order in order_range:
    error = bc.run_newton_raphson(order, iteration=500)
    error_rate.append(error)
plt.figure(7)
plt.title('logistic regression with newton raphson way')
plt.plot(order_range, error_rate, '-o')
plt.xticks(np.arange(21))
plt.xlabel('order')
plt.ylabel('error rate')
print('4. newton raphson: {:.2f}'.format(min(error_rate)))

# lasso regression
X, Y = np.meshgrid(order_range, lmd_range)
Z = np.zeros(X.shape)
for i, lmd in enumerate(lmd_range):
    for j, order in enumerate(order_range):
        Z[i, j] = bc.run_lasso_regression(order=order, lmd=10.0**(-lmd))
fig = plt.figure(8)
ax = Axes3D(fig)
ax.set_title('lasso regression on test data')
surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow')
ax.set_xticks(np.arange(0, 21, 5))
ax.set_xlabel('order')
ax.set_ylabel('lambda(10^-n)')
ax.set_zlabel('error rate')
fig.colorbar(surf, shrink=0.5)
print('5. lasso regression: {:.2f}'.format(np.min(Z)))

# linear regression under bayesian framework
error_rate = []
for order in order_range:
    error_rate.append(bc.run_ridge_regression(order=order, lmd=0, mode='ml'))
plt.figure(9)
plt.title('linear regression under bayesian framework')
plt.xticks(np.arange(21))
plt.xlabel('order')
plt.ylabel('model evidence')
plt.plot(order_range, error_rate, '-o')

# fisher's LDA
w, _ = bc.run_fisher_lda(1)
plt.figure(10)
plt.title('The distribution of raw data(order=1)')
train_y_0 = np.where(train_y == 0)
train_y_1 = np.where(train_y == 1)
plt.scatter(train_x[:, 0][train_y_0], train_x[:, 1][train_y_0], c = 'red', label='0')
plt.scatter(train_x[:, 0][train_y_1], train_x[:, 1][train_y_1], c = 'blue', label='1')

x_range = np.linspace(-4,4, 50)
x1_range = np.linspace(-0.5, 0.5, 20)
w_range = (w[2]/w[1])*x1_range
y_range = -(w[1]/w[2])*x_range - (w[0]/w[2])
plt.plot(x1_range, w_range, linestyle='--', c = 'orange', label='projected line')
plt.plot(x_range, y_range, linewidth=4, c = 'pink', label='discriminant line')
plt.xlim(-4, 4)
plt.ylim(-4, 4)
plt.xlabel('x1')
plt.ylabel('x2')
# ax = plt.gca()
# ax.spines['right'].set_color('none')
# ax.spines['top'].set_color('none')
# ax.spines['bottom'].set_position(('data', 0))
# ax.spines['left'].set_position(('data',0))
plt.legend()

error_rate = []
for order in order_range:
    _, error = bc.run_fisher_lda(order)
    error_rate.append(error)
plt.figure(11)
plt.title('fisher\'s LDA')
plt.plot(order_range, error_rate, '-o')
plt.xticks(np.arange(21))
plt.xlabel('order')
plt.ylabel('error rate')
print('6. fisher\'s LDA: {:.2f}'.format(min(error_rate)))

# k-means
train_index_0 = np.where(train_y==0)
train_index_1 = np.where(train_y==1)
plt.figure(12)
plt.title('distribution of raw data')
plt.scatter(train_x[:, 0][train_index_0], train_x[:, 1][train_index_0], label='0')
plt.scatter(train_x[:, 0][train_index_1], train_x[:, 1][train_index_1], label='1')
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()

# multi-class
n_clusters = 6
train_label, test_label, predict, error_rate = bc.run_multi_class(order=1, n_clusters=n_clusters)
plt.figure(13)
plt.title('distribution after k-means')
for i in range(n_clusters):
    index = np.where(train_label==i)
    dot = train_x[index]
    plt.scatter(dot[:, 0], dot[:, 1], label=str(i))
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()

plt.figure(14)
plt.title('distribution after one-vs-rest classification')
for i in range(n_clusters):
    index = np.where(test_label==i)
    dot = test_x[index]
    plt.scatter(dot[:, 0], dot[:, 1], label=str(i))
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()

plt.figure(15)
plt.title('distribution after convert to 0-1')
for i in range(2):
    index = np.where(predict==i)
    dot = test_x[index]
    plt.scatter(dot[:, 0], dot[:, 1], label=str(i))
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()

print('7. muti-class: {:.2f}'.format(error_rate))

plt.show()