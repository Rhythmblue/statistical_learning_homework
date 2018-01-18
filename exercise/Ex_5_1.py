import numpy as np
from matplotlib import pyplot as plt


def perceptron_origin(x, y, w, rate):
    num = x.shape[0]
    flag = False
    for i in range(100):
        j = i % num
        if y[j] != np.sign(np.dot(x[j], w)):
            w += rate * y[j] * np.reshape(x[j], [3, 1])
        if np.sum(np.sign(np.dot(x, w)) - y) == 0:
            flag = True
            break
    if not flag:
        print('can\'t find proper w' )
    return w

def perceptron_dual(x, y, rate):
    num, dim = x.shape
    alpha = np.zeros([num, 1])
    flag = False
    w = np.sum(np.tile(alpha * y, [1, dim]) * x, axis=0).T
    for i in range(100):
        j = i % num
        if y[j] * np.dot(x[j], w) <= 0 :
            alpha[j] += rate
            w = np.sum(np.tile(alpha * y, [1, dim]) * x, axis=0).T
        if np.sum(np.sign(np.dot(x, w)) - y) == 0:
            flag = True
            break
    if not flag:
        print('can\'t find proper w' )
    w = np.sum(np.tile(alpha * y, [1, dim]) * x, axis=0).T
    return w


x = np.array([[1,3,3], [1,4,3], [1,1,1]])
y = np.array([[1], [1], [-1]])
w = np.zeros([3, 1])
rate = 1

w = perceptron_origin(x, y, w, rate)
w1 = perceptron_dual(x,y, rate)
print(w)
plt.figure(1)
plt.title('Origin form of perceptron learning')
for i in range(x.shape[0]):
    color = 'r' if y[i]==1 else 'b'
    plt.plot(x[i][1], x[i][2], '.' + color)
x_range = np.array([0, 5])
y_range = - (w[1]/w[2])*x_range - w[0]/w[2]

plt.plot(x_range, y_range)
plt.xlabel('x1')
plt.ylabel('x2')

plt.figure(2)
plt.title('Dual form of perceptron learning')
for i in range(x.shape[0]):
    color = 'r' if y[i]==1 else 'b'
    plt.plot(x[i][1], x[i][2], '.' + color)
x_range = np.array([0, 5])
y_range = - (w1[1]/w1[2])*x_range - w1[0]/w1[2]

plt.plot(x_range, y_range)
plt.xlabel('x1')
plt.ylabel('x2')

plt.show()