import numpy as np
import matplotlib.pyplot as plt

def raw_function(x):
    return np.sin(2 * np.pi * x)


def get_poly(x_raw, order):
    num_sample = x_raw.shape[0]
    x_poly = np.zeros([num_sample, order + 1])
    y = np.zeros(num_sample)
    for i in range(num_sample):
        for j in range(order+1):
            x_poly[i, j] = x_raw[i] ** j
        y[i] = raw_function(x_poly[i, 1])
    return x_poly, y


def ridge_regression(lmd, x, y):
    xTx = np.dot(x.T, x)
    m, n = xTx.shape
    return np.dot(np.linalg.inv(xTx+lmd*np.eye(m,n)), np.dot(x.T, y))


def main():
    order = 7
    n = -10
    lmb = np.e**n
    y_average = np.zeros(101)
    plt.figure(figsize=(10,5))
    plt.suptitle("ln(lambda) = "+str(n))
    plt.subplot(1,2,1)
    plt.title("Ridge regression")
    plt.axis([0,1,-1.2,1.2])
    x_sample = np.random.rand(100, 25)
    for i in range(100):
        x_sample = np.sort(x_sample)
        x_train, y_train = get_poly(x_sample[i], order=order)

        noise = np.random.normal(0, 0.1, 25)
        y_train = y_train + noise

        w = ridge_regression(lmb, x_train, y_train)

        x_plot = np.arange(101)/100
        x_test, y_correct = get_poly(x_plot, order=order)
        y_predict = np.dot(x_test, w)
        y_average += y_predict
        plt.plot(x_test[:, 1], y_predict)
    y_average = y_average/100
    plt.subplot(1,2,2)
    plt.title("compare")
    plt.axis([0,1,-1.2,1.2])
    plt.plot(x_test[:,1], y_average, label = "average_result")
    plt.plot(x_test[:,1], y_correct, label = "correct_result")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()