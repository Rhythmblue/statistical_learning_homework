import numpy as np
import matplotlib.pyplot as plt


def cart(lmd, x, y, order):
    split_time = 1000
    alpha = 0.001
    pos = [0, x.shape[0]]
    gate = [0, 1]
    tag = [0, 1]
    w = [0]
    time = 0
    while 1 in tag:
        time += 1
        if time>split_time:
            break
        index = tag.index(1)
        if pos[index]-pos[index-1]<100:
            pos[index] = 0
            continue
        threshold, bia_ave, w_now =find_threshold(lmd, x[pos[index-1]: pos[index]], y[pos[index-1]: pos[index]], order)
        pos.insert(index, threshold+pos[index-1])
        gate.insert(index, x[pos[index]])
        w[index - 1] = w_now[0]
        w.insert(index, w_now[1])
        if bia_ave[0]<=alpha:
            tag.insert(index, 0)
        else:
            tag.insert(index, 1)
        if bia_ave[1]<=alpha:
            tag[index+1] = 0
    return gate, w


def plot_cart(x, gate, w, order):
    start = 0
    end = 0
    y = []
    j = 0
    x_poly, y_correct = get_poly(x, order)
    for i in range(1, len(gate)):
        while x[j]<=gate[i]:
            j += 1
            end = j
            if j >= x.shape[0]:
                break
        if start==end:
            continue
        y.append(np.dot(x_poly[start:end], w[i-1]))
        start = end
    return np.concatenate(y), y_correct


def find_threshold(lmd, x, y, order):
    end = x.shape[0]
    x = get_poly(x, order)[0]
    bia_total = 0
    threshold = 1
    bia_ave = []
    w = []
    for i in range(1, end):
        w1 = ridge_regression(lmd, x[0: i], y[0: i])
        bias1 = y[0: i] - np.dot(x[0: i], w1)
        w2 = ridge_regression(lmd, x[i: end], y[i: end])
        bias2 = y[i: end] - np.dot(x[i: end], w2)
        bias1 = np.sum(np.square(bias1))
        bias2 = np.sum(np.square(bias2))
        bia_now = bias1+bias2
        if i==1 or bia_now < bia_total:
            bia_total = bia_now
            bia_ave = [bias1/i, bias2/(end-i)]
            w = [w1, w2]
            threshold = i
    return threshold, bia_ave, w


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
    order = 3
    lmb = 0.05
    x_sample = np.random.rand(100, 25)
    x_sort = np.sort(np.reshape(x_sample, [2500]))
    y_average = np.zeros(101)

    plt.figure(figsize=(15,5))
    plt.suptitle("lambda = " + str(lmb))
    x_train, y_train = get_poly(x_sort, order)
    noise = np.random.normal(0, 0.1, 2500)
    y_train = y_train + noise
    gate, w = cart(lmb, x_sort, y_train, order)
    x_plot = np.arange(2501) / 2500
    y_predict, y_correct = plot_cart(x_plot, gate, w, order)
    plt.subplot(1, 3, 1)
    plt.title("Regression Tree: CART")
    plt.axis([0, 1, -1.2, 1.2])
    plt.plot(x_plot, y_predict, label="cart_result")
    plt.plot(x_plot, y_correct, label="correct_result")


    plt.subplot(1,3,2)
    plt.title("Ridge Regression")
    plt.axis([0,1,-1.2,1.2])
    for i in range(100):
        x_train, y_train = get_poly(x_sample[i], order)
        noise = np.random.normal(0, 0.1, 25)
        y_train = y_train + noise
        w = ridge_regression(lmb, x_train, y_train)
        x_plot = np.arange(101)/100
        x_test, y_correct = get_poly(x_plot, order=order)
        y_predict = np.dot(x_test, w)
        y_average += y_predict
        plt.plot(x_test[:, 1], y_predict)
    y_average = y_average/100
    plt.subplot(1,3,3)
    plt.title("average result of ridge")
    plt.axis([0,1,-1.2,1.2])
    plt.plot(x_test[:,1], y_average, label = "average_result")
    plt.plot(x_test[:,1], y_correct, label = "correct_result")
    plt.legend()

    plt.show()


if __name__ == '__main__':
    main()
