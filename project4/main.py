import time
import numpy as np
from  matplotlib import pyplot as plt
from img_retrieval import Retrieval


features = np.load('./data/features.npy')
labels = np.load('./data/labels.npy')
start = time.time()
library = Retrieval(features, labels, clutter=12)
library.cal_distance()
avg_time = (time.time()-start)/features.shape[0]
k = list(range(1, 101))
precision, recall, F1, MRR = library.retrieval_for_all(k=k)

plt.figure(1)
plt.title('Evaluation')
plt.plot(k, precision, label='precision')
plt.plot(k, recall, label='recall')
plt.xlabel('k')
plt.legend()

plt.figure(2)
plt.title('F1-score')
plt.plot(k, F1)
plt.xlabel('k')

plt.figure(3)
plt.title('mean reciprocal rank')
plt.plot(k, MRR)
plt.xlabel('k')
plt.show()

print('top:10, 20, 50, 100')
print('P:{:.2f}, {:.2f}, {:.2f}, {:.2f}'.format(precision[9], precision[19], precision[49], precision[99]))
print('R:{:.2f}, {:.2f}, {:.2f}, {:.2f}'.format(recall[9], recall[19], recall[49], recall[99]))
print('F:{:.2f}, {:.2f}, {:.2f}, {:.2f}'.format(F1[9], F1[19], F1[49], F1[99]))
print('M:{:.2f}, {:.2f}, {:.2f}, {:.2f}'.format(MRR[9], MRR[19], MRR[49], MRR[99]))
print('average_time:{:.2f}ms'.format(avg_time*1000))