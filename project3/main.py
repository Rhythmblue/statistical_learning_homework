from user_and_movie import User_and_Movie

clf = User_and_Movie()
mode = ['movie_feature', 'genre_feature', 'genre_mask_feature']
target = ['gender', 'age']

print('movie_feature')
avg_error = []
for i in range(10):
    error_rate, train_time, test_time = clf.estimator(fold=i, mode=mode[0], target=target[0])
    print('fold: {:d}, train_time: {:.2f}, test_time: {:.2f}, gender_error_rate: {:.4f}'.format(i, train_time, test_time, error_rate))
    avg_error.append(error_rate)
print('avg_error: {:.4f}'.format(sum(avg_error)/10))

print('genre_feature')
avg_error = []
for i in range(10):
    error_rate, train_time, test_time = clf.estimator(fold=i, mode=mode[1], target=target[0])
    print('fold: {:d}, train_time: {:.2f}, test_time: {:.2f}, gender_error_rate: {:.4f}'.format(i, train_time, test_time, error_rate))
    avg_error.append(error_rate)
print('avg_error: {:.4f}'.format(sum(avg_error)/10))

print('genre_mask_feature')
avg_error = []
for i in range(10):
    error_rate, train_time, test_time = clf.estimator(fold=i, mode=mode[2], target=target[0])
    print('fold: {:d}, train_time: {:.2f}, test_time: {:.2f}, gender_error_rate: {:.4f}'.format(i, train_time, test_time, error_rate))
    avg_error.append(error_rate)
print('avg_error: {:.4f}'.format(sum(avg_error)/10))

print('movie_feature+genre_feature')
avg_error = []
for i in range(10):
    error_rate, train_time, test_time = clf.estimator(fold=i, mode=mode[0]+'+'+mode[1], target=target[0])
    print('fold: {:d}, train_time: {:.2f}, test_time: {:.2f}, gender_error_rate: {:.4f}'.format(i, train_time, test_time, error_rate))
    avg_error.append(error_rate)
print('avg_error: {:.4f}'.format(sum(avg_error)/10))



print('movie_feature')
avg_error = []
for i in range(10):
    error_rate, train_time, test_time = clf.estimator(fold=i, mode=mode[0], target=target[1])
    print('fold: {:d}, train_time: {:.2f}, test_time: {:.2f}, age_error_rate: {:.4f}'.format(i, train_time, test_time, error_rate))
    avg_error.append(error_rate)
print('avg_error: {:.4f}'.format(sum(avg_error)/10))

print('genre_feature')
avg_error = []
for i in range(10):
    error_rate, train_time, test_time = clf.estimator(fold=i, mode=mode[1], target=target[1])
    print('fold: {:d}, train_time: {:.2f}, test_time: {:.2f}, age_error_rate: {:.4f}'.format(i, train_time, test_time, error_rate))
    avg_error.append(error_rate)
print('avg_error: {:.4f}'.format(sum(avg_error)/10))

print('genre_mask_feature')
avg_error = []
for i in range(10):
    error_rate, train_time, test_time = clf.estimator(fold=i, mode=mode[2], target=target[1])
    print('fold: {:d}, train_time: {:.2f}, test_time: {:.2f}, age_error_rate: {:.4f}'.format(i, train_time, test_time, error_rate))
    avg_error.append(error_rate)
print('avg_error: {:.4f}'.format(sum(avg_error)/10))

print('movie_feature+genre_feature')
avg_error = []
for i in range(10):
    error_rate, train_time, test_time = clf.estimator(fold=i, mode=mode[0]+'+'+mode[1], target=target[1])
    print('fold: {:d}, train_time: {:.2f}, test_time: {:.2f}, age_error_rate: {:.4f}'.format(i, train_time, test_time, error_rate))
    avg_error.append(error_rate)
print('avg_error: {:.4f}'.format(sum(avg_error)/10))

