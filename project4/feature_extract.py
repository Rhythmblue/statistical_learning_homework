import os
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.applications.inception_v3 import preprocess_input
from keras.models import Model
import numpy as np

data_dir = './proj4_data'

base_model = InceptionV3(weights='imagenet')
model = Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)

features = []
labels = []
label_names = []
label = 0
f = open('./data/img_path.txt', 'w')
for label_name in os.listdir(data_dir):
    label_names.append(label_name)
    print(label_name)
    for img_name in os.listdir(os.path.join(data_dir, label_name)):
        img_path = os.path.join(data_dir, label_name, img_name)
        f.write(img_path + '\n')
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        features.append(model.predict(x))
        labels.append(label)
        print(label)
    label += 1
f.close()

f = open('./data/label_map.txt', 'w')
for label_name in label_names:
    f.write(label_name+'\n')
f.close()

features = np.concatenate(features)
labels = np.stack(labels)
print(features.shape)
print(labels.shape)

np.save('./data/features.npy', features)
np.save('./data/labels.npy', labels)