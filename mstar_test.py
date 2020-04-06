import os
import numpy as np
from cv2 import resize
from sklearn.utils import shuffle
from matplotlib import image
from tensorflow.keras import models
from sklearn.metrics.classification import confusion_matrix

traindata_path = os.path.abspath('structure/')

BMP_path = os.path.join(traindata_path, "BMP2/") 
BTR_path = os.path.join(traindata_path, "BTR70/")
T_path = os.path.join(traindata_path, "T72/")
BTR60_path = os.path.join(traindata_path, "BTR60/")
_2S1_path = os.path.join(traindata_path, "2S1/")
BRDM_path = os.path.join(traindata_path, "BRDM_2/")
D7_path = os.path.join(traindata_path, "D7/")
T62_path = os.path.join(traindata_path, "T62/")
ZIL131_path = os.path.join(traindata_path, "ZIL131/")
ZSU_23_4_path = os.path.join(traindata_path, "ZSU_23_4/")
train_candidates = [BMP_path, BTR_path, T_path, BTR60_path, _2S1_path, BRDM_path, D7_path, T62_path, ZIL131_path, ZSU_23_4_path]

model = models.load_model('mstar.h5')

train_data = list()
train_labels = list()

car_type = -1
for car in train_candidates[:10]:
	car_type += 1
	for element in os.listdir(car):
		print(os.path.join(car, element))
		if element.endswith('.jpeg') or element.endswith('.JPG'):
			try:
				img = image.imread(os.path.join(car, element))
				img = resize(img, (128,128))
				train_data.append(img)
				train_labels.append(car_type)
			except Exception as e:
				print('no image: {}'.format(e))

train_data = np.array(train_data)
train_labels = np.array(train_labels)
print(train_data.shape)
print(train_labels.shape)
train_data = train_data.reshape((train_data.shape[0], train_data.shape[1], train_data.shape[2], 1))
train_labels = train_labels.reshape((train_labels.shape[0], 1))

loss, acc = model.evaluate(train_data, train_labels)
print(loss)
print(acc)

pred = model.predict_classes(train_data)
confusion = confusion_matrix(train_labels, pred)
print(confusion)