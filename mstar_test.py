import os
import numpy as np
from sklearn.utils import shuffle
from matplotlib import image
from tensorflow.keras import models
from sklearn.metrics.classification import confusion_matrix

testdata_path = os.path.abspath('dataset/TARGETS/TEST/15_DEG')
print(testdata_path)

BMP_path = [os.path.join(testdata_path, "BMP2/SN_C21/jpeg"), 
			os.path.join(testdata_path, "BMP2/SN_9566/jpeg"), 
			os.path.join(testdata_path, "BMP2/SN_9563/jpeg")]

BTR_path = [os.path.join(testdata_path, "BTR70/SN_C71/jpeg")]

T_path = [os.path.join(testdata_path, "T72/SN_132/jpeg"), 
		  os.path.join(testdata_path, "T72/SN_812/jpeg"), 
		  os.path.join(testdata_path, "T72/SN_S7/jpeg")]

model = models.load_model('mstar.h5')

test_data = list()
test_labels = list()

car_type = 0
for car in [BMP_path, BTR_path, T_path]:
	for path in car:
		for img_index in range(2400):
			try:
				img = image.imread(os.path.join(path, '{}.jpeg'.format(img_index)))
				test_data.append(img)
				test_labels.append(car_type)
			except:
				print('No Image')
	car_type += 1

test_data = np.array(test_data)
test_labels = np.array(test_labels)
print(test_data.shape)
print(test_labels.shape)
test_data = test_data.reshape((test_data.shape[0], test_data.shape[1], test_data.shape[2], 1))
test_labels = test_labels.reshape((test_labels.shape[0], 1))

loss, acc = model.evaluate(test_data, test_labels)
print(loss)
print(acc)

pred = model.predict_classes(test_data)
confusion = confusion_matrix(test_labels, pred)
print(confusion)