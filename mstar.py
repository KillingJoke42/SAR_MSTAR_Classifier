import pickle
import numpy as np
from tensorflow.keras import models, layers
from tensorflow.keras.callbacks import TensorBoard
from matplotlib import pyplot, image
from sklearn.utils import shuffle
from cv2 import resize
import os
import sys
from datetime import datetime
np.set_printoptions(threshold=sys.maxsize)

traindata_path = os.path.abspath('structure/')

train_data = list()
train_labels = list()

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

car_type = -1
for car in train_candidates[:10]:
	car_type += 1
	for element in os.listdir(car):
		print(os.path.join(car, element))
		if element.endswith('.jpeg') or element.endswith('.JPG'):
			#print(os.path.join(car,image))
			#x = image.imread(os.path.join(car,element))
			#pyplot.imshow(x)
			try:
				img = image.imread(os.path.join(car, element))
				img = resize(img, (128,128))
				train_data.append(img)
				train_labels.append(car_type)
			except Exception as e:
				print('no image')

print(set(train_labels))
train_data = np.array(train_data)
train_labels = np.array(train_labels)

train_data = train_data.reshape((train_data.shape[0], 128, 128, 1))
train_data = train_data / 255.0
train_labels = train_labels.reshape((train_labels.shape[0], 1))

print(train_data.shape)
print(train_labels.shape)

shuffle(train_data, train_labels)

model = models.Sequential()
model.add(layers.Conv2D(32, (5,5), activation="relu", input_shape = (128, 128, 1)))
model.add(layers.MaxPooling2D((2,2)))
#model.add(layers.Dropout(0.25))

model.add(layers.Conv2D(64, (5,5), activation="relu"))
model.add(layers.MaxPooling2D((2,2)))
#model.add(layers.Dropout(0.25))

model.add(layers.Conv2D(64, (5,5), activation="relu"))
model.add(layers.MaxPooling2D((2,2)))
#model.add(layers.Dropout(0.25))

model.add(layers.Conv2D(96, (3,3), activation="relu"))
model.add(layers.MaxPooling2D((2,2)))
#model.add(layers.Dropout(0.25))

model.add(layers.Conv2D(96, (3,3), activation="relu"))
model.add(layers.MaxPooling2D((2,2)))
#model.add(layers.Dropout(0.25))

model.add(layers.Flatten())
model.add(layers.Dense(96, activation="relu"))
#model.add(layers.Dropout(0.25))
model.add(layers.Dense(10, activation="softmax"))

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callbacks = TensorBoard(log_dir=log_dir, histogram_freq=1)
model.fit(train_data, train_labels, epochs=20, validation_split=0.1, callbacks=[tensorboard_callbacks])

model.save('mstar.h5')
