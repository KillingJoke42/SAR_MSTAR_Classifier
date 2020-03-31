import pickle
import numpy as np
from tensorflow.keras import models, layers
from matplotlib import pyplot, image
from sklearn.utils import shuffle
import os

traindata_path = os.path.abspath('dataset/TARGETS/TRAIN/17_DEG')
print(traindata_path)

train_data = list()
train_labels = list()

BMP_path = [os.path.join(traindata_path, "BMP2/SN_C21/jpeg"), 
			os.path.join(traindata_path, "BMP2/SN_9566/jpeg"), 
			os.path.join(traindata_path, "BMP2/SN_9563/jpeg")]

BTR_path = [os.path.join(traindata_path, "BTR70/SN_C71/jpeg")]

T_path = [os.path.join(traindata_path, "T72/SN_132/jpeg"), 
		  os.path.join(traindata_path, "T72/SN_812/jpeg"), 
		  os.path.join(traindata_path, "T72/SN_S7/jpeg")]

car_type = 0
for car in [BMP_path, BTR_path, T_path]:
	for path in car:
		for img_index in range(300):
			try:
				img = image.imread(os.path.join(path, '{}.jpeg'.format(img_index)))
				train_data.append(img)
				train_labels.append(car_type)
			except:
				print('No Image')
	car_type += 1

train_data = np.array(train_data)
train_labels = np.array(train_labels)

train_data = train_data.reshape((1622, 128, 128, 1))
train_data = train_data / 255.0
train_labels = train_labels.reshape((1622, 1))

print(train_data.shape)
print(train_labels.shape)

shuffle(train_data, train_labels)

model = models.Sequential()
model.add(layers.Conv2D(32, (5,5), activation="relu", input_shape = (128, 128, 1)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (5,5), activation="relu"))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (5,5), activation="relu"))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(96, (3,3), activation="relu"))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(96, (3,3), activation="relu"))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(96, activation="relu"))
model.add(layers.Dense(3, activation="softmax"))

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(train_data, train_labels, epochs=20, validation_split=0.1)

model.save('mstar.h5')