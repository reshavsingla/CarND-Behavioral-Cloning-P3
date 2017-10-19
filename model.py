import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten,Dense,Lambda

lines = []
with open("./data/driving_log.csv") as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

lines.pop(0)
images = []
measurements = []
for line in lines:
    image_path = line[0]
    image_name = image_path.split('/')[-1]
    source_path = './data/IMG/' + image_name
    image = cv2.imread(source_path)
    images.append(image)
    measurements.append(float(line[3]))

X_train = np.array(images)
y_train = np.array(measurements)


model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Flatten())
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=5)
model.save('model.h5')
