import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten,Dense

lines = []
with open("./driving_log.csv") as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images = []
measurements = []
for line in lines:
    image_path = line[0]
    image_name = image_path.split('/')[-1]
    source_path = './Img/' + image_name
    image = cv2.imread(source_path)
    images.append(image)
    measurements.append(float(line[3]))

X_train = np.array(images)
y_train = np.array(measurements)


model = Sequential()
model.add(Flatten(input_shape=(160, 320, 3)))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True)
model.save("model.h5")