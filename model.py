import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Conv2D, Cropping2D

lines = []
with open("./data/driving_log.csv") as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

lines.pop(0)
images = []
measurements = []
correction = 0.2
steering_correction = [0, correction, -correction]
for line in lines:
    for i in range(3):
        image_path = line[i]
        image_name = image_path.split('/')[-1]
        source_path = './data/IMG/' + image_name
        image = cv2.imread(source_path)
        steering_measurement = float(line[3]) + steering_correction[i]
        images.append(image)
        measurements.append(steering_measurement)
        image_flipped = np.fliplr(image)
        images.append(image_flipped)
        measurements.append((-1 * steering_measurement))

X_train = np.array(images)
y_train = np.array(measurements)


model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70, 25), (0, 0))))
model.add(Conv2D(24, (5, 5), strides=(2, 2), activation="relu"))
model.add(Conv2D(36, (5, 5), strides=(2, 2), activation="relu"))
model.add(Conv2D(48, (5, 5), strides=(2, 2), activation="relu"))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=5)
model.save('model.h5')
