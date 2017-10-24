import csv
import cv2
import numpy as np
import sklearn
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Conv2D, Cropping2D, Dropout
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


lines = []
with open("./training_data/driving_log.csv") as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

lines.pop(0)
train_samples, validation_samples = train_test_split(lines, test_size=0.2)


def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            measurements = []
            correction = 0.2
            steering_correction = [0, correction, -correction]
            for line in batch_samples:
                for i in range(3):
                    image_path = line[i]
                    image_name = image_path.split('/')[-1]
                    source_path = './training_data/IMG/' + image_name
                    image = cv2.imread(source_path)
                    steering_measurement = float(line[3]) + steering_correction[i]
                    images.append(image)
                    measurements.append(steering_measurement)
                    if i == 0:
                        image_flipped = np.fliplr(image)
                        images.append(image_flipped)
                        measurements.append((-1 * steering_measurement))

            X_train = np.array(images)
            y_train = np.array(measurements)
            yield sklearn.utils.shuffle(X_train, y_train)


# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)


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
model.add(Dropout(0.5))
model.add(Dense(50))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, samples_per_epoch=len(train_samples), validation_data=validation_generator,
                    nb_val_samples=len(validation_samples), nb_epoch=3)
model.save('model.h5')
