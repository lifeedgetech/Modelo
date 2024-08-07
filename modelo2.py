import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from sklearn.preprocessing import LabelBinarizer

def binarize_labels(y):
    lb = LabelBinarizer()
    return lb.fit_transform(y)

X_train = np.random.rand(100, 128, 128, 3)
y_train = np.random.randint(0, 2, 100)
X_test = np.random.rand(20, 128, 128, 3)
y_test = np.random.randint(0, 2, 20) 

y_train = binarize_labels(y_train)
y_test = binarize_labels(y_test)

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')  
])

model.compile(optimizer=Adam(), loss=BinaryCrossentropy(), metrics=['accuracy'])

datagen = ImageDataGenerator(rescale=1./255)

train_generator = datagen.flow(X_train, y_train, batch_size=32)
validation_generator = datagen.flow(X_test, y_test, batch_size=32)

history = model.fit(train_generator, epochs=50, validation_data=validation_generator)

model.save('parkinson_mri_cnn_model.h5')
