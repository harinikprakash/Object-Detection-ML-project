

import tensorflow as tf
import os
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, AveragePooling2D, BatchNormalization, Activation
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint

# Image augmentation for training
train = ImageDataGenerator(
    rescale=1./255,
    rotation_range=25,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Only rescaling for validation
validation = ImageDataGenerator(rescale=1./255)

# Load training and validation datasets
train_dataset = train.flow_from_directory("C:/Users/Dell/Desktop/ML2/basedata2/train2/",
                                          target_size=(200, 200),
                                          batch_size=16,
                                          class_mode="binary",
                                          shuffle=True)

validation_dataset = validation.flow_from_directory("C:/Users/Dell/Desktop/ML2/basedata2/val2/",
                                                    target_size=(200, 200),
                                                    batch_size=16,
                                                    class_mode="binary",
                                                    shuffle=False)

# Swish activation function
def swish(x):
    return x * tf.nn.sigmoid(x)

# Model definition
model = Sequential([
    Conv2D(16, (3, 3), padding='same', input_shape=(200, 200, 3)),
    Activation(swish),
    BatchNormalization(),
    AveragePooling2D(2, 2),

    Conv2D(32, (3, 3), padding='same'),
    Activation(swish),
    BatchNormalization(),
    AveragePooling2D(2, 2),

    Conv2D(64, (3, 3), padding='same'),
    Activation(swish),
    BatchNormalization(),
    AveragePooling2D(2, 2),

    Conv2D(128, (3, 3), padding='same'),
    Activation(swish),
    BatchNormalization(),
    AveragePooling2D(2, 2),

    Flatten(),
    Dense(512),
    Activation(swish),
    Dropout(0.4),  # Slightly increased for better regularization
    Dense(1, activation="sigmoid")
])

# Compile the model
model.compile(
    loss="binary_crossentropy",
    optimizer=RMSprop(learning_rate=0.0003),  # Lower learning rate to help model distinguish better
    metrics=["accuracy"]
)

# Model checkpointing (best model will still be saved)
checkpoint = ModelCheckpoint("best_model.h5", monitor='val_loss', save_best_only=True)

# Train the model for full 50 epochs
model_fit = model.fit(
    train_dataset,
    epochs=50,
    validation_data=validation_dataset,
    callbacks=[checkpoint]
)

# Load best weights after training
model.load_weights("best_model.h5")


# In[6]:


# Prediction
dir_path = "C:/Users/Dell/Desktop/ML2/basedata2/test2"

for i in os.listdir(dir_path):
    img_path = os.path.join(dir_path, i)
    img = load_img(img_path, target_size=(200, 200))
    plt.imshow(img)
    plt.show()

    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x /= 255.0

    prediction = model.predict(x)

    if prediction[0] < 0.5:
        print("COW")
    else:
        print("PATEL")


