import os
import tensorflow as tf
from tensorflow import keras
from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import numpy as np
def get_model(model_path):
    if os.path.exists(model_path):
        print(f"Loading model from {model_path}...")
        model = load_model(model_path)
        return model
    
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X = np.vstack([X_train, X_test])
    y = np.hstack([y_train, y_test])
    X = X.astype('float32') / 255.0
    X = X[..., np.newaxis]
    model = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(64, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.25),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X, y, epochs=50, batch_size=64, validation_split=0.2, verbose=1)
    model.save(model_path)
    print(f"Model saved to {model_path}.")
    return model