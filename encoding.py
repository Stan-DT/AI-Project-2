# Import necessary libraries
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Reshape
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers, models
import pandas as pd
import pyaml
import yaml


class Encoding:

    def __init__(self,x_train, y_train, x_test, y_test):

        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

        self.x_train_split = None
        self.y_train_split = None
        self.y_train_converted = None
        self.y_val_converted = None
        self.y_test_converted = None

        self.loss_fn = None
        self.seed_np = None
        self.seed_tf = None
        self.latent_dim = None

    def set_seed(self,seed):
        self.seed_np = np.random.seed(seed)
        self.seed_tf = tf.random.set_seed(seed)
    
    def set_latent_dim(self,latent_dim):
        self.latent_dim = latent_dim
    
    def set_loss_function(self,loss_fn):
        self.loss_fn = loss_fn

    def split_to_val(self,test_size):
            self.x_train_split, self.x_val, self.y_train_split, self.y_val = train_test_split(self.x_train, 
                                                                                          self.y_train, 
                                                                                          random_state=42,
                                                                                          test_size=test_size)
                    
    def normalize_data(self):
        self.x_train_split = self.x_train_split/ 255.0
        self.x_val = self.x_val / 255.0
        self.x_test = self.x_test / 255.0

    def reshaper(self):
        
        self.x_train_split = self.x_train_split.reshape((x_train.shape[0], (x_train.shape[1]*x_train.shape[2])))
        self.x_val = self.x_val.reshape((self.x_val.shape[0], (self.x_val.shape[1]*self.x_val.shape[2])))

    
    

    

# Define reconstruction metric and set random seed

np.random.seed(36)

# Load and preprocess data
(x_train, _), (x_val, _) = fashion_mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_val = x_val.astype('float32') / 255.
x_train = x_train.reshape((x_train.shape[0], 784))
x_val = x_val.reshape((x_val.shape[0], 784))

# Define initial model architecture
latent_dim = (4, 4, 32)
inputs = tf.keras.Input(shape=(784,))
encoded = Flatten()(inputs)
encoded = Dense(units=latent_dim[0] * latent_dim[1] * latent_dim[2], activation='relu')(encoded)
decoded = Dense(units=784, activation='sigmoid')(encoded)
autoencoder = Model(inputs, decoded)

# Print model architecture
autoencoder.summary()

# Train initial model and visualize learning curves
history = autoencoder.compile(optimizer='adam', loss=loss_fn).fit(x_train, x_train,
                                                                epochs=20, validation_data=(x_val, x_val))
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.legend()
plt.show()

# Fine-tune model and evaluate
autoencoder.compile(optimizer='adam', loss=loss_fn)
autoencoder.fit(x_train, x_train, epochs=10, validation_data=(x_val, x_val))
val_loss = autoencoder.evaluate(x_val, x_val)
print("Validation loss:", val_loss)

# Reconstruct and display validation samples
decoded_imgs = autoencoder.predict(x_val)
plt.figure(figsize=(10, 5))
for i in range(5):
    # Reshape and normalize reconstructed image
    decoded_img = decoded_imgs[i].reshape(28, 28) * 255
    # Original image
    original_img = x_val[i].reshape(28, 28) * 255
    plt.subplot(1, 10, 2*i+1)
    plt.imshow(original_img, cmap='gray')
    plt.subplot(1, 10, 2*i+2)
    plt.imshow(decoded_img, cmap='gray')
plt.show()

# Experiment with decreasing feature dimension
latent_dim = (4, 4, 28)
encoded = Dense(units=latent_dim[0] * latent_dim[1] * latent_dim[2], activation='relu')(encoded)
autoencoder = Model(inputs, decoded)
autoencoder.compile(optimizer='adam', loss=loss_fn)

# Train and evaluate the model
history = autoencoder.fit(x_train, x_train, epochs=20, validation_data=(x_val, x_val))
val_loss = autoencoder.evaluate(x_val, x_val)
print("Validation loss:", val_loss)

# Reconstruct and display validation samples
decoded_imgs = autoencoder.predict(x_val)
plt.figure(figsize=(10, 5))
for i in range(5):
    decoded_img = decoded_imgs[i].reshape(28, 28) * 255
    original_img = x_val[i].reshape(28, 28) * 255
    plt.subplot(1, 10, 2*i+1)
    plt.imshow(original_img, cmap='gray')
    plt.subplot(1, 10, 2*i+2)
    plt.imshow(decoded_img, cmap='gray')
plt.show()

# Continue decreasing dimension until performance degrades
# ... (Repeat the previous steps with decreasing latent_dim[2])

# Train final model on entire training data and test
autoencoder.compile(optimizer='adam', loss=loss_fn)
autoencoder.fit(x_train, x_train, epochs=20)
test_loss = autoencoder.evaluate(x_test, x_test)
print("Test loss:", test_loss)

# Save model weights
autoencoder.save_weights("autoencoder_weights.h5")

# Reconstruct and display test samples
decoded_imgs = autoencoder.predict(x_test[:5])
plt.figure(figsize=(10, 5))
for i in range(5):
    decoded_img = decoded_imgs[i].reshape(28, 28) * 255
    original_img = x_test[i].reshape(28, 28) * 255
    plt.subplot(1, 10, 2*i+1)
    plt.imshow(original_img, cmap='gray')
    plt.subplot(1, 10, 2*i+2)
    plt.imshow(decoded_img, cmap='gray')
plt.show()