# Import necessary libraries
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Reshape
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.layers import Conv2D

from sklearn.model_selection import train_test_split
import pandas as pd
import pyaml
import yaml
from tensorflow.keras.layers import Activation, Dense, Input
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Reshape, Conv2DTranspose
from sklearn.neighbors import NearestNeighbors






class Encoding:

    def __init__(self,x_train, x_test):

        self.x_train = x_train
        self.x_val = None
        self.x_test = x_test

        self.y_train = None
        self.y_test = None
        

        self.x_train_original = x_train
        self.x_val_original = None
        self.x_test_original = x_test
        

        self.loss_fn = None
        self.seed_np = None
        self.seed_tf = None
        self.latent_dim = None
        self.shape = None

        self.inputs = None
        self.encoder = None
        self.decoder = None
        self.autoencoder = None
        self.history = None
        self.encoded = None

        self.batch_sizes_list = None
        self.epochs_list = None
        self.optimizers_list = None

    def set_seed(self,seed):
        self.seed_np = np.random.seed(seed)
        self.seed_tf = tf.random.set_seed(seed)
    
    def set_latent_dim(self,latent_dim):
        self.latent_dim = latent_dim
    
    def set_loss_function(self,loss_fn):
        self.loss_fn = loss_fn
                    
    def normalize_data(self):
        self.x_train= self.x_train/ 255.0
        self.x_test = self.x_test / 255.0

    def split_to_val(self,test_size):
        self.x_train, self.x_val = train_test_split(self.x_train, 
                                                    random_state=42,
                                                    test_size=test_size)
        self.x_val_original = self.x_val

    def reshaper(self):
        self.x_train = self.x_train.reshape((self.x_train.shape[0], self.x_train.shape[1], self.x_train.shape[2], 1))
        if self.x_val is not None:
            self.x_val = self.x_val.reshape((self.x_val.shape[0], self.x_val.shape[1], self.x_val.shape[2], 1))
        self.x_test = self.x_test.reshape((self.x_test.shape[0], self.x_test.shape[1], self.x_test.shape[2], 1))


    def define_encoder(self,input_shape,filters,kernel_size):
        self.inputs = Input(shape=input_shape, name='encoder_input')
        x = self.inputs

        x = Conv2D(filters=filters,
                    kernel_size=kernel_size,
                    strides=2,
                    activation='relu',
                    padding='same')(x)
        
        self.shape = K.int_shape(x)
        x = Flatten()(x)
        latent_outputs = Dense(self.latent_dim, name='latent_vector')(x)

        self.encoder = Model(inputs=self.inputs, outputs=latent_outputs, name='encoder')
        self.encoder.summary()

    def define_decoder(self,filters,kernel_size):

        latent_inputs = Input(shape=(self.latent_dim,), name='decoder_input')

        # afterwards we have one dense layer that allows us to 
        # transform the input into (None, 7, 7, 64)
        x = Dense(self.shape[1] * self.shape[2] * self.shape[3])(latent_inputs)
        x = Reshape((self.shape[1], self.shape[2], self.shape[3]))(x)

        # then we add transposed convolutional layers but in a reversed order compared to the encoder

        x = Conv2DTranspose(filters=filters,
                        kernel_size=kernel_size,
                        strides=2,
                        activation='relu',
                        padding='same')(x)

        # then we add one more convolutional layer to control the channel dimension   
        x = Conv2DTranspose(filters=1,
                            kernel_size=kernel_size,
                            padding='same')(x)

        # and one activation later with the sigmoid activation function
        outputs = Activation('sigmoid', name='decoder_output')(x)

        self.decoder = Model(inputs=latent_inputs, outputs=outputs, name='decoder')
        self.decoder.summary(line_length=110)

    def define_autoencoder(self):
        self.autoencoder = Model(self.inputs, self.decoder(self.encoder(self.inputs)), name='autoencoder')
        self.autoencoder.summary()

    def compile_model(self,loss,optimizer):
        self.encoder.compile(loss=loss, optimizer=optimizer)

    def train_model(self, epochs, batch_size):
        self.history = self.autoencoder.fit(self.x_train,
                self.x_train,
                validation_data=(self.x_val, self.x_val),
                epochs=epochs,
                batch_size=batch_size)
        
    def train_model_no_val(self, epochs,batch_size ):
        self.history = self.autoencoder.fit(self.x_train,
                self.x_train,
                epochs=epochs,
                batch_size=batch_size)
        
    def evaluate_model(self):
        return self.autoencoder.evaluate(self.x_test, self.x_test, verbose=0)
    
    def save_model(self,file_name):
        self.autoencoder.save_weights(file_name)

    def save_encoder_and_dump_to_yaml(self,h5_file,yaml_file):
        self.encoder.save_weights(h5_file)
        with open(yaml_file, 'w') as file:
            pyaml.dump(self.encoder.get_config(), file)

        
    def plot_training_and_validation_curves(self):
        plt.plot(self.history.history['loss'], label='Training Loss')
        print('loss', self.history.history['loss'][-1])
        plt.plot(self.history.history['val_loss'], label='Validation Loss')
        print('val_loss',self.history.history['val_loss'][-1])
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

    def init_hyperparameter_tuning(self,batch_sizes_list,epochs_list,optimizers_list):
        self.batch_sizes_list = batch_sizes_list
        self.epochs_list = epochs_list
        self.optimizers_list = optimizers_list


    def tune_hyperparameters(self,file_name):
        results = []

        for batch in self.batch_sizes_list:
            for epoch in self.epochs_list:
                for optimizer in self.optimizers_list:
                        
                        self.compile_model('mse',optimizer)
                        self.train_model(epoch, batch)

                        training_loss = self.history.history['loss']
                        validation_loss = self.history.history['val_loss']
                        
                        # Store results
                        results.append({
                            "batch_size": batch,
                            "epochs": epoch,
                            "optimizer": optimizer,
                            "training_loss": training_loss[-1],
                            "validation_loss": validation_loss[-1]
                        })
            

        results_df = pd.DataFrame(results)
        results_df.to_csv(str(file_name), index=False)

        print(results_df)

    def show_val_plots(self):

        decoded_imgs = self.autoencoder.predict(self.x_val_original)
        plt.figure(figsize=(10, 5))
        for i in range(5):
            # Reshape and normalize reconstructed image
            decoded_img = decoded_imgs[i].reshape(self.x_val.shape[1], self.x_val.shape[2])
            # Original image
            original_img = self.x_val_original[i].reshape(self.x_val.shape[1], self.x_val.shape[2])
            plt.subplot(1, 10, 2*i+1)
            plt.imshow(original_img, cmap='gray')
            plt.subplot(1, 10, 2*i+2)
            plt.imshow(decoded_img, cmap='gray')
        plt.show()

    def show_test_plots(self):
    
        decoded_imgs = self.autoencoder.predict(self.x_test_original)
        plt.figure(figsize=(10, 5))
        for i in range(5):
            # Reshape and normalize reconstructed image
            decoded_img = decoded_imgs[i].reshape(self.x_test_original.shape[1], self.x_test_original.shape[2])
            # Original image
            original_img = self.x_test_original[i].reshape(self.x_test_original.shape[1], self.x_test_original.shape[2])
            plt.subplot(1, 10, 2*i+1)
            plt.imshow(original_img, cmap='gray')
            plt.subplot(1, 10, 2*i+2)
            plt.imshow(decoded_img, cmap='gray')
        plt.show()

    def dump_to_yaml(self,file_name):
        with open(file_name, 'w') as file:
            pyaml.dump(self.autoencoder.get_config(), file)

    def load_config(self, h5_file,file_name):
        with open(str(file_name), 'r') as yaml_file:
            loaded_model_config = yaml.safe_load(yaml_file)

        self.autoencoder = Model.from_config(loaded_model_config)

        self.autoencoder.load_weights(str(h5_file))

    def load_config_encoder(self, h5_file,file_name):
        with open(str(file_name), 'r') as yaml_file:
            loaded_model_config = yaml.safe_load(yaml_file)

        self.encoder = Model.from_config(loaded_model_config)

        self.encoder.load_weights(str(h5_file))

    def encode_predict(self):
        self.encoded_x_train = self.encoder.predict(self.x_train)

    def closest_samples(self,number, y_train, y_test):

        self.y_train = y_train
        self.y_test = y_test

        random_index = np.random.randint(0, len(self.x_test))
        random_sample = np.expand_dims(self.x_train[random_index], axis=0)
        random_label = self.y_test[random_index]

        encoded_random_sample = self.encoder.predict(random_sample)

        encoded_x_train_reshaped = self.encoded_x_train.reshape(len(self.encoded_x_train), -1)
        encoded_random_sample_reshaped = random_sample.reshape(len(encoded_random_sample), -1)

        nn = NearestNeighbors(n_neighbors=number)
        nn.fit(encoded_x_train_reshaped)
        distances, indices = nn.kneighbors(encoded_random_sample_reshaped)

        plt.figure(figsize=(10, 6))
        plt.subplot(2, 6, 1)
        plt.imshow(random_sample.reshape(28, 28), cmap='gray')
        plt.title('Random Sample')
        plt.text(0, 39, f'Label: {random_label}', color='red', fontsize=10, ha='left')

        for i, index in enumerate(indices[0]):
            plt.subplot(2, 6, i + 2)
            plt.imshow(self.x_train[index].reshape(28, 28), cmap='gray')
            plt.text(0, 39, f'Label: {self.y_train[index]}', color='red', fontsize=10, ha='left',va = 'bottom')

        plt.tight_layout()
        plt.show()


    
    

    

# # Define reconstruction metric and set random seed

# np.random.seed(36)

# # Load and preprocess data
# (x_train, _), (x_val, _) = fashion_mnist.load_data()
# x_train = x_train.astype('float32') / 255.
# x_val = x_val.astype('float32') / 255.
# x_train = x_train.reshape((x_train.shape[0], 784))
# x_val = x_val.reshape((x_val.shape[0], 784))

# Define initial model architecture
# latent_dim = (4, 4, 32)
# inputs = tf.keras.Input(shape=(784,))
# encoded = Flatten()(inputs)
# encoded = Dense(units=latent_dim[0] * latent_dim[1] * latent_dim[2], activation='relu')(encoded)
# decoded = Dense(units=784, activation='sigmoid')(encoded)
# autoencoder = Model(inputs, decoded)

# # Print model architecture
# autoencoder.summary()

# # Train initial model and visualize learning curves
# history = autoencoder.compile(optimizer='adam', loss=loss_fn).fit(x_train, x_train,
#                                                                 epochs=20, validation_data=(x_val, x_val))
# plt.plot(history.history['loss'], label='Train')
# plt.plot(history.history['val_loss'], label='Validation')
# plt.legend()
# plt.show()

# # Fine-tune model and evaluate
# autoencoder.compile(optimizer='adam', loss=loss_fn)
# autoencoder.fit(x_train, x_train, epochs=10, validation_data=(x_val, x_val))
# val_loss = autoencoder.evaluate(x_val, x_val)
# print("Validation loss:", val_loss)

# # Reconstruct and display validation samples
# decoded_imgs = autoencoder.predict(x_val)
# plt.figure(figsize=(10, 5))
# for i in range(5):
#     # Reshape and normalize reconstructed image
#     decoded_img = decoded_imgs[i].reshape(28, 28) * 255
#     # Original image
#     original_img = x_val[i].reshape(28, 28) * 255
#     plt.subplot(1, 10, 2*i+1)
#     plt.imshow(original_img, cmap='gray')
#     plt.subplot(1, 10, 2*i+2)
#     plt.imshow(decoded_img, cmap='gray')
# plt.show()

# # Experiment with decreasing feature dimension
# latent_dim = (4, 4, 28)
# encoded = Dense(units=latent_dim[0] * latent_dim[1] * latent_dim[2], activation='relu')(encoded)
# autoencoder = Model(inputs, decoded)
# autoencoder.compile(optimizer='adam', loss=loss_fn)

# # Train and evaluate the model
# history = autoencoder.fit(x_train, x_train, epochs=20, validation_data=(x_val, x_val))
# val_loss = autoencoder.evaluate(x_val, x_val)
# print("Validation loss:", val_loss)

# # Reconstruct and display validation samples
# decoded_imgs = autoencoder.predict(x_val)
# plt.figure(figsize=(10, 5))
# for i in range(5):
#     decoded_img = decoded_imgs[i].reshape(28, 28) * 255
#     original_img = x_val[i].reshape(28, 28) * 255
#     plt.subplot(1, 10, 2*i+1)
#     plt.imshow(original_img, cmap='gray')
#     plt.subplot(1, 10, 2*i+2)
#     plt.imshow(decoded_img, cmap='gray')
# plt.show()

# # Continue decreasing dimension until performance degrades
# # ... (Repeat the previous steps with decreasing latent_dim[2])

# # Train final model on entire training data and test
# autoencoder.compile(optimizer='adam', loss=loss_fn)
# autoencoder.fit(x_train, x_train, epochs=20)
# test_loss = autoencoder.evaluate(x_test, x_test)
# print("Test loss:", test_loss)

# # Save model weights
# autoencoder.save_weights("autoencoder_weights.h5")

# # Reconstruct and display test samples
# decoded_imgs = autoencoder.predict(x_test[:5])
# plt.figure(figsize=(10, 5))
# for i in range(5):
#     decoded_img = decoded_imgs[i].reshape(28, 28) * 255
#     original_img = x_test[i].reshape(28, 28) * 255
#     plt.subplot(1, 10, 2*i+1)
#     plt.imshow(original_img, cmap='gray')
#     plt.subplot(1, 10, 2*i+2)
#     plt.imshow(decoded_img, cmap='gray')
# plt.show()