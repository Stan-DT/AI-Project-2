import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers, models
import pandas as pd
import pyaml
from pyaml import yaml


class Classification:
    def __init__(self,x_train,y_train,x_test,y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test


        self.class_labels = None
        self.class_counts = None
        self.seed = None
        self.x_val = None
        self.y_val = None

        self.model = None
        self.epochs = None
        self.batch_size = None
        self.num_classes = None
        self.learning_rate = None

        self.hyperparameters = None
        self.loss = None
        self.history = None

        self.batch_sizes_list = None
        self.epochs_list = None
        self.optimizers_list = None
        self.loss_list = None

    def get_sizes(self):
        print("Shape of training set: x: " + str(self.x_train.shape))
        print("Shape of training set: y: " + str(self.y_train.shape))
        print("Training set size:", self.x_train.shape[0])
        print("Test set size:", self.x_test.shape[0])
    
    def plot_class_distribution(self):
        self.class_labels, self.class_counts = np.unique(np.append(self.y_train,self.y_test), return_counts=True)
        plt.bar(self.class_labels, self.class_counts)
        plt.xlabel('Class Label')
        plt.ylabel('Number of Samples')
        plt.title('Class Distribution')
        plt.show()

    def plot_samples(self,samples_per_class):
        num_classes = len(self.class_labels)

        plt.figure(figsize=(10, 10))
        for i in range(num_classes):
            idx = np.where(self.y_train == i)[0][:samples_per_class]
            for j, sample_idx in enumerate(idx):
                plt.subplot(num_classes, samples_per_class, i * samples_per_class + j + 1)
                plt.imshow(self.x_train[sample_idx], cmap='gray')
                plt.title(f'Class {i}')
                plt.axis('off')
        plt.tight_layout()
        plt.show()

    def set_seed(self,number):
        self.seed = np.random.seed(number)

    def split_to_val(self,test_size):

        if test_size != 0:
            self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(self.x_train, 
                                                                                          self.y_train, 
                                                                                          random_state=42,
                                                                                          test_size=test_size)
        else:
            self.x_val = None
            self.y_val = None
            # self.x_train_split = None
            # self.y_train_split = None
        
    def normalize_data(self):

        self.x_train = self.x_train / 255.0
        if self.x_val is not None:
            self.x_val = self.x_val / 255.0
        self.x_test = self.x_test / 255.0

    def convert_to_binary_class_matrices(self,num_classes):

        self.y_train = to_categorical(self.y_train, num_classes)
        if self.y_val is not None:
            self.y_val = to_categorical(self.y_val, num_classes)
        self.y_test = to_categorical(self.y_test, num_classes)

    def define_model(self,model):
        self.model = model

    def add_to_model(self,arg):
        self.model.add(arg)

    def print_summary(self):
        self.model.summary()

    def set_hyperparameters(self,epochs = 10,batch_size = 64):
        self.epochs = epochs
        self.batch_size = batch_size
        self.hyperparameters = {'epochs':epochs,'batch_size':batch_size}

    def compile_model(self,optimizer,loss,metrics):
        self.model.compile(optimizer=str(optimizer),
              loss=loss,
              metrics=[str(metrics)])
        
        self.loss = loss
        
    def print_hyperparameters(self):
        print(self.hyperparameters)

    def print_loss(self):
        print(self.loss)
    
    def train_model(self):

        self.history = self.model.fit(self.x_train, self.y_train,
                    batch_size=self.batch_size,
                    epochs=self.epochs,
                    verbose=1,
                    validation_data=(self.x_val, self.y_val)
                    )
        
    def plot_training_and_validation_curves(self):
        plt.plot(self.history.history['accuracy'], label='Training Accuracy')
        plt.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()

        plt.plot(self.history.history['loss'], label='Training Loss')
        plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

    def init_hyperparameter_tuning(self,batch_sizes_list,epochs_list,optimizers_list,loss_list):
        self.batch_sizes_list = batch_sizes_list
        self.epochs_list = epochs_list
        self.optimizers_list = optimizers_list
        self.loss_list = loss_list

    def tune_hyperparameters(self,file_name):
        results = []

        for batch in self.batch_sizes_list:
            for epoch in self.epochs_list:
                for optimizer in self.optimizers_list:
                    for loss_function in self.loss_list:
                        model = models.Sequential()

                        model.add(layers.Input(shape=(28, 28, 1)))
                        model.add(layers.Conv2D(32, (3, 3), activation='relu'))
                        model.add(layers.MaxPooling2D((2, 2)))
                        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
                        model.add(layers.MaxPooling2D((2, 2)))
                        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
                        model.add(layers.Flatten())
                        model.add(layers.Dense(64, activation='relu'))
                        model.add(layers.Dense(10, activation='softmax'))

                        model.compile(optimizer=optimizer,
                                    loss=loss_function,
                                    metrics=['accuracy'])
                        

                        model.fit(self.x_train, self.y_train,
                                batch_size=batch,
                                epochs=epoch,
                                verbose=1,
                                validation_data=(self.x_val, self.y_val))

                        loss, accuracy = model.evaluate(self.x_val, self.y_val, verbose=1)
                        
                        # Store results
                        results.append({
                            "batch_size": batch,
                            "epochs": epoch,
                            "optimizer": optimizer,
                            "validation_loss": loss,
                            "validation_accuracy": accuracy
                        })
            

        results_df = pd.DataFrame(results)
        results_df.to_csv(str(file_name), index=False)

        print(results_df)

    def evaluate_model(self):
        return self.model.evaluate(self.x_test, self.y_test, verbose=0)

    def save_model(self,file_name):
        self.model.save_weights(str(file_name))

    def dump_to_yaml(self,file_name):
        file_name = str(file_name)
        with open('final_model_config.yaml', 'w') as file:
            pyaml.dump(self.model.get_config(), file)

    def load_config(self, model, h5_file, file_name):
        with open(str(file_name), 'r') as yaml_file:
            loaded_model_config = yaml.safe_load(yaml_file)

        self.model = model.from_config(loaded_model_config)

        self.model.load_weights(str(h5_file))

    def plot_random_samples(self, num_samples_to_plot=10):

        indices = np.random.choice(len(self.x_test), num_samples_to_plot, replace=False)
        x_samples = self.x_test[indices]
        y_true = np.argmax(self.y_test[indices], axis=1)
        
        y_pred = np.argmax(self.model.predict(x_samples), axis=1)
        
        plt.figure(figsize=(15, 8))
        for i in range(num_samples_to_plot):
            plt.subplot(2, 5, i + 1)
            plt.imshow(x_samples[i].reshape(28, 28), cmap='gray')
            plt.title(f"True: {y_true[i]}, Predicted: {y_pred[i]}")
            plt.axis('off')
        plt.tight_layout()
        plt.show()

    def predict_test(self):
        y_pred = np.argmax(self.model.predict(self.x_test), axis=1)
        return y_pred





