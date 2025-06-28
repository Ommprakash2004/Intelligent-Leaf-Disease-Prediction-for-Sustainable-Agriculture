import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Input
from tensorflow.keras.applications import VGG16, ResNet50, MobileNetV2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class LeafDiseaseModel:
    """
    Class for building and training models for leaf disease classification.
    """
    
    def __init__(self, model_type='cnn', input_shape=(224, 224, 3), num_classes=10):
        """
        Initialize the model builder.
        
        Args:
            model_type (str): Type of model to build ('cnn', 'transfer_learning')
            input_shape (tuple): Input shape of the images
            num_classes (int): Number of disease classes to predict
        """
        self.model_type = model_type
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
    
    def build_custom_cnn(self):
        """
        Build a custom CNN model for leaf disease classification.
        
        Returns:
            tensorflow.keras.models.Model: Built model
        """
        model = Sequential()
        
        # First convolutional block
        model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=self.input_shape))
        model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        
        # Second convolutional block
        model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        
        # Third convolutional block
        model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        
        # Fully connected layers
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.num_classes, activation='softmax'))
        
        return model
    
    def build_transfer_learning_model(self, base_model_name='vgg16'):
        """
        Build a transfer learning model using a pre-trained base model.
        
        Args:
            base_model_name (str): Name of the pre-trained model to use
                                  ('vgg16', 'resnet50', 'mobilenet')
        
        Returns:
            tensorflow.keras.models.Model: Built model
        """
        # Load the pre-trained model
        if base_model_name == 'vgg16':
            base_model = VGG16(weights='imagenet', include_top=False, input_shape=self.input_shape)
        elif base_model_name == 'resnet50':
            base_model = ResNet50(weights='imagenet', include_top=False, input_shape=self.input_shape)
        elif base_model_name == 'mobilenet':
            base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=self.input_shape)
        else:
            raise ValueError(f"Unsupported base model: {base_model_name}")
        
        # Freeze the base model layers
        for layer in base_model.layers:
            layer.trainable = False
        
        # Add custom layers on top
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)
        predictions = Dense(self.num_classes, activation='softmax')(x)
        
        # Create the final model
        model = Model(inputs=base_model.input, outputs=predictions)
        
        return model
    
    def build_model(self):
        """
        Build the model based on the specified model type.
        
        Returns:
            tensorflow.keras.models.Model: Built model
        """
        if self.model_type == 'cnn':
            self.model = self.build_custom_cnn()
        elif self.model_type == 'transfer_learning':
            self.model = self.build_transfer_learning_model()
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        return self.model
    
    def compile_model(self, learning_rate=0.001):
        """
        Compile the model with appropriate loss function and optimizer.
        
        Args:
            learning_rate (float): Learning rate for the optimizer
        """
        if self.model is None:
            self.build_model()
        
        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
    
    def train(self, train_data, validation_data, epochs=50, batch_size=32, callbacks=None):
        """
        Train the model on the provided data.
        
        Args:
            train_data (tuple): Training data (images, labels)
            validation_data (tuple): Validation data (images, labels)
            epochs (int): Number of epochs to train
            batch_size (int): Batch size for training
            callbacks (list): List of Keras callbacks for training
            
        Returns:
            tensorflow.keras.callbacks.History: Training history
        """
        if self.model is None:
            self.build_model()
            self.compile_model()
        
        # Default callbacks if none provided
        if callbacks is None:
            callbacks = [
                ModelCheckpoint(
                    'models/best_model.h5',
                    monitor='val_accuracy',
                    save_best_only=True,
                    mode='max',
                    verbose=1
                ),
                EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True,
                    verbose=1
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.2,
                    patience=5,
                    min_lr=1e-6,
                    verbose=1
                )
            ]
        
        # Unpack the data
        X_train, y_train = train_data
        X_val, y_val = validation_data
        
        # Data augmentation
        datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2,
            fill_mode='nearest'
        )
        datagen.fit(X_train)
        
        # Train the model
        history = self.model.fit(
            datagen.flow(X_train, y_train, batch_size=batch_size),
            steps_per_epoch=len(X_train) // batch_size,
            epochs=epochs,
            validation_data=(X_val, y_val),
            callbacks=callbacks
        )
        
        return history
    
    def evaluate(self, test_data):
        """
        Evaluate the model on test data.
        
        Args:
            test_data (tuple): Test data (images, labels)
            
        Returns:
            tuple: (loss, accuracy)
        """
        if self.model is None:
            raise ValueError("Model has not been built or trained yet.")
        
        X_test, y_test = test_data
        return self.model.evaluate(X_test, y_test)
    
    def predict(self, images):
        """
        Make predictions on new images.
        
        Args:
            images (numpy.ndarray): Input images
            
        Returns:
            numpy.ndarray: Predicted class probabilities
        """
        if self.model is None:
            raise ValueError("Model has not been built or trained yet.")
        
        return self.model.predict(images)
    
    def save_model(self, filepath):
        """
        Save the model to disk.
        
        Args:
            filepath (str): Path to save the model
        """
        if self.model is None:
            raise ValueError("Model has not been built or trained yet.")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """
        Load a model from disk.
        
        Args:
            filepath (str): Path to the saved model
        """
        self.model = tf.keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")
        
        return self.model