import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, Add, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
import os

# process the data
data = pd.read_csv('./datasets/train.csv')

# Load the test data (without labels)
test_data = pd.read_csv('./datasets/test.csv')

labels = data.iloc[:, 0]
pixels = data.iloc[:, 1:]

# Hyperparamaters
learning_rate = 0.0005
epoch_num = 20
batch_size_num = 64

# normalize the pixels
pixels = pixels / 255.0

# Convert to numpy arrays for easier manipulation and re shape them
X = pixels.values
y = labels.values
X = X.reshape(-1, 28, 28, 1)  # -1 means "infer the size based on other dimensions"
y = to_categorical(y, num_classes=10)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Residual block function
def residual_block(x, filters, kernel_size=3, stride=1):
    # Main path
    shortcut = x

    # First convolution layer
    x = Conv2D(filters, kernel_size, strides=stride, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Second convolution layer
    x = Conv2D(filters, kernel_size, padding='same')(x)
    x = BatchNormalization()(x)

    # Add shortcut to the output
    x = Add()([x, shortcut])
    x = Activation('relu')(x)

    return x

# Define the ResNet-like model
def create_resnet_model(input_shape):
    inputs = Input(shape=input_shape)

    # Initial convolutional layer
    x = Conv2D(32, (3, 3), padding='same', strides=(1, 1))(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Add residual blocks
    x = residual_block(x, 32)
    x = residual_block(x, 32)

    # Flatten the output and add a dense layer for classification
    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)
    outputs = Dense(10, activation='softmax')(x)

    # Create the model
    model = Model(inputs, outputs)

    return model

# Create the model
input_shape = (28, 28, 1)
model = create_resnet_model(input_shape)

# Set learning rate for adam optimizer
optimizer = Adam(learning_rate=learning_rate)

model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Add ReduceLROnPlateau to adjust the learning rate when the validation loss plateaus
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)

# Train the model
model.fit(X_train, y_train, epochs=epoch_num, batch_size=batch_size_num, validation_data=(X_val, y_val), callbacks=[reduce_lr])

val_loss, val_accuracy = model.evaluate(X_val, y_val)
print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")

############# TEST DATA ###############

# Normalize the pixels
test_pixels = test_data / 255.0

# Convert to numpy array and reshape them
X_test = test_pixels.values
X_test = X_test.reshape(-1, 28, 28, 1)

# Make predictions
predictions = model.predict(X_test)
predicted_labels = np.argmax(predictions, axis=1)

# Create a DataFrame for the output
output = pd.DataFrame({
    "ImageId": np.arange(1, len(predicted_labels) + 1),
    "Label": predicted_labels
})

# Save the predictions to a CSV file
output.to_csv('./datasets/submission3.csv', index=False)

print("Predictions saved to submission3.csv")