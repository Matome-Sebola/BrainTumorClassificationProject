import os
import re
import numpy as np
import cv2
import cv2 as cv
import time
import shutil
import zipfile
import zipfile as zf
import urllib.request
import imutils
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
from os import listdir
from os.path import isfile, join
from random import randrange
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.layers import Conv2D, Activation, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array


#Data Splitting
# Paths to the dataset
dataset_dir = 'Sebola' 
output_dir = 'Mudau' # New path to store splitted data   

# Create output directories
train_dir = os.path.join(output_dir, 'train')
val_dir = os.path.join(output_dir, 'val')
test_dir = os.path.join(output_dir, 'test')

# Ensure directories exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Define the split ratios
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1

# Iterate through each category
categories = os.listdir(dataset_dir)
for category in categories:
    category_path = os.path.join(dataset_dir, category)
    if not os.path.isdir(category_path):
        continue

    # Get all images in the category
    images = os.listdir(category_path)
    images = [os.path.join(category_path, img) for img in images]

    # Split the images into train, validation, and test sets
    train_images, temp_images = train_test_split(images, train_size=train_ratio, random_state=42)
    val_images, test_images = train_test_split(temp_images, test_size=test_ratio/(test_ratio + val_ratio), random_state=42)

    # Define function to copy images to respective directories
    def copy_images(image_list, target_dir):
        category_target_dir = os.path.join(target_dir, category)
        os.makedirs(category_target_dir, exist_ok=True)
        for img_path in image_list:
            shutil.copy(img_path, category_target_dir)

    # Copy images to the respective directories
    copy_images(train_images, train_dir)
    copy_images(val_images, val_dir)
    copy_images(test_images, test_dir)
    
    #Successfull spliting message
print(f"Dataset split completed and it is split into {int(train_ratio * 100)}% training set, {int(test_ratio * 100)}% test set and {int(val_ratio * 100)}% validation set")


Data Preprocessing
# Data augmentation setup
demo_datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.05,
    height_shift_range=0.05,
    rescale=1./255,
    shear_range=0.05,
    brightness_range=[0.1, 1.5],
    horizontal_flip=True,
    vertical_flip=True
)

def cropAndAugmentation():
    # Augmentation code
    flag1 = 0
    flag2 = 0
    j = 0
    IMG_SIZE = 224
    dim = (IMG_SIZE, IMG_SIZE)
    
    # Define directories
    main_dirs = ['Mudau/train', 'Mudau/test', 'Mudau/val']
    subdirs = ['glioma_tumor', 'meningioma_tumor','no_tumor', 'pituitary_tumor']
    
    for main_dir in main_dirs:
        for subdir in subdirs:
            input_path = os.path.join(main_dir, subdir)
            if not os.path.exists(input_path):
                print(f"Directory {input_path} does not exist.")
                continue
            output_folder = os.path.join(main_dir + '_output', subdir)
            os.makedirs(output_folder, exist_ok=True)
            
            for img_name in os.listdir(input_path):
                img_path = os.path.join(input_path, img_name)
                image = cv2.imread(img_path)

                if image is None:
                    print(f"Image {img_path} could not be read.")
                    continue
                
                image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                gray = cv2.GaussianBlur(gray, (5, 5), 0)
                
                if flag1 == 0:
                    plt.figure(figsize=(15, 6))
                    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                    plt.xticks([])
                    plt.yticks([])
                    plt.title('Step1: Before Crop')
                    plt.show()
                    flag1 = 1

                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                gray = cv2.GaussianBlur(gray, (5, 5), 0)

                thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
                thresh = cv2.erode(thresh, None, iterations=2)
                thresh = cv2.dilate(thresh, None, iterations=2)

                cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cnts = imutils.grab_contours(cnts)
                if not cnts:
                    print(f"No contours found in {img_path}.")
                    continue
                c = max(cnts, key=cv2.contourArea)

                extLeft = tuple(c[c[:, :, 0].argmin()][0])
                extRight = tuple(c[c[:, :, 0].argmax()][0])
                extTop = tuple(c[c[:, :, 1].argmin()][0])
                extBot = tuple(c[c[:, :, 1].argmax()][0])

                img_cnt = cv2.drawContours(image.copy(), [c], -1, (0, 255, 255), 4)
                img_pnt = cv2.circle(img_cnt.copy(), extLeft, 5, (0, 0, 255), -1)
                img_pnt = cv2.circle(img_pnt, extRight, 5, (0, 255, 0), -1)
                img_pnt = cv2.circle(img_pnt, extTop, 5, (255, 0, 0), -1)
                img_pnt = cv2.circle(img_pnt, extBot, 5, (255, 255, 0), -1)

                ADD_PIXELS = 0
                new_image = image[extTop[1]-ADD_PIXELS:extBot[1]+ADD_PIXELS, extLeft[0]-ADD_PIXELS:extRight[0]+ADD_PIXELS].copy()

                if flag2 == 0:
                    plt.figure(figsize=(15, 6))
                    plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
                    plt.xticks([])
                    plt.yticks([])
                    plt.title('Step2: After Crop')
                    plt.show()
                    flag2 = 1

                plt.figure(figsize=(15, 6))
                plt.subplot(141)
                plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                plt.xticks([])
                plt.yticks([])
                plt.title('Step 1. Get the original image')
                plt.subplot(142)
                plt.imshow(cv2.cvtColor(img_cnt, cv2.COLOR_BGR2RGB))
                plt.xticks([])
                plt.yticks([])
                plt.title('Step 2. Find the biggest contour')
                plt.subplot(143)
                plt.imshow(cv2.cvtColor(img_pnt, cv2.COLOR_BGR2RGB))
                plt.xticks([])
                plt.yticks([])
                plt.title('Step 3. Find the extreme points')
                plt.subplot(144)
                plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
                plt.xticks([])
                plt.yticks([])
                plt.title('Step 4. Crop the image')
                plt.show()
                
                x = new_image
                x = x.reshape((1,) + x.shape)
                i = 0

                for batch in demo_datagen.flow(x, batch_size=1, save_to_dir=output_folder, save_prefix='{}_{}'.format(subdir, j), save_format='jpg'):
                    i += 1
                    if i > 20:
                        break 
                    j += 1  

                plt.figure(figsize=(15, 6))
                plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
                plt.xticks([])
                plt.yticks([])
                plt.title('Original Image')
                plt.show()

                plt.figure(figsize=(15, 6))
                i = 1
                for img_name in os.listdir(output_folder):
                    img_path = os.path.join(output_folder, img_name)
                    img = cv2.imread(img_path)
                    if img is None:
                        continue
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    plt.subplot(3, 7, i)
                    plt.imshow(img)
                    plt.xticks([]) 
                    plt.yticks([])
                    i += 1
                    if i > 3 * 7:
                        break
                plt.suptitle('Augmented Images')
                plt.show()

cropAndAugmentation()


CNN model building to brain tumor classification
# Model parameters
num_conv_layers = 3
num_dense_layers = 1
layer_size = 32
num_training_epochs = 15
MODEL_NAME = "brain-tumor"
batch_size = 32

# Directory paths
base_dir = 'Mudau'
train_dir = os.path.join(base_dir, 'train_output')
val_dir = os.path.join(base_dir, 'val_output')
test_dir = os.path.join(base_dir, 'test_output')

# Data generators
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Data generators for training, validation, and test sets
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(256, 256),
    batch_size=batch_size,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(256, 256),
    batch_size=batch_size,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(256, 256),
    batch_size=batch_size,
    class_mode='categorical'
)

# Initiate model variable
model = Sequential()

# Adding properties to model variable
# Add a convolution layer
model.add(Conv2D(layer_size, (3, 3), input_shape=(256, 256, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Add additional layers based on num_conv_layers
for _ in range(num_conv_layers - 1):
    model.add(Conv2D(layer_size, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

# Reduce dimensionality by flattening
model.add(Flatten())

# Add fully connected "dense" layers
for _ in range(num_dense_layers):
    model.add(Dense(layer_size))
    model.add(Activation('relu'))

# Output layer
model.add(Dense(4))
model.add(Activation('softmax'))

# Compile the sequential model with all added properties
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# Use the data already loaded previously to train the model
history = model.fit(
    train_generator,
    epochs=num_training_epochs,
    validation_data=val_generator
)

# Save the trained model
model.save(f'{MODEL_NAME}.h5')


#Plot the training and validation set to detect overfitting
# Plot the training and validation loss
def plot_loss(history):
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()

plot_loss(history)


# Model parameters
num_conv_layers = 3
num_dense_layers = 1
layer_size = 16
num_training_epochs = 15
MODEL_NAME = "brain-tumor2"
batch_size = 32

# Directory paths
base_dir = 'Mudau'
train_dir = os.path.join(base_dir, 'train_output')
val_dir = os.path.join(base_dir, 'val_output')
test_dir = os.path.join(base_dir, 'test_output')

# Data generators
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Data generators for training, validation, and test sets
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(256, 256),
    batch_size=batch_size,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(256, 256),
    batch_size=batch_size,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(256, 256),
    batch_size=batch_size,
    class_mode='categorical'
)

# Initiate model variable
model = Sequential()

# Adding properties to model variable
# Add a convolution layer
model.add(Conv2D(layer_size, (3, 3), input_shape=(256, 256, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Add additional layers based on num_conv_layers
for _ in range(num_conv_layers - 1):
    model.add(Conv2D(layer_size, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

# Reduce dimensionality
model.add(Flatten())

# Add fully connected "dense" layers 
for _ in range(num_dense_layers):
    model.add(Dense(layer_size))
    model.add(Activation('relu'))

# Output layer
model.add(Dense(4))
model.add(Activation('softmax'))

# Compile the sequential model with all added properties
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# Use the data already loaded previously to train the model
history = model.fit(
    train_generator,
    epochs=num_training_epochs,
    validation_data=val_generator
)

# Save the trained model
model.save(f'{MODEL_NAME}.h5')

# Plot the training and validation loss
def plot_loss(history):
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()

plot_loss(history)

# Model parameters
num_conv_layers = 3
num_dense_layers = 1
layer_size = 8
num_training_epochs = 15
MODEL_NAME = "brain-tumor3"
batch_size = 32

# Directory paths
base_dir = 'Mudau'
train_dir = os.path.join(base_dir, 'train_output')
val_dir = os.path.join(base_dir, 'val_output')
test_dir = os.path.join(base_dir, 'test_output')

# Data generators
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Data generators for training, validation, and test sets
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(256, 256),
    batch_size=batch_size,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(256, 256),
    batch_size=batch_size,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(256, 256),
    batch_size=batch_size,
    class_mode='categorical'
)

# Initiate model variable
model = Sequential()

# Adding properties to model variable
# Add a convolution layer
model.add(Conv2D(layer_size, (3, 3), input_shape=(256, 256, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Add additional layers based on num_conv_layers
for _ in range(num_conv_layers - 1):
    model.add(Conv2D(layer_size, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

# Reduce dimensionality
model.add(Flatten())

# Add fully connected "dense" layers if specified
for _ in range(num_dense_layers):
    model.add(Dense(layer_size))
    model.add(Activation('relu'))

# Output layer
model.add(Dense(4))
model.add(Activation('softmax'))

# Compile the sequential model with all added properties
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# Use the data already loaded previously to train the model
history = model.fit(
    train_generator,
    epochs=num_training_epochs,
    validation_data=val_generator
)

# Save the trained model
model.save(f'{MODEL_NAME}.h5')

# Plot the training and validation loss
def plot_loss(history):
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()

plot_loss(history)

# Model parameters
num_conv_layers = 4
num_dense_layers = 1
layer_size = 32
num_training_epochs = 15
MODEL_NAME = "brain-tumor4"
batch_size = 32

# Directory paths
base_dir = 'Mudau'
train_dir = os.path.join(base_dir, 'train_output')
val_dir = os.path.join(base_dir, 'val_output')
test_dir = os.path.join(base_dir, 'test_output')

# Data generators
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Data generators for training, validation, and test sets
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(256, 256),
    batch_size=batch_size,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(256, 256),
    batch_size=batch_size,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(256, 256),
    batch_size=batch_size,
    class_mode='categorical'
)

# Initiate model variable
model = Sequential()

# Adding properties to model variable
# Add a convolution layer
model.add(Conv2D(layer_size, (3, 3), input_shape=(256, 256, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Add additional layers based on num_conv_layers
for _ in range(num_conv_layers - 1):
    model.add(Conv2D(layer_size, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

# Reduce dimensionality
model.add(Flatten())

# Add fully connected "dense" layers if specified
for _ in range(num_dense_layers):
    model.add(Dense(layer_size))
    model.add(Activation('relu'))

# Output layer
model.add(Dense(4))
model.add(Activation('softmax'))

# Compile the sequential model with all added properties
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# Use the data already loaded previously to train the model
history = model.fit(
    train_generator,
    epochs=num_training_epochs,
    validation_data=val_generator
)

# Save the trained model
model.save(f'{MODEL_NAME}.h5')

# Plot the training and validation loss
def plot_loss(history):
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()

plot_loss(history)
# Model parameters
num_conv_layers = 4
num_dense_layers = 1
layer_size = 32
num_training_epochs = 15
MODEL_NAME = "brain-tumor4"
batch_size = 32

# Directory paths
base_dir = 'Mudau'
train_dir = os.path.join(base_dir, 'train_output')
val_dir = os.path.join(base_dir, 'val_output')
test_dir = os.path.join(base_dir, 'test_output')

# Data generators
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Data generators for training, validation, and test sets
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(256, 256),
    batch_size=batch_size,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(256, 256),
    batch_size=batch_size,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(256, 256),
    batch_size=batch_size,
    class_mode='categorical'
)

# Initiate model variable
model = Sequential()

# Adding properties to model variable
# Add a convolution layer
model.add(Conv2D(layer_size, (3, 3), input_shape=(256, 256, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Add additional layers based on num_conv_layers
for _ in range(num_conv_layers - 1):
    model.add(Conv2D(layer_size, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

# Reduce dimensionality
model.add(Flatten())

# Add fully connected "dense" layers if specified
for _ in range(num_dense_layers):
    model.add(Dense(layer_size))
    model.add(Activation('relu'))

# Output layer
model.add(Dense(4))
model.add(Activation('softmax'))

# Compile the sequential model with all added properties
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# Use the data already loaded previously to train the model
history = model.fit(
    train_generator,
    epochs=num_training_epochs,
    validation_data=val_generator
)

# Save the trained model
model.save(f'{MODEL_NAME}.h5')

# Plot the training and validation loss
def plot_loss(history):
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()

plot_loss(history)

# Model parameters
num_conv_layers = 4
num_dense_layers = 1
layer_size = 32
num_training_epochs = 15
MODEL_NAME = "brain-tumor4"
batch_size = 32

# Directory paths
base_dir = 'Mudau'
train_dir = os.path.join(base_dir, 'train_output')
val_dir = os.path.join(base_dir, 'val_output')
test_dir = os.path.join(base_dir, 'test_output')

# Data generators
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Data generators for training, validation, and test sets
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(256, 256),
    batch_size=batch_size,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(256, 256),
    batch_size=batch_size,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(256, 256),
    batch_size=batch_size,
    class_mode='categorical'
)

# Initiate model variable
model = Sequential()

# Adding properties to model variable
# Add a convolution layer
model.add(Conv2D(layer_size, (3, 3), input_shape=(256, 256, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Add additional layers based on num_conv_layers
for _ in range(num_conv_layers - 1):
    model.add(Conv2D(layer_size, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

# Reduce dimensionality
model.add(Flatten())

# Add fully connected "dense" layers if specified
for _ in range(num_dense_layers):
    model.add(Dense(layer_size))
    model.add(Activation('relu'))

# Output layer
model.add(Dense(4))
model.add(Activation('softmax'))

# Compile the sequential model with all added properties
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# Use the data already loaded previously to train the model
history = model.fit(
    train_generator,
    epochs=num_training_epochs,
    validation_data=val_generator
)

# Save the trained model
model.save(f'{MODEL_NAME}.h5')

# Plot the training and validation loss
def plot_loss(history):
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()

plot_loss(history)

# Model parameters
num_conv_layers = 5
num_dense_layers = 1
layer_size = 32
num_training_epochs = 15
MODEL_NAME = "brain-tumor4"
batch_size = 32

# Directory paths
base_dir = 'Mudau'
train_dir = os.path.join(base_dir, 'train_output')
val_dir = os.path.join(base_dir, 'val_output')
test_dir = os.path.join(base_dir, 'test_output')

# Data generators
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Data generators for training, validation, and test sets
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(256, 256),
    batch_size=batch_size,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(256, 256),
    batch_size=batch_size,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(256, 256),
    batch_size=batch_size,
    class_mode='categorical'
)

# Initiate model variable
model = Sequential()

# Adding properties to model variable
# Add a convolution layer
model.add(Conv2D(layer_size, (3, 3), input_shape=(256, 256, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Add additional layers based on num_conv_layers
for _ in range(num_conv_layers - 1):
    model.add(Conv2D(layer_size, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

# Reduce dimensionality
model.add(Flatten())

# Add fully connected "dense" layers if specified
for _ in range(num_dense_layers):
    model.add(Dense(layer_size))
    model.add(Activation('relu'))

# Output layer
model.add(Dense(4))
model.add(Activation('softmax'))

# Compile the sequential model with all added properties
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# Use the data already loaded previously to train the model
history = model.fit(
    train_generator,
    epochs=num_training_epochs,
    validation_data=val_generator
)

# Save the trained model
model.save(f'{MODEL_NAME}.h5')

# Plot the training and validation loss
def plot_loss(history):
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()

plot_loss(history)

# Model parameters
num_conv_layers = 5
num_dense_layers = 1
layer_size = 16
num_training_epochs = 15
MODEL_NAME = "brain-tumor5"
batch_size = 32

# Directory paths
base_dir = 'Mudau'
train_dir = os.path.join(base_dir, 'train_output')
val_dir = os.path.join(base_dir, 'val_output')
test_dir = os.path.join(base_dir, 'test_output')

# Data generators
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Data generators for training, validation, and test sets
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(256, 256),
    batch_size=batch_size,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(256, 256),
    batch_size=batch_size,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(256, 256),
    batch_size=batch_size,
    class_mode='categorical'
)

# Initiate model variable
model = Sequential()

# Adding properties to model variable
# Add a convolution layer
model.add(Conv2D(layer_size, (3, 3), input_shape=(256, 256, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Add additional layers based on num_conv_layers
for _ in range(num_conv_layers - 1):
    model.add(Conv2D(layer_size, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

# Reduce dimensionality
model.add(Flatten())

# Add fully connected "dense" layers 
for _ in range(num_dense_layers):
    model.add(Dense(layer_size))
    model.add(Activation('relu'))

# Output layer
model.add(Dense(4))
model.add(Activation('softmax'))

# Compile the sequential model with all added properties
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# Use the data already loaded previously to train the model
history = model.fit(
    train_generator,
    epochs=num_training_epochs,
    validation_data=val_generator
)

# Save the trained model
model.save(f'{MODEL_NAME}.h5')

# Plot the training and validation loss
def plot_loss(history):
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()

plot_loss(history)

# Model parameters
num_conv_layers = 5
num_dense_layers = 1
layer_size = 8
num_training_epochs = 30
MODEL_NAME = "brain-tumor4"
batch_size = 32

# Directory paths
base_dir = 'Mudau'
train_dir = os.path.join(base_dir, 'train_output')
val_dir = os.path.join(base_dir, 'val_output')
test_dir = os.path.join(base_dir, 'test_output')

# Data generators
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Data generators for training, validation, and test sets
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(256, 256),
    batch_size=batch_size,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(256, 256),
    batch_size=batch_size,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(256, 256),
    batch_size=batch_size,
    class_mode='categorical'
)

# Initiate model variable
model = Sequential()

# Adding properties to model variable
# Add a convolution layer
model.add(Conv2D(layer_size, (3, 3), input_shape=(256, 256, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Add additional layers based on num_conv_layers
for _ in range(num_conv_layers - 1):
    model.add(Conv2D(layer_size, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

# Reduce dimensionality
model.add(Flatten())

# Add fully connected "dense" 
for _ in range(num_dense_layers):
    model.add(Dense(layer_size))
    model.add(Activation('relu'))

# Output layer
model.add(Dense(4))
model.add(Activation('softmax'))

# Compile the sequential model with all added properties
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# Use the data already loaded previously to train the model
history = model.fit(
    train_generator,
    epochs=num_training_epochs,
    validation_data=val_generator
)

# Save the trained model
model.save(f'{MODEL_NAME}.h5')

# Plot the training and validation loss
def plot_loss(history):
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()

plot_loss(history)




#Confusion matrix for the best performing model
# Load the trained model
MODEL_NAME = "brain-tumor5"
model = load_model(f'{MODEL_NAME}.h5')

# Directory path for the test dataset
base_dir = 'Mudau'
test_dir = os.path.join(base_dir, 'test_output')

# Data generator for the test set
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(256, 256),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# Predict the labels for the test set
Y_pred = model.predict(test_generator)
y_pred = np.argmax(Y_pred, axis=1)

# True labels
y_true = test_generator.classes

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Plot the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=test_generator.class_indices.keys(), yticklabels=test_generator.class_indices.keys())
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Print classification report
class_report = classification_report(y_true, y_pred, target_names=test_generator.class_indices.keys())
print('Classification Report:')
print(class_report)


#Dropout and patience
# Model parameters
num_conv_layers = 5
num_dense_layers = 1
layer_size = 16
num_training_epochs = 15
MODEL_NAME = "brain-tumor"
batch_size = 32

# Directory paths
base_dir = 'Mudau'
train_dir = os.path.join(base_dir, 'train_output')
val_dir = os.path.join(base_dir, 'val_output')
test_dir = os.path.join(base_dir, 'test_output')

# Data generators
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Data generators for training, validation, and test sets
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(256, 256),
    batch_size=batch_size,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(256, 256),
    batch_size=batch_size,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(256, 256),
    batch_size=batch_size,
    class_mode='categorical'
)

# Initiate model variable
model = Sequential()

# Adding properties to model variable
# Add a convolution layer
model.add(Conv2D(layer_size, (3, 3), input_shape=(256, 256, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))  # Adding dropout

# Add additional layers based on num_conv_layers
for _ in range(num_conv_layers - 1):
    model.add(Conv2D(layer_size, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))  # Adding dropout

# Reduce dimensionality
model.add(Flatten())

# Add fully connected "dense" layers
for _ in range(num_dense_layers):
    model.add(Dense(layer_size))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))  # Adding dropout

# Output layer
model.add(Dense(4))
model.add(Activation('softmax'))

# Compile the sequential model with all added properties
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# Define early stopping callback
early_stopping = EarlyStopping(
    monitor='val_loss', 
    patience=2, 
    restore_best_weights=True
)

# Use the data already loaded previously to train the model
history = model.fit(
    train_generator,
    epochs=num_training_epochs,
    validation_data=val_generator,
    callbacks=[early_stopping]  # Add early stopping callback here
)

# Save the trained model
model.save(f'{MODEL_NAME}.h5')


# Extract the training history
history_dict = history.history

# Extract the loss and accuracy for training and validation
train_loss = history_dict['loss']
val_loss = history_dict['val_loss']
train_acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']

# Define the number of epochs
epochs = range(1, len(train_loss) + 1)

# Plot training and validation loss
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs, train_loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss', color = 'red')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()


# Display the plots
plt.show()


# Load the trained model
MODEL_NAME = "brain-tumor5"
model = load_model(f'{MODEL_NAME}.h5')

# Directory path for the test dataset
base_dir = 'Mudau'
test_dir = os.path.join(base_dir, 'test_output')

# Data generator for the test set
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(256, 256),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# Predict the labels for the test set
Y_pred = model.predict(test_generator)
y_pred = np.argmax(Y_pred, axis=1)

# True labels
y_true = test_generator.classes

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Plot the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=test_generator.class_indices.keys(), yticklabels=test_generator.class_indices.keys())
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Print classification report
class_report = classification_report(y_true, y_pred, target_names=test_generator.class_indices.keys())
print('Classification Report:')
print(class_report)


# Load the trained model
MODEL_NAME = "brain-tumor"
model = load_model(f'{MODEL_NAME}.h5')

# Directory path for the test dataset
base_dir = 'Mudau'
test_dir = os.path.join(base_dir, 'test_output')

# Data generator for the test set
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(256, 256),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# Predict the labels for the test set
Y_pred = model.predict(test_generator)
y_pred = np.argmax(Y_pred, axis=1)

# True labels
y_true = test_generator.classes

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Plot the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=test_generator.class_indices.keys(), yticklabels=test_generator.class_indices.keys())
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Print classification report
class_report = classification_report(y_true, y_pred, target_names=test_generator.class_indices.keys())
print('Classification Report:')
print(class_report)


#Predictions
# Load the trained model
model = load_model(f'{MODEL_NAME}.h5')

# Function to predict and display results
def make_prediction(image_fp):
    im = cv2.imread(image_fp)  # Load image
    plt.imshow(im[:, :, [2, 1, 0]])  # OpenCV loads images in BGR format, matplotlib expects RGB
    plt.show()
    
    img = image.load_img(image_fp, target_size=(256, 256))  # Ensure target size matches the model input
    img = image.img_to_array(img)
    
    image_array = img / 255.0  # Scale the image
    img_batch = np.expand_dims(image_array, axis=0)
    
    class_ = ["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"]  # Update this to match your classes
    predicted_value = class_[model.predict(img_batch).argmax()]
    
    true_value_search = re.search(r'(glioma_tumor)|(meningioma_tumor)|(no_tumor)|(pituitary_tumor)', image_fp)
    true_value = true_value_search.group(0) if true_value_search else 'Unknown'
    
    Out = f"""Predicted tumor type: {predicted_value}
True tumor type: {true_value}
Is prediction correct?: {predicted_value == true_value}"""
    
    return Out

image_path = 'Mudau/test_output/no_tumor/no_tumor_54240_0_8660.jpg'
print(make_prediction(image_path))

///
image_path = 'Mudau/test_output/glioma_tumor/glioma_tumor_50620_0_3190.jpg'
print(make_prediction(image_path))

///
image_path = 'Mudau/test_output/meningioma_tumor/meningioma_tumor_53580_0_2447.jpg'
print(make_prediction(image_path))

///
image_path = 'Mudau/test_output/no_tumor/no_tumor_54380_0_5650.jpg'
print(make_prediction(image_path))

///
image_path = 'Mudau/test_output/pituitary_tumor/pituitary_tumor_55040_0_3085.jpg'
print(make_prediction(image_path))
