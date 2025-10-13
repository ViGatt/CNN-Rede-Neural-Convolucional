import tensorflow as tf
import numpy as np
import shutil
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Caminho para a pasta de dados local
train_dir = "data/train" 
validation_dir = "data/test"
img_height, img_width = 224, 224
batch_size = 32
epochs = 30

# Criar gerador de dados para treinamento e validação

train_datagenerator = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

validation_datagenerator = ImageDataGenerator(rescale=1./255)

datagenerator = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2 
)

treinamento_dataset = train_datagenerator.flow_from_directory(
    directory=train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

validacao_dataset = validation_datagenerator.flow_from_directory(
    directory=validation_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

print("Classes encontradas:", treinamento_dataset.class_indices)