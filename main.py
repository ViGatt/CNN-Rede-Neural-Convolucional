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

modelo = Sequential([
    # Camada de Convolução 1
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    MaxPooling2D(2, 2),

    # Camada de Convolução 2
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    # Camada de Achatamento e Camadas Densas
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5), # Camada de Dropout para combater overfitting
    Dense(treinamento_dataset.num_classes, activation='softmax') 
])
# Compilar o modelo
modelo.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Treinar o modelo
history = modelo.fit(
    treinamento_dataset,
    steps_per_epoch=len(treinamento_dataset),
    epochs=epochs,  # Usando a variável que você já definiu
    validation_data=validacao_dataset,
    validation_steps=len(validacao_dataset)
)